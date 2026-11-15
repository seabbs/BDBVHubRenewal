"""The fine-tune simulator's factorial design and its delay forms."""

import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).parents[1]
LIB = ROOT / "output" / "finetune" / "rlib"

# Loads the simulator's functions without running a batch.
SOURCE = r"""
ex <- parse("scripts/finetune_simulate.R")
for (e in ex) {
  txt <- paste(deparse(e), collapse = "")
  if (grepl("commandArgs|args\\[\\[|^if \\(sys.nframe", txt)) next
  if (grepl("^lib <-", txt)) { lib <- "output/finetune/rlib"; next }
  eval(e, globalenv())
}
draws <- 2L
"""

needs_r = pytest.mark.skipif(
    not LIB.exists() or shutil.which("Rscript") is None,
    reason="needs R and epinowcast's simulate branch in output/finetune/rlib",
)


def run_r(script: str, tmp_path: Path) -> str:
    """Run R code after loading the simulator and return its output."""
    path = tmp_path / "run.R"
    path.write_text(SOURCE + script)
    out = subprocess.run(
        ["Rscript", str(path)],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
        timeout=900,
    )
    return out.stdout


@needs_r
def test_every_factor_level_appears_equally_in_a_block(tmp_path):
    out = run_r(
        """
cells <- lapply(0:(block - 1L), cell_for)
for (f in names(factors)) {
  counts <- table(vapply(cells, `[[`, "", f))
  cat(f, length(unique(counts)) == 1L, "\\n")
}
""",
        tmp_path,
    )
    assert all(line.split()[-1] == "TRUE" for line in out.strip().splitlines())


@needs_r
@pytest.mark.parametrize("form", ["parametric", "non_parametric", "both"])
def test_each_delay_form_simulates(form, tmp_path):
    out = run_r(
        f"""
set.seed(3)
d <- draw_design(1L)
d$strata <- "single"; d$n_places <- 1L; d$n_ages <- 1L
d$delay_form <- "{form}"; d$delay_change <- "walk"
v <- draw_values(d); v$weeks <- 16L
cat("cases", sum(simulate_outbreak(d, v)$counts$count), "\\n")
""",
        tmp_path,
    )
    assert int(out.split("cases")[-1].split()[0]) > 0


@needs_r
def test_places_by_ages_always_have_three_or_more_ages(tmp_path):
    out = run_r(
        """
set.seed(1)
ok <- TRUE
for (g in 0:2000) {
  d <- draw_design(g)
  if (d$strata == "places_ages") {
    ok <- ok && d$n_ages >= 3 && d$n_places * d$n_ages <= 60
  }
}
cat("ok", ok, "\\n")
""",
        tmp_path,
    )
    assert out.split()[-1] == "TRUE"


@needs_r
def test_batches_record_the_stack_they_ran_on(tmp_path):
    out = run_r(
        "s <- build_stack(lib)\n"
        "cat(paste(names(s), unlist(s), sep = '=', collapse = '\\n'))\n",
        tmp_path,
    )
    stack = dict(line.split("=", 1) for line in out.strip().splitlines())
    assert set(stack) == {
        "r_version",
        "blas",
        "lapack",
        "enw_fingerprint",
        "host",
    }
    assert stack["r_version"].startswith("R version")
    assert len(stack["enw_fingerprint"]) == 32
    assert all(stack.values())


@needs_r
def test_concentrated_outbreaks_have_few_active_places_among_many(tmp_path):
    out = run_r(
        """
set.seed(1)
g <- which(vapply(0:(block - 1L), function(i) {
  cell_for(i)$spread == "concentrated" && cell_for(i)$strata == "places"
}, TRUE))[1] - 1L
d <- draw_design(g)
for (i in 1:50) {
  d <- draw_design(g)
  v <- draw_values(d)
  active <- d$n_places * (1 - v$quiet)
  cat(d$n_places >= 15, active >= 1 - 1e-9, active <= 4 + 1e-9,
    v$stratum_sd >= 2.5, v$target_size >= 30 * round(active),
    v$target_size <= 3000 * round(active), "\\n")
}
""",
        tmp_path,
    )
    assert out.split()
    assert all(x == "TRUE" for x in out.split())


@needs_r
def test_outbreaks_run_long_enough_for_deployed_contexts(tmp_path):
    out = run_r(
        """
set.seed(2)
weeks <- vapply(0:239, function(g) {
  d <- draw_design(g)
  c(d$strata == "places_ages", draw_values(d)$weeks)
}, numeric(2))
cat(max(weeks[2, weeks[1, ] == 0]), max(weeks[2, weeks[1, ] == 1]),
  min(weeks[2, ]), "\\n")
""",
        tmp_path,
    )
    longest, longest_by_ages, shortest = map(int, out.split())
    # BDBV's contexts reach 40 weeks: blocks are cut up to a week before
    # the end of the onsets, so outbreaks run to 44 weeks.
    assert longest == 44
    assert longest_by_ages == 28
    assert shortest == 16


@needs_r
def test_places_rt_relate_as_their_level_says(tmp_path):
    out = run_r(
        """
set.seed(3)
strata <- data.table::data.table(
  .group = 1:6, place = rep(c("P01", "P02", "P03"), each = 2),
  age = rep(c("A1", "A2"), 3)
)
d <- draw_design(0)
d$trajectory <- "rise_fall_sharp"
d$n_places <- 3L
v <- draw_values(d)
v$coupling <- 1  # no stratum noise, so places' paths show
n <- 120
for (level in c("shared", "independent", "correlated", "lagged")) {
  d$place_rt <- level
  g <- place_growth(n, strata, d, v)
  same_age <- isTRUE(all.equal(g[, 1], g[, 2]))
  same_place <- isTRUE(all.equal(g[, 1], g[, 3]))
  shift <- v$place_lags[2]
  lagged <- shift > 0 && isTRUE(all.equal(
    g[(shift + 1):n, 3], g[1:(n - shift), 1]
  ))
  cat(level, same_age, same_place, lagged, "\\n")
}
""",
        tmp_path,
    )
    rows = {r.split()[0]: r.split()[1:] for r in out.strip().splitlines()}
    # Ages share their place's path at every level.
    assert all(r[0] == "TRUE" for r in rows.values())
    assert rows["shared"][1] == "TRUE"
    assert rows["independent"][1] == "FALSE"
    assert rows["correlated"][1] == "FALSE"
    # A lagged place follows the first with its own delay.
    assert rows["lagged"][1:] == ["FALSE", "TRUE"]


@needs_r
def test_late_places_start_tiny_and_jump_mid_outbreak(tmp_path):
    out = run_r(
        """
set.seed(4)
strata <- data.table::data.table(
  .group = 1:3, place = c("P01", "P02", "P03"), age = "A1"
)
d <- draw_design(0)
d$n_places <- 3L
d$seeding <- "mid"
v <- draw_values(d)
g <- matrix(0, 100, 3)
late <- seed_late(g, strata, d, v)
jump <- colSums(late$growth - g)
cat(round(jump, 4), late$seed, "\\n")
d$seeding <- "start"
start <- seed_late(g, strata, d, v)
cat(all(start$growth == g), start$seed, "\\n")
""",
        tmp_path,
    )
    mid, start = out.strip().splitlines()
    values = [float(x) for x in mid.split()]
    jumps, seeds = values[:3], values[3:]
    # The first place starts as usual; late places start a thousandth as
    # large and grow by log(1000) mid-outbreak.
    assert jumps[0] == 0
    assert seeds[0] == 1
    late = [j for j in jumps[1:] if j > 0]
    assert late
    assert all(abs(j - 6.9078) < 1e-3 for j in late)
    pairs = zip(jumps[1:], seeds[1:], strict=True)
    assert all(s == 1e-3 for j, s in pairs if j > 0)
    assert start.split() == ["TRUE", "1", "1", "1"]


@needs_r
def test_every_mid_seeded_place_has_cases_after_its_seed_day(tmp_path):
    out = run_r(
        """
set.seed(5)
d <- draw_design(1L)
d$strata <- "places"; d$n_places <- 15L; d$n_ages <- 1L
d$spread <- "concentrated"; d$seeding <- "mid"
d$place_rt <- "shared"; d$trajectory <- "flat"
for (i in 1:40) {
  v <- draw_values(d)
  cat("drawn", all(!v$late_places %in% v$quiet_places), "\\n")
}
v$weeks <- 20L; v$target_size <- 3000
v$late_places <- 2L; v$quiet_places <- setdiff(3:15, 4L)
out <- simulate_outbreak(d, v)
counts <- out$counts[draw == 1]
dates <- sort(unique(counts$reference_date))
seed_day <- dates[1] + round(v$late_at * length(dates))
late <- counts[place == "P02" & reference_date >= seed_day, sum(count)]
cat("late", late, "\\n")
""",
        tmp_path,
    )
    lines = out.strip().splitlines()
    # Late places are drawn among the active ones.
    assert all(x.split()[-1] == "TRUE" for x in lines if x.startswith("drawn"))
    late = [x for x in lines if x.startswith("late")][0]
    assert float(late.split()[-1]) > 0
