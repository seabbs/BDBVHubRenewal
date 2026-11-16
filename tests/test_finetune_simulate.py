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
def test_mid_seeded_places_start_empty_and_are_imported(tmp_path):
    out = run_r(
        """
set.seed(4)
strata <- data.table::data.table(
  .group = 1:3, place = c("P01", "P02", "P03"), age = "A1"
)
d <- draw_design(0)
d$n_places <- 3L; d$seeding <- "mid"; d$imports <- "none"
v <- draw_values(d)
v$quiet_places <- integer(0); v$late_places <- 2L
n <- 100
share <- c(0.5, 0.3, 0.2)
seed <- stratum_seed(share, strata, d, v)
imp <- stratum_imports(n, share, strata, d, v)
from <- max(1L, round(v$late_at * n))
cat(seed, "\\n")
cat(sum(imp[, 1]), sum(imp[, 3]), round(sum(imp[, 2]), 10), "\\n")
cat(all(imp[seq_len(from - 1L), 2] == 0), imp[from, 2] > 0, "\\n")
d$seeding <- "start"
cat(stratum_seed(share, strata, d, v),
  sum(stratum_imports(n, share, strata, d, v)), "\\n")
""",
        tmp_path,
    )
    seed, totals, timing, start = out.strip().splitlines()
    # A late place has no seed and is imported its share over its seed days.
    assert seed.split() == ["0.5", "0", "0.2"]
    assert totals.split() == ["0", "0", "0.3"]
    assert timing.split() == ["TRUE", "TRUE"]
    assert start.split() == ["0.5", "0.3", "0.2", "0"]


@needs_r
def test_injected_latent_reproduces_lambda_exactly(tmp_path):
    out = run_r(
        """
set.seed(7)
dates <- as.IDate("2026-01-03") + 0:59
grid <- CJ(place = c("P01", "P02", "P03"), age = "A1",
  reference_date = dates, delay = 0:4)
grid[, report_date := reference_date + delay][, delay := NULL]
grid <- grid[report_date <= max(dates)][, confirm := 0L]
pobs <- suppressMessages(enw_preprocess_data(grid, by = c("place", "age"),
  max_delay = 5))
strata <- unique(pobs$metareference[[1]][, .(.group, place, age)])
gt <- gt_pmf(7, "gamma_0.6")
n_r <- length(dates) - length(gt)
log_r <- matrix(rnorm(n_r * 3, 0.02, 0.05), n_r, 3)
d <- draw_design(0); d$n_places <- 3L; d$n_ages <- 1L
d$mixing <- "symmetric"; d$imports <- "constant"; d$seeding <- "start"
v <- draw_values(d); v$quiet_places <- integer(0); v$mixing_eps <- 0.3
seed <- c(20, 0, 5)
lambda <- expected_onsets(seed, gt, log_r,
  mixing = place_mixing(n_r, strata, d, v),
  imports = 3 * stratum_imports(n_r, seed / sum(seed), strata, d, v))
latent_of <- function(sims) {
  m <- sims$fit[[1]]$draws("exp_llatent", format = "matrix")
  idx <- do.call(rbind, regmatches(colnames(m),
    gregexpr("[0-9]+", colnames(m))))
  out <- matrix(NA_real_, max(as.integer(idx[, 2])), 3)
  out[cbind(as.integer(idx[, 2]), as.integer(idx[, 1]))] <- exp(m[1, ])
  out
}
run <- function(expectation, growth_rate, lelatent) {
  suppressMessages(suppressWarnings(enw_simulate(pobs,
    growth_rate = growth_rate,
    parameters = list(expr_lelatent_int = lelatent, sqrt_phi = array(0.5)),
    reference = enw_reference(~1, distribution = "lognormal", data = pobs),
    expectation = expectation, model = model, draws = 1L, refresh = 0,
    show_messages = FALSE, sig_figs = 18)))
}
inj <- latent_expectation(pobs, lambda, ~1)
got <- latent_of(run(inj$expectation, inj$growth_rate, inj$lelatent))
cat("rows", nrow(got) == nrow(lambda), "\\n")
cat("mixed", max(abs(got / pmax(lambda, 1e-8) - 1)) < 1e-10, "\\n")
# With no mixing or imports, the injected path matches epinowcast's own
# renewal driven by the same Rt.
seed <- c(20, 1, 5)
plain <- expected_onsets(seed, gt, log_r)
inj <- latent_expectation(pobs, plain, ~1)
got <- latent_of(run(inj$expectation, inj$growth_rate, inj$lelatent))
ren <- enw_expectation(r = ~1, generation_time = gt, data = pobs)
own <- latent_of(run(ren, as.numeric(log_r),
  matrix(log(seed), length(gt), 3, byrow = TRUE)))
cat("renewal", max(abs(got / own - 1)) < 1e-10, "\\n")
""",
        tmp_path,
    )
    rows = dict(line.split() for line in out.strip().splitlines())
    assert rows == {"rows": "TRUE", "mixed": "TRUE", "renewal": "TRUE"}


@needs_r
def test_each_mixing_kind_gives_its_cross_place_structure(tmp_path):
    out = run_r(
        """
set.seed(8)
strata <- data.table::data.table(
  .group = 1:4, place = sprintf("P%02d", 1:4), age = "A1"
)
d <- draw_design(0); d$n_places <- 4L; d$n_ages <- 1L
v <- draw_values(d)
v$quiet_places <- integer(0); v$mixing_eps <- 0.3
n <- 90
gt <- gt_pmf(7, "gamma_0.6")
log_r <- matrix(0, n, 4)
onsets <- function(kind, seed) {
  d$mixing <- kind
  expected_onsets(seed, gt, log_r, mixing = place_mixing(n, strata, d, v))
}
first <- function(x) apply(x, 2, function(y) which(y > 1e-3)[1])
o <- onsets("none", c(1, 0, 0, 0))
cat("none", all(o[, 2:4] == 0), "\\n")
d$mixing <- "symmetric"
m <- place_mixing(n, strata, d, v)
o <- onsets("symmetric", c(1, 0, 0, 0))
cat("symmetric", isSymmetric(m$K), all(o[n, 2:4] > 0),
  isTRUE(all.equal(o[, 2], o[, 4])), "\\n")
ord <- v$mixing_order
seed <- numeric(4); seed[ord[4]] <- 1
o_last <- onsets("asymmetric", seed)
seed <- numeric(4); seed[ord[1]] <- 1
o_first <- onsets("asymmetric", seed)
cat("asymmetric", all(o_last[, ord[1:3]] == 0), all(o_first[n, ] > 0),
  all(diff(first(o_first)[ord]) >= 0), "\\n")
v$mixing_x <- c(0, 0.1, 0.5, 1); v$mixing_y <- rep(0, 4)
v$mixing_range <- 0.2
o <- onsets("distance", c(1, 0, 0, 0))
cs <- colSums(o)
cat("distance", cs[2] > cs[3], cs[3] > cs[4], all(diff(first(o)[2:4]) >= 0),
  "\\n")
v$mixing_switch <- "on"; v$mixing_switch_at <- 0.5
d$mixing <- "time_varying"
m <- place_mixing(n, strata, d, v)
on <- which(m$eps > 0)[1]
o <- onsets("time_varying", c(1, 0, 0, 0))
cat("time_on", m$eps[1] == 0, m$eps[n] == v$mixing_eps,
  all(o[seq_len(length(gt) + on - 1L), 2:4] == 0), all(o[n, 2:4] > 0), "\\n")
v$mixing_switch <- "off"
m <- place_mixing(n, strata, d, v)
cat("time_off", m$eps[1] == v$mixing_eps, m$eps[n] == 0, "\\n")
v$quiet_places <- 4L
o <- onsets("symmetric", c(1, 0, 0, 0))
cat("quiet", all(o[, 4] == 0), all(o[n, 2:3] > 0), "\\n")
""",
        tmp_path,
    )
    rows = {r.split()[0]: r.split()[1:] for r in out.strip().splitlines()}
    assert set(rows) == {
        "none",
        "symmetric",
        "asymmetric",
        "distance",
        "time_on",
        "time_off",
        "quiet",
    }
    assert all(x == "TRUE" for r in rows.values() for x in r), rows


@needs_r
def test_imports_appear_in_active_strata_only(tmp_path):
    out = run_r(
        """
set.seed(9)
strata <- data.table::data.table(
  .group = 1:3, place = c("P01", "P02", "P03"), age = "A1"
)
d <- draw_design(0); d$n_places <- 3L; d$n_ages <- 1L
d$seeding <- "start"; d$mixing <- "none"
v <- draw_values(d); v$quiet_places <- 3L
n <- 80
gt <- gt_pmf(7, "gamma_0.6")
log_r <- matrix(-0.02, n, 3)
share <- c(0.6, 0.4, 0)
for (kind in c("none", "constant", "pulses")) {
  d$imports <- kind
  imp <- stratum_imports(n, share, strata, d, v)
  o <- expected_onsets(numeric(3), gt, log_r, imports = imp)
  days <- which(imp[, 1] > 0)
  cat(kind, sum(imp[, 1:2]) > 0, all(imp[, 3] == 0), all(o[, 3] == 0),
    all(o[length(gt) + days, 1] >= imp[days, 1]),
    length(days), "\\n")
}
""",
        tmp_path,
    )
    rows = {r.split()[0]: r.split()[1:] for r in out.strip().splitlines()}
    assert rows["none"][:3] == ["FALSE", "TRUE", "TRUE"]
    assert rows["none"][4] == "0"
    for kind in ("constant", "pulses"):
        assert rows[kind][:4] == ["TRUE"] * 4
    assert rows["constant"][4] == "80"
    assert 1 <= int(rows["pulses"][4]) <= 4


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
late <- counts[location == "P02" & reference_date >= seed_day, sum(count)]
cat("late", late, "\\n")
""",
        tmp_path,
    )
    lines = out.strip().splitlines()
    # Late places are drawn among the active ones.
    assert all(x.split()[-1] == "TRUE" for x in lines if x.startswith("drawn"))
    late = next(x for x in lines if x.startswith("late"))
    assert float(late.split()[-1]) > 0


@needs_r
def test_settings_hold_one_row_per_simulation(tmp_path):
    out = run_r(
        """
set.seed(6)
rows <- lapply(1:3, function(i) {
  d <- draw_design(i)
  d$n_places <- 5L
  flat_values(c(list(outbreak = i), d, draw_values(d)))
})
s <- data.table::rbindlist(rows, fill = TRUE)
cat(nrow(s), is.character(s$place_lags), "\\n")
""",
        tmp_path,
    )
    assert out.split() == ["3", "TRUE"]


@needs_r
def test_failures_hold_one_row_per_try(tmp_path):
    out = run_r(
        """
set.seed(8)
d <- draw_design(1L)
d$n_places <- 5L
rows <- lapply(1:2, function(i) {
  failure_row("x", i, "too small", d, draw_values(d))
})
cat(nrow(data.table::rbindlist(rows, fill = TRUE)), "\\n")
""",
        tmp_path,
    )
    assert out.split() == ["2"]


SHOCK = """
set.seed(7)
d <- draw_design(1L)
d$strata <- "places"; d$n_places <- 2L; d$n_ages <- 1L
d$spread <- "even"; d$seeding <- "start"; d$place_rt <- "shared"
d$trajectory <- "flat"; d$delay_change <- "static"
d$delay_form <- "parametric"; d$delay_family <- "lognormal"
d$reporting <- "daily"; d$shocks <- "{level}"
v <- draw_values(d)
v$weeks <- 16L; v$target_size <- 6000; v$quiet_places <- integer(0)
v$delay_median <- 3; v$delay_sdlog <- 0.6; v$flat <- 0
v$shock_places <- 1L; v$shock_at <- 0.4; v$shock_weeks <- 3L
v$shock_mult <- 4
out <- simulate_outbreak(d, v)
c <- out$counts[draw == 1 & location == "P01"]
w <- shock_window(out$counts, v)
inside <- c[reference_date >= w[1] & reference_date <= w[2]]
outside <- c[reference_date < w[1] - 14]
delay <- function(x) {
  x[, sum(count * as.numeric(report_date - reference_date)) / sum(count)]
}
cat("delay", delay(inside), delay(outside), "\\n")
during <- c[report_date >= w[1] & report_date <= w[2], sum(count)]
cat("reported_in_window", during, "\\n")
"""


@needs_r
def test_a_backlog_lengthens_one_place_s_delays_for_a_while(tmp_path):
    out = run_r(SHOCK.replace("{level}", "backlog"), tmp_path)
    line = next(x for x in out.splitlines() if x.startswith("delay"))
    inside, outside = map(float, line.split()[1:])
    assert inside > 2 * outside


@needs_r
def test_an_outage_stops_one_place_s_reports_for_a_while(tmp_path):
    out = run_r(SHOCK.replace("{level}", "outage"), tmp_path)
    line = next(x for x in out.splitlines() if x.startswith("reported"))
    assert float(line.split()[-1]) == 0


# A design from the grid whose cell has the given levels, run through the
# batch's own tries (`simulate_cell`): it gives an outbreak within 5 tries.
CELL_RUNS = """
want <- list({levels})
g <- 0L
repeat {{
  cell <- cell_for(g)
  if (all(vapply(names(want), function(k) cell[[k]] == want[[k]], TRUE))) break
  g <- g + 1L
}}
ok <- 0L
for (k in 1:3) {{
  set.seed(k)
  design <- draw_design(g)
  res <- simulate_cell(design)
  if (!is.null(res$out) && sum(res$out$counts$count) > 0) ok <- ok + 1L
}}
cat("ok", ok, "\\n")
"""


@needs_r
@pytest.mark.parametrize(
    "levels",
    [
        'strata = "places", spread = "concentrated", seeding = "mid"',
        'strata = "places", mixing = "asymmetric", seeding = "mid", '
        'imports = "constant"',
        'strata = "single", reporting = "weekdays", seeding = "mid"',
    ],
)
def test_each_cell_gives_an_outbreak_within_five_tries(levels, tmp_path):
    out = run_r(CELL_RUNS.format(levels=levels), tmp_path)
    line = next(x for x in out.splitlines() if x.startswith("ok"))
    assert line.split()[1] == "3"


@needs_r
def test_a_place_with_cases_starts_at_the_outset(tmp_path):
    out = run_r(
        """
g <- 0L
repeat {
  cell <- cell_for(g)
  if (cell$strata == "places" && cell$spread == "concentrated" &&
      cell$seeding == "mid" && cell$trajectory == "growth") break
  g <- g + 1L
}
set.seed(3)
design <- draw_design(g)
bad <- 0L
for (k in 1:20) {
  v <- draw_values(design)
  # No place seeded mid-outbreak is the only place with cases.
  cases <- setdiff(seq_len(design$n_places), v$quiet_places)
  if (all(cases %in% v$late_places)) bad <- bad + 1L
}
cat("lonely_late", bad, "\\n")
""",
        tmp_path,
    )
    line = next(x for x in out.splitlines() if x.startswith("lonely_late"))
    assert line.split()[1] == "0"


# Places' mean delays under each relationship between places' delays.
DELAY_PLACES = """
set.seed(11)
d <- draw_design(1L)
d$strata <- "places"; d$n_places <- 4L; d$n_ages <- 1L
d$spread <- "even"; d$seeding <- "start"; d$place_rt <- "shared"
d$mixing <- "none"; d$imports <- "none"
d$trajectory <- "flat"; d$delay_change <- "static"
d$delay_form <- "parametric"; d$delay_family <- "lognormal"
d$reporting <- "daily"; d$shocks <- "none"; d$delay_places <- "{level}"
v <- draw_values(d)
v$weeks <- 16L; v$target_size <- 20000; v$quiet_places <- integer(0)
v$delay_median <- 4; v$delay_sdlog <- 0.6; v$flat <- 0
v$delay_place_sd <- 0.8
out <- simulate_outbreak(d, v)
c <- out$counts[draw == 1]
m <- c[, .(d = sum(count * as.numeric(report_date - reference_date)) /
  sum(count)), by = location][order(location), d]
cat("means", m, "\\n")
"""


def _place_means(level: str, tmp_path: Path) -> list[float]:
    out = run_r(DELAY_PLACES.replace("{level}", level), tmp_path)
    line = next(x for x in out.splitlines() if x.startswith("means"))
    return [float(x) for x in line.split()[1:]]


@needs_r
def test_shared_delays_are_the_same_in_every_place(tmp_path):
    means = _place_means("shared", tmp_path)
    assert len(means) == 4
    assert max(means) / min(means) < 1.15


@needs_r
@pytest.mark.parametrize("level", ["shifted", "independent"])
def test_shifted_and_independent_delays_differ_by_place(level, tmp_path):
    means = _place_means(level, tmp_path)
    assert max(means) / min(means) > 1.3


@needs_r
def test_delay_places_is_a_balanced_factor(tmp_path):
    out = run_r(
        'f <- factors$delay_places\ncat(f, block %% length(f), "\\n")',
        tmp_path,
    )
    assert out.split() == ["shared", "shifted", "independent", "0"]


# Observation noise by stratum: each stratum's variance over mean across
# draws of the same expected reports.
OBS_STRATA = """
draws <- 40L
set.seed(5)
d <- draw_design(1L)
d$strata <- "places"; d$n_places <- 4L; d$n_ages <- 1L
d$spread <- "even"; d$seeding <- "start"; d$place_rt <- "shared"
d$mixing <- "none"; d$imports <- "none"; d$shocks <- "none"
d$trajectory <- "flat"; d$delay_change <- "static"
d$delay_form <- "parametric"; d$delay_family <- "lognormal"
d$reporting <- "daily"; d$report_days <- "all"; d$delay_places <- "shared"
d$observation <- "negbin"; d$obs_strata <- "{level}"
v <- draw_values(d)
v$weeks <- 16L; v$target_size <- 40000; v$quiet_places <- integer(0)
v$delay_median <- 3; v$delay_sdlog <- 0.6; v$flat <- 0
{set_values}
out <- simulate_outbreak(d, v)
c <- out$counts
full <- CJ(location = unique(c$location), reference_date =
  unique(c$reference_date), report_date = unique(c$report_date),
  draw = seq_len(draws))
c <- c[full, on = .(location, reference_date, report_date, draw)]
c[is.na(count), count := 0L]
s <- c[, .(m = mean(count), v = var(count)),
  by = .(location, reference_date, report_date)][m > 20]
r <- s[, .(ratio = median((v - m) / m^2)), by = location][order(location)]
cat("excess", r$ratio, "\\n")
cat("families", out$values$obs_family, "\\n")
"""


def _obs(level: str, set_values: str, tmp_path: Path) -> dict[str, list]:
    out = run_r(
        OBS_STRATA.replace("{level}", level).replace(
            "{set_values}", set_values
        ),
        tmp_path,
    )
    rows = {}
    for x in out.splitlines():
        if x.startswith(("excess", "families")):
            k, *vals = x.split()
            rows[k] = vals
    return rows


@needs_r
def test_dispersion_differs_by_stratum_as_set(tmp_path):
    # sqrt_phi 0.1 (near Poisson) in places 1 and 3, 1 (phi = 1) in 2, 4.
    rows = _obs("dispersion", "v$obs_sqrt_phi <- c(0.1, 1, 0.1, 1)", tmp_path)
    excess = [float(x) for x in rows["excess"]]
    assert len(excess) == 4
    # NB2: (var - mean) / mean^2 is 1 / phi = sqrt_phi^2.
    assert excess[1] > 0.6
    assert excess[3] > 0.6
    assert excess[0] < 0.1
    assert excess[2] < 0.1


@needs_r
def test_families_differ_by_stratum_and_are_recorded(tmp_path):
    rows = _obs(
        "families",
        'v$obs_family <- c("poisson", "negbin", "poisson", "negbin")\n'
        "v$obs_sqrt_phi <- rep(1, 4)",
        tmp_path,
    )
    excess = [float(x) for x in rows["excess"]]
    assert rows["families"] == ["poisson", "negbin", "poisson", "negbin"]
    assert abs(excess[0]) < 0.1
    assert abs(excess[2]) < 0.1
    assert excess[1] > 0.6
    assert excess[3] > 0.6


@needs_r
def test_obs_strata_is_a_balanced_factor(tmp_path):
    out = run_r(
        'f <- factors$obs_strata\ncat(f, block %% length(f), "\\n")',
        tmp_path,
    )
    assert out.split() == ["shared", "dispersion", "families", "0"]
