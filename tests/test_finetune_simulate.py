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
def test_every_factor_level_appears_at_its_weight_in_a_block(tmp_path):
    # A level listed k times in its factor takes k shares of the block.
    out = run_r(
        """
cells <- lapply(0:(block - 1L), cell_for)
for (f in names(factors)) {
  got <- table(vapply(cells, `[[`, "", f))
  want <- table(rep_len(factors[[f]], block))
  cat(f, block %% length(factors[[f]]) == 0L &&
    identical(as.vector(got[names(want)]), as.vector(want)), "\\n")
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
    ok <- ok && d$n_ages >= 3 &&
      d$n_places * d$n_ages <= max_strata(d$weeks)
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
  cases <- d$n_places - length(v$quiet_places)
  cat(d$n_places >= 15, active >= 1 - 1e-9, active <= 4 + 1e-9,
    v$stratum_sd >= 2.5, v$target_size >= 30 * cases,
    v$target_size <= 3e4 * cases, "\\n")
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
weeks <- vapply(0:839, function(g) {
  d <- draw_design(g)
  c(d$strata == "places_ages", draw_values(d)$weeks)
}, numeric(2))
cat(max(weeks[2, weeks[1, ] == 0]), max(weeks[2, weeks[1, ] == 1]),
  min(weeks[2, ]), "\\n")
""",
        tmp_path,
    )
    longest, longest_by_ages, shortest = map(int, out.split())
    # Every strata structure runs up to a year, for contexts of 364 days.
    assert longest == 52
    assert longest_by_ages == 52
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
v$stratum_sd <- 0
v$weeks <- 16L; v$target_size <- 2e5; v$quiet_places <- integer(0)
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
  by = .(location, reference_date, report_date)][m > 10]
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
def test_obs_strata_is_a_balanced_factor(tmp_path):
    out = run_r(
        'f <- factors$obs_strata\ncat(f, block %% length(f), "\\n")',
        tmp_path,
    )
    assert out.split() == ["shared", "dispersion", "0"]


@needs_r
def test_independent_strata_are_separate_simulations(tmp_path):
    out = run_r(
        """
g <- 0L
repeat {
  cell <- cell_for(g)
  if (cell$strata == "places" && cell$strata_link == "independent" &&
      cell$spread == "even") break
  g <- g + 1L
}
set.seed(2)
design <- draw_design(g)
res <- simulate_cell(design)
c <- res$out$counts
v <- res$out$values
cat("n", design$n_places, "\\n")
cat("locations", sort(unique(c$location)), "\\n")
cat("cells", length(strsplit(v$stratum_cells, ";")[[1]]), "\\n")
cat("trajectories", length(strsplit(v$stratum_trajectory, ";")[[1]]), "\\n")
cat("observation", strsplit(v$stratum_observation, ";")[[1]], "\\n")
cat("family", design$observation, "\\n")
cat("drawn", vapply(res$strata_values, function(x) x$obs_family, ""), "\\n")
""",
        tmp_path,
    )
    rows = {x.split()[0]: x.split()[1:] for x in out.splitlines() if x}
    n = int(rows["n"][0])
    assert 2 <= n <= 8
    # One simulation per place with cases (quiet places have none).
    assert 1 <= len(rows["locations"]) <= n
    assert int(rows["cells"][0]) == len(rows["locations"])
    assert int(rows["trajectories"][0]) == len(rows["locations"])
    # The family each stratum drew: the outbreak's, in every stratum.
    assert rows["observation"] == rows["drawn"]
    assert set(rows["observation"]) == {rows["family"][0]}


@needs_r
def test_strata_link_is_a_balanced_factor(tmp_path):
    out = run_r(
        'f <- factors$strata_link\ncat(f, block %% length(f), "\\n")',
        tmp_path,
    )
    assert out.split() == ["linked", "independent", "0"]


@needs_r
def test_concentrated_independent_strata_simulate_only_active_places(tmp_path):
    out = run_r(
        """
g <- 0L
repeat {
  cell <- cell_for(g)
  if (cell$strata == "places" && cell$strata_link == "independent" &&
      cell$spread == "concentrated") break
  g <- g + 1L
}
set.seed(4)
design <- draw_design(g)
res <- simulate_cell(design)
c <- res$out$counts
cat("n", design$n_places, "\\n")
cat("active", length(unique(c$location)), "\\n")
cat("cells", length(strsplit(res$out$values$stratum_cells, ";")[[1]]), "\\n")
""",
        tmp_path,
    )
    rows = {x.split()[0]: x.split()[1:] for x in out.splitlines() if x}
    assert int(rows["n"][0]) >= 15
    assert 1 <= int(rows["active"][0]) <= 4
    assert int(rows["cells"][0]) == int(rows["active"][0])


@needs_r
def test_a_batch_writes_settings_of_linked_and_independent_outbreaks(tmp_path):
    out = run_r(
        """
linked <- flat_values(list(outbreak = "1-0001", strata = "places"))
independent <- flat_values(list(
  outbreak = "1-0002", strata = "places", stratum_cells = "4;9"
))
s <- settings_table(list(linked, independent))
cat("rows", nrow(s), "cols", ncol(s), "\\n")
""",
        tmp_path,
    )
    assert out.split() == ["rows", "2", "cols", "3"]


@needs_r
def test_sizes_reach_thirty_thousand_cases_per_active_stratum(tmp_path):
    out = run_r(
        """
set.seed(8)
ratio <- c()
fits <- TRUE
for (want in list(
  list(strata = "single"),
  list(strata = "places", spread = "even"),
  list(strata = "places", spread = "concentrated")
)) {
  g <- 0L
  repeat {
    cell <- cell_for(g)
    ok <- vapply(names(want), function(k) cell[[k]] == want[[k]], TRUE)
    if (all(ok)) break
    g <- g + 1L
  }
  d <- draw_design(g)
  for (k in 1:300) {
    v <- draw_values(d)
    active <- max(1, (d$n_places - length(v$quiet_places)) * d$n_ages)
    ratio <- c(ratio, v$target_size / active)
    fits <- fits && max_size(d) >= 3 * v$target_size
  }
}
cat("range", min(ratio), max(ratio), "\\n")
cat("upper_share", mean(ratio > 1e4), "\\n")
cat("fits", fits, "\\n")
cat("grid", grid_name, "\\n")
""",
        tmp_path,
    )
    rows = {x.split()[0]: x.split()[1:] for x in out.splitlines() if x}
    lo, hi = map(float, rows["range"])
    assert lo >= 30
    assert hi <= 30000
    assert hi > 20000
    # Log-uniform on 30 to 30,000: about a sixth above 10,000.
    assert float(rows["upper_share"][0]) > 0.08
    assert rows["fits"] == ["TRUE"]
    assert rows["grid"] == ["r5"]


@needs_r
def test_observation_is_nb1_four_times_as_often_as_poisson(tmp_path):
    out = run_r(
        """
set.seed(5)
cat("levels", round(table(rep_len(factors$observation, block)) / block, 2),
  names(table(factors$observation)), "\\n")
d <- draw_design(1L)
d$n_places <- 50L; d$n_ages <- 1L; d$obs_strata <- "dispersion"
d$observation <- "poisson"
cat("families", unique(draw_values(d)$obs_family), "\\n")
phi <- vapply(1:3000, function(i) draw_values(d)$sqrt_phi, 1)
cat("sqrt_phi", min(phi), max(phi), mean(phi < 0.1), "\\n")
""",
        tmp_path,
    )
    rows = {x.split()[0]: x.split()[1:] for x in out.splitlines() if x}
    assert rows["levels"] == ["0.8", "0.2", "negbin1d", "poisson"]
    # One family per outbreak.
    assert rows["families"] == ["poisson"]
    lo, hi, low = map(float, rows["sqrt_phi"])
    # Log-uniform on 0.02 to 1: about 41% below 0.1.
    assert lo >= 0.02
    assert hi <= 1
    assert 0.35 < low < 0.47


@needs_r
def test_multi_wave_waves_differ_in_size(tmp_path):
    out = run_r(
        """
set.seed(1)
d <- draw_design(1L)
v <- draw_values(d)
v$waves <- 2L; v$phase <- 0; v$noise_sd <- 0; v$amplitude <- 0.08
v$wave_scale <- c(0.25, 1, 1)
r <- trajectory_path(364L, "multi_wave", v)
half <- 364 / 2
cat("rise", sum(pmax(r[1:half], 0)), sum(pmax(r[(half + 1):364], 0)), "\\n")
w <- vapply(1:500, function(i) {
  x <- draw_values(d)$wave_scale
  max(x) / min(x)
}, 1)
cat("spread", mean(w > 2.5), "\\n")
""",
        tmp_path,
    )
    rows = {x.split()[0]: x.split()[1:] for x in out.splitlines() if x}
    first, second = map(float, rows["rise"])
    # The second wave climbs four times as far in log as the first.
    assert 3.5 < second / first < 4.5
    # Waves often differ by more than x2.5 in their growth rates.
    assert float(rows["spread"][0]) > 0.3


# Two places on a flat path, daily reporting, with a reporting irregularity
# (`delay_change`) everywhere: report days with no reports, or a slowdown.
IRREGULAR = """
set.seed(7)
d <- draw_design(1L)
d$strata <- "places"; d$n_places <- 2L; d$n_ages <- 1L
d$spread <- "even"; d$seeding <- "start"; d$place_rt <- "shared"
d$mixing <- "none"; d$imports <- "none"
d$trajectory <- "flat"; d$delay_change <- "{level}"
d$delay_form <- "parametric"; d$delay_family <- "lognormal"
d$reporting <- "daily"; d$shocks <- "none"; d$delay_places <- "shared"
v <- draw_values(d)
v$weeks <- 20L; v$target_size <- 20000; v$quiet_places <- integer(0)
v$delay_median <- 3; v$delay_sdlog <- 0.6; v$flat <- 0
v$bunch_at <- c(0.3, 0.6); v$bunch_len <- c(2L, 3L)
v$slow_at <- 0.4; v$slow_weeks <- 4L; v$slow_mult <- 3
out <- simulate_outbreak(d, v)
c <- out$counts[draw == 1]
by_day <- c[, .(n = sum(count)), by = .(location, report_date)]
held <- as.IDate(out$values$bunch_dates)
cat("held", nrow(held_rows <- by_day[report_date %in% held]),
  sum(held_rows$n), "\\n")
first <- min(held)
after <- by_day[report_date == first + 2L, sum(n)]
before <- by_day[report_date == first - 2L, sum(n)]
cat("catch_up", after / before, "\\n")
w <- slowdown_window(c, v)
delay <- function(x) {
  x[, sum(count * as.numeric(report_date - reference_date)) / sum(count)]
}
for (p in sort(unique(c$location))) {
  x <- c[location == p]
  cat("delay", p, delay(x[reference_date >= w[1] & reference_date <= w[2]]),
    delay(x[reference_date < w[1] - 14]), "\\n")
}
"""


def _irregular(level: str, tmp_path: Path) -> list[list[str]]:
    out = run_r(IRREGULAR.replace("{level}", level), tmp_path)
    return [x.split() for x in out.splitlines() if x]


@needs_r
def test_bunching_holds_reports_everywhere_then_catches_up(tmp_path):
    rows = _irregular("bunching", tmp_path)
    held = next(r for r in rows if r[0] == "held")
    # No place reports on the five held report days.
    assert held[2] == "0"
    catch_up = next(r for r in rows if r[0] == "catch_up")
    # The two days held arrive on the next report day, with its own.
    assert float(catch_up[1]) > 2


@needs_r
def test_a_slowdown_lengthens_every_place_s_delays_for_a_while(tmp_path):
    rows = _irregular("slowdown", tmp_path)
    delays = [r for r in rows if r[0] == "delay"]
    assert len(delays) == 2
    for _, _, inside, outside in delays:
        assert float(inside) > 1.8 * float(outside)


@needs_r
def test_bunching_and_slowdown_are_delay_change_levels(tmp_path):
    out = run_r(
        """
set.seed(3)
cat("levels", factors$delay_change, "\\n")
d <- draw_design(1L)
d$delay_change <- "bunching"
v <- replicate(300, draw_values(d), simplify = FALSE)
n <- vapply(v, function(x) length(x$bunch_at), 1)
len <- unlist(lapply(v, `[[`, "bunch_len"))
cat("bunches", min(n), max(n), min(len), max(len), "\\n")
s <- vapply(v, function(x) c(x$slow_weeks, x$slow_mult), numeric(2))
cat("slow", min(s[1, ]), max(s[1, ]), min(s[2, ]), max(s[2, ]), "\\n")
""",
        tmp_path,
    )
    rows = {x.split()[0]: x.split()[1:] for x in out.splitlines() if x}
    assert rows["levels"] == [
        "static",
        "drift",
        "step",
        "periodic",
        "walk",
        "bunching",
        "slowdown",
    ]
    lo, hi, len_lo, len_hi = map(float, rows["bunches"])
    assert lo >= 1
    assert hi <= 13
    assert (len_lo, len_hi) == (1, 3)
    w_lo, w_hi, m_lo, m_hi = map(float, rows["slow"])
    assert (w_lo, w_hi) == (2, 6)
    assert m_lo >= 1.5
    assert m_hi <= 3


@needs_r
def test_settings_record_the_days_before_one_expected_onset(tmp_path):
    out = run_r(
        """
set.seed(2)
d <- draw_design(1L)
d$strata <- "single"; d$n_places <- 1L; d$n_ages <- 1L
d$trajectory <- "growth"; d$process <- "growth"; d$delay_change <- "static"
d$reporting <- "daily"; d$shocks <- "none"; d$seeding <- "start"
d$imports <- "none"
v <- draw_values(d)
v$weeks <- 52L; v$target_size <- 100; v$rate <- log(2) / 7
out <- simulate_outbreak(d, v)
cat("low", out$values$low_start_days, "\\n")
""",
        tmp_path,
    )
    low = int(out.split("low")[-1].split()[0])
    # Doubling weekly to 100 cases in a year starts far below one a day.
    assert low > 100


@needs_r
def test_enw_weeks_fixes_every_run_s_length_for_a_pilot(tmp_path):
    out = run_r(
        """
Sys.setenv(ENW_WEEKS = "52")
d <- lapply(0:200, draw_design)
cat("weeks", unique(vapply(d, `[[`, 1, "weeks")), "\\n")
pa <- Filter(function(x) x$strata == "places_ages", d)
cat("most", max(vapply(pa, function(x) x$n_places * x$n_ages, 1)), "\\n")
""",
        tmp_path,
    )
    rows = {x.split()[0]: x.split()[1:] for x in out.splitlines() if x}
    assert rows["weeks"] == ["52"]
    assert int(rows["most"][0]) <= 32


@needs_r
def test_mixing_strength_is_log_uniform_from_one_to_fifty_percent(tmp_path):
    out = run_r(
        """
set.seed(4)
cat("factor", "mixing_strength" %in% names(factors), "\\n")
d <- draw_design(1L)
eps <- vapply(1:3000, function(i) draw_values(d)$mixing_eps, 1)
cat("eps", min(eps), max(eps), mean(eps < 0.05), "\\n")
""",
        tmp_path,
    )
    rows = {x.split()[0]: x.split()[1:] for x in out.splitlines() if x}
    assert rows["factor"] == ["FALSE"]
    lo, hi, low = map(float, rows["eps"])
    assert lo >= 0.01
    assert hi <= 0.5
    # Log-uniform: log(5) / log(50), about 41%, below 0.05.
    assert 0.35 < low < 0.47


@needs_r
def test_one_to_three_places_each_get_one_to_three_shocks(tmp_path):
    out = run_r(
        """
set.seed(6)
d <- draw_design(1L)
d$strata <- "places"; d$n_places <- 10L; d$spread <- "even"
d$shocks <- "backlog"
for (i in 1:300) {
  v <- draw_values(d)
  cases <- setdiff(seq_len(d$n_places), v$quiet_places)
  n <- table(v$shock_places)
  cat(length(n) >= 1, length(n) <= min(3, length(cases)),
    all(n >= 1 & n <= 3), all(v$shock_places %in% cases),
    length(v$shock_at) == length(v$shock_places),
    length(v$shock_mult) == length(v$shock_places),
    all(v$shock_weeks >= 1 & v$shock_weeks <= 4), "\\n")
}
""",
        tmp_path,
    )
    assert out.split()
    assert all(x == "TRUE" for x in out.split())


@needs_r
def test_each_of_a_place_s_outages_stops_its_reports(tmp_path):
    out = run_r(
        SHOCK.replace("{level}", "outage")
        .replace(
            "v$shock_places <- 1L; v$shock_at <- 0.4; v$shock_weeks <- 3L",
            "v$shock_places <- c(1L, 1L); v$shock_at <- c(0.2, 0.6)\n"
            "v$shock_weeks <- c(1L, 2L)",
        )
        .replace("v$shock_mult <- 4", "v$shock_mult <- c(4, 2)")
        + """
for (k in 1:2) {
  w <- shock_window(out$counts, v, k)
  cat("window", k, c[report_date >= w[1] & report_date <= w[2], sum(count)],
    "\\n")
}
""",
        tmp_path,
    )
    rows = [x.split() for x in out.splitlines() if x.startswith("window")]
    assert [r[2] for r in rows] == ["0", "0"]


# Loads only the simulator's grid index, which needs no R packages.
INDEX = r"""
for (e in parse("scripts/finetune_simulate.R")) {
  txt <- paste(deparse(e), collapse = "")
  if (grepl("^(block|grid_index) <- ", txt)) eval(e, globalenv())
}
"""


def run_index(script: str, tmp_path: Path) -> subprocess.CompletedProcess:
    path = tmp_path / "index.R"
    path.write_text(INDEX + script)
    return subprocess.run(
        ["Rscript", str(path)], cwd=ROOT, capture_output=True, text=True
    )


@pytest.mark.skipif(shutil.which("Rscript") is None, reason="needs R")
def test_a_round_of_40_batches_of_21_is_exactly_its_grid_block(tmp_path):
    out = run_index(
        """
g <- unlist(lapply(30900:30939, function(s) {
  vapply(1:21, function(i) grid_index(s, 21L, i, 30000L), 0)
}))
cat(block, length(g), length(unique(g)), min(g), max(g),
  all(g %/% block == 9), "\\n")
cat(grid_index(31000, 21L, 1L, 30000L), "\\n")
""",
        tmp_path,
    )
    assert out.returncode == 0, out.stderr
    first, second = out.stdout.strip().splitlines()
    assert first.split() == ["840", "840", "840", "7560", "8399", "TRUE"]
    # Round 10 is the next block.
    assert second.split() == ["8400"]


@pytest.mark.skipif(shutil.which("Rscript") is None, reason="needs R")
def test_a_batch_past_its_round_s_block_is_refused(tmp_path):
    out = run_index("grid_index(30940, 21L, 1L, 30000L)\n", tmp_path)
    assert out.returncode != 0
    assert "block" in out.stderr


# A places outbreak on given weekdays with a non-parametric hazard, so its
# Stan data hold matrices and four-dimensional arrays.
STAN_INPUTS = """
set.seed(5)
d <- draw_design(7L)
d$strata <- "places"; d$n_places <- 3L; d$n_ages <- 1L
d$delay_form <- "both"; d$delay_change <- "walk"
d$reporting <- "weekdays"; d$report_days <- "Monday;Thursday"
v <- draw_values(d); v$weeks <- 16L
inputs <- NULL
simulate_data <- enw_simulate_data
enw_simulate_data <- function(data, ...) {
  inputs <<- list(data = data, ...)
  simulate_data(data, ...)
}
invisible(simulate_outbreak(d, v))
"""


@needs_r
def test_stan_data_in_slabs_parse_as_cmdstanr_writes_them(tmp_path):
    out = run_r(
        STAN_INPUTS
        + """
data <- do.call(enw_simulate_data, inputs)$data
variables <- model$variables()$data
ours <- write_stan_data(data, tempfile(), variables, min_size = 50,
  slab = 2e4)
theirs <- cmdstanr:::process_data(data, list(data = variables))
a <- jsonlite::fromJSON(ours, simplifyVector = FALSE)
b <- jsonlite::fromJSON(theirs, simplifyVector = FALSE)
cat("slabbed", sum(lengths(data) >= 50) > 1,
  any(lengths(lapply(data, dim)) == 4 & lengths(data) >= 50), "\\n")
cat("same", setequal(names(a), names(b)), identical(a[names(b)], b), "\\n")
""",
        tmp_path,
    )
    assert "slabbed TRUE TRUE" in out
    assert "same TRUE TRUE" in out


@needs_r
def test_json_arrays_have_jsonlite_s_text(tmp_path):
    # Every shape, slab size and value kind, -0 included.
    out = run_r(
        """
set.seed(2)
cases <- list(
  c(-0, runif(20), 0, 1, 1e15, -2.5e-7),
  matrix(sample(c(0, 1, 0.25, -0, 1 / 3), 35, TRUE), 7, 5),
  array(sample(0:9, 24, TRUE), dim = c(2, 3, 4)),
  array(sample(0:9, 120, TRUE), dim = c(3, 2, 4, 5)),
  array(runif(12), dim = c(2, 1, 3, 2)),
  matrix(runif(6), 6, 1),
  matrix(runif(6), 1, 6)
)
same <- vapply(cases, function(x) {
  all(vapply(c(1, 3, 7, 1e6), function(slab) {
    f <- tempfile()
    file.create(f)
    write_json_array(x, f, slab)
    ours <- gsub("[[:space:]]", "", paste(readLines(f, warn = FALSE),
      collapse = ""))
    theirs <- jsonlite::toJSON(x, always_decimal = TRUE, digits = NA)
    identical(ours, as.character(theirs))
  }, TRUE))
}, TRUE)
cat("same", all(same), "\\n")
""",
        tmp_path,
    )
    assert "same TRUE" in out


@needs_r
def test_two_step_simulation_draws_what_enw_simulate_draws(tmp_path):
    # The same draw and the same R random stream after it.
    out = run_r(
        STAN_INPUTS
        + """
set.seed(11)
fit <- suppressMessages(do.call(enw_simulate, c(inputs, list(
  model = model, seed = 99L, draws = 1L, refresh = 0
))))
theirs <- as.numeric(fit$fit[[1]]$draws("pp_obs", format = "matrix")[1, ])
after_theirs <- .Random.seed
set.seed(11)
stan <- do.call(enw_simulate_data, inputs)
stan$file <- write_stan_data(stan$data, tempfile(fileext = ".json"),
  model$variables()$data, min_size = 50)
ours <- enw_simulate_draw(stan, model, seed = 99L, refresh = 0)
cat("same", identical(ours, theirs), identical(.Random.seed, after_theirs),
  file.exists(stan$file), "\\n")
cat("cells", identical(stan$cells, fit$new_confirm[[1]]), "\\n")
""",
        tmp_path,
    )
    assert "same TRUE TRUE FALSE" in out
    assert "cells TRUE" in out
