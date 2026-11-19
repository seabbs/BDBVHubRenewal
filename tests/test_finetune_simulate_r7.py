"""Grid r7 of the fine-tune simulator: renewal with depletion and imports."""

import re
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).parents[1]
SIMULATOR = ROOT / "scripts" / "finetune_simulate.R"

# The simulator's functions and settings without epinowcast (as
# tests/test_sim_settings.py loads them), switched to grid r7.
LIGHT = r"""
suppressPackageStartupMessages(library(data.table))
for (e in parse("scripts/finetune_simulate.R")) {
  if (!is.call(e) || !identical(e[[1]], as.name("<-"))) next
  rhs <- e[[3]]
  f <- is.call(rhs) && identical(rhs[[1]], as.name("function"))
  if (!f && grepl("enw_|commandArgs|args\\[\\[",
    paste(deparse(rhs), collapse = ""))) next
  eval(e, globalenv())
}
default_grid <- grid_name
grid_name <- "r7"
"""

needs_rscript = pytest.mark.skipif(
    shutil.which("Rscript") is None, reason="needs R"
)


def run_r7(script: str, tmp_path: Path) -> str:
    path = tmp_path / "run.R"
    path.write_text(LIGHT + script)
    out = subprocess.run(
        ["Rscript", str(path)],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
        timeout=300,
    )
    return out.stdout


def values(out: str) -> dict[str, str]:
    """Lines `key value` as a dict."""
    return dict(line.split(" ", 1) for line in out.strip().splitlines())


def _function(name: str) -> str:
    text = SIMULATOR.read_text()
    body = text[text.index(f"{name} <- function(") :]
    return body[: body.index("\n}\n")]


@needs_rscript
def test_r6_stays_the_default_and_r7_has_its_own_balanced_factors(tmp_path):
    out = values(
        run_r7(
            """
cat("default", default_grid, "\\n")
f <- grid_factors()
cat("process", "process" %in% names(f), "\\n")
cat("observation", identical(f$observation, factors$observation), "\\n")
cells <- lapply(0:(block - 1L), cell_for)
share <- function(k, level) mean(vapply(cells, `[[`, "", k) == level)
cat("early", share("window", "early"), "\\n")
cat("static", share("delay_change", "static"), "\\n")
cat("weekdays", share("reporting", "weekdays"), "\\n")
cat("balanced", all(block %% lengths(f) == 0L), "\\n")
""",
            tmp_path,
        )
    )
    assert values_of(out) == {
        "default": "r6",
        "process": "FALSE",
        "observation": "TRUE",
        "early": "0.25",
        "static": "0.4",
        "weekdays": "0.3",
        "balanced": "TRUE",
    }


def values_of(out: dict[str, str]) -> dict[str, str]:
    return {k: v.strip() for k, v in out.items()}


@needs_rscript
def test_r7_draws_its_distributions_without_truncation(tmp_path):
    out = values_of(
        values(
            run_r7(
                """
set.seed(1)
x <- draw_setting("delay_mean", 1e5)
cat("r7", min(x) < 1 && max(x) > 21, "\\n")
grid_name <- "r6"
y <- draw_setting("delay_mean", 1e5)
cat("r6", min(y) >= 1 && max(y) <= 21, "\\n")
""",
                tmp_path,
            )
        )
    )
    assert out == {"r7": "TRUE", "r6": "TRUE"}


@needs_rscript
def test_r7_series_run_a_median_30_weeks_and_at_least_4(tmp_path):
    out = values_of(
        values(
            run_r7(
                """
set.seed(2)
w <- vapply(0:3000, function(g) draw_design(g)$weeks, 1)
cat("min", min(w), "\\n")
cat("median", median(w), "\\n")
cat("over52", mean(w > 52) > 0, "\\n")
""",
                tmp_path,
            )
        )
    )
    assert int(out["min"]) >= 4
    assert 28 <= float(out["median"]) <= 32
    assert out["over52"] == "TRUE"


@needs_rscript
def test_r7_values_keep_the_gamma_shape_below_two_and_periods_a_month(
    tmp_path,
):
    out = values_of(
        values(
            run_r7(
                """
set.seed(3)
shape <- period <- waves <- numeric(0)
for (g in 0:400) {
  d <- draw_design(g)
  v <- draw_cell_values(d)
  shape <- c(shape, 1 / v$delay_cv^2)
  period <- c(period, v$period)
  waves <- c(waves, v$waves)
  stopifnot(length(v$wave_scale) >= v$waves + 1L, v$gt_mean > 0)
}
cat("shape", all(shape > 0 & shape < 2), "\\n")
cat("period", min(period) >= 30, "\\n")
cat("waves", min(waves) >= 2 && max(waves) > 3, "\\n")
""",
                tmp_path,
            )
        )
    )
    assert out == {"shape": "TRUE", "period": "TRUE", "waves": "TRUE"}


@needs_rscript
def test_depletion_bounds_each_stratum_by_its_population(tmp_path):
    out = values_of(
        values(
            run_r7(
                """
gt <- gt_pmf_r7(7, "gamma_0.6")
n <- 300L
log_r <- matrix(log(c(2.5, 1.8)), n, 2, byrow = TRUE)
start <- matrix(0, length(gt), 2); start[length(gt), ] <- 5
N <- c(1e4, 1e6)
run <- r7_infections(start, gt, log_r, N)
total <- colSums(run$infections)
cat("bounded", all(total <= N), "\\n")
cat("attack", paste(round(total / N, 2), collapse = ";"), "\\n")
cat("reff_falls", run$reff[n, 1] < 1, "\\n")
""",
                tmp_path,
            )
        )
    )
    assert out["bounded"] == "TRUE"
    small, large = map(float, out["attack"].split(";"))
    # R 2.5 infects most of a pool and R 1.8 well over half.
    assert small > 0.8
    assert large > 0.5
    assert out["reff_falls"] == "TRUE"


@needs_rscript
def test_a_reactive_shift_lowers_r_while_growing_and_raises_it_while_falling(
    tmp_path,
):
    out = values_of(
        values(
            run_r7(
                """
gt <- gt_pmf_r7(7, "gamma_0.6")
n <- 120L
log_r <- matrix(log(c(1.4, 0.7)), n, 2, byrow = TRUE)
start <- matrix(100, length(gt), 2)
shifts <- list(at = 60, size = 0.5, days = 1, reactive = TRUE, up = TRUE)
run <- r7_infections(start, gt, log_r, c(1e9, 1e9), shifts = shifts)
cat("down", round(run$r[n, 1] / 1.4, 3), "\\n")
cat("up", round(run$r[n, 2] / 0.7, 3), "\\n")
cat("before", round(run$r[30, 1] / 1.4, 3), "\\n")
""",
                tmp_path,
            )
        )
    )
    assert float(out["down"]) == pytest.approx(0.607, abs=0.002)
    assert float(out["up"]) == pytest.approx(1.649, abs=0.002)
    assert float(out["before"]) == pytest.approx(1.0, abs=0.002)


@needs_rscript
@pytest.mark.parametrize("window", ["early", "established"])
def test_windows_open_at_seeding_or_on_a_running_epidemic(window, tmp_path):
    out = values_of(
        values(
            run_r7(
                f"""
set.seed(4)
d <- draw_design(7L)
d$strata <- "places"; d$n_places <- 3L; d$n_ages <- 1L
d$seeding <- "start"; d$mixing <- "none"; d$window <- "{window}"
v <- draw_cell_values(d)
v$quiet_places <- 3L
strata <- data.table(.group = 1:3, place = sprintf("P%02d", 1:3),
  age = "A1")
gt <- gt_pmf_r7(v$gt_mean, d$gt_shape)
run <- r7_onsets(100L, gt, strata, d, v)
w <- seq_along(gt)
first <- run$infections[w, , drop = FALSE]
cat("quiet", all(run$onsets[, 3] == 0), "\\n")
cat("early_zero", all(first[-length(w), 1:2] == 0), "\\n")
cat("seed_floor", all(first[length(w), 1:2] >= 1), "\\n")
cat("flat_level", all(apply(first[, 1:2], 2, function(x) {{
  diff(range(x)) < 1e-9 && x[1] > 0
}})), "\\n")
cat("population", all(colSums(run$onsets) <= run$values$population), "\\n")
""",
                tmp_path,
            )
        )
    )
    assert out["quiet"] == "TRUE"
    assert out["population"] == "TRUE"
    if window == "early":
        assert out["early_zero"] == "TRUE"
        assert out["seed_floor"] == "TRUE"
    else:
        assert out["flat_level"] == "TRUE"


def test_r7_draws_only_named_distributions():
    # Discrete counts are 1 or 2 plus a Poisson and choices are Bernoulli
    # (`rbinom`); continuous settings come from `draw_setting`.
    body = _function("draw_values_r7")
    raw = set(re.findall(r"\b(runif|rnorm|rlnorm|sample|rpois|rbinom)\(", body))
    assert raw <= {"rpois", "rbinom"}


@needs_rscript
def test_r7_layers_have_their_shares_in_a_block(tmp_path):
    out = values_of(
        values(
            run_r7(
                """
cells <- lapply(0:(block - 1L), cell_for)
share <- function(k, level) mean(vapply(cells, `[[`, "", k) == level)
cat("early", share("window", "early"), "\\n")
cat("endemic", share("window", "endemic"), "\\n")
cat("bunching", share("bunching", "yes"), "\\n")
cat("revisions", share("revisions", "yes"), "\\n")
cat("smooth", share("delay_change", "smooth"), "\\n")
cat("old_bunching", share("delay_change", "bunching"), "\\n")
""",
                tmp_path,
            )
        )
    )
    assert out == {
        "early": "0.25",
        "endemic": "0.35",
        "bunching": "0.15",
        "revisions": "0.3",
        "smooth": "0.1",
        "old_bunching": "0",
    }


@needs_rscript
def test_a_variant_returns_immune_to_susceptible_and_raises_r(tmp_path):
    out = values_of(
        values(
            run_r7(
                """
gt <- gt_pmf_r7(7, "gamma_0.6")
n <- 200L
log_r <- matrix(log(2), n, 1)
start <- matrix(10, length(gt), 1)
variants <- list(at = 150, escape = 0.5, mult = 1.3, days = 1)
run <- r7_infections(start, gt, log_r, 1e5, variants = variants)
s <- run$susceptible[, 1]
immune <- 1e5 - s[149]
cat("jump", round((s[150] + run$infections[length(gt) + 150, 1] - s[149]) /
  immune, 3), "\\n")
cat("r", round(run$r[n, 1] / 2, 3), "\\n")
""",
                tmp_path,
            )
        )
    )
    assert float(out["jump"]) == pytest.approx(0.5, abs=0.01)
    assert float(out["r"]) == pytest.approx(1.3, abs=0.002)


@needs_rscript
def test_endemic_and_controlled_paths_follow_a_smooth_weekly_walk(tmp_path):
    out = values_of(
        values(
            run_r7(
                """
v <- list(walk_r = c(0, 0.1, -0.1, 0.05, 0, 0, 0, 0, 0, 0, 0),
  amplitude = 0.5, control_days = 14)
e <- r7_shape(56L, "endemic", v)
cat("knots", isTRUE(all.equal(e[c(1, 8, 15)], c(0, 0.1, -0.1))), "\\n")
cat("smooth", max(abs(diff(e))) <= 0.2 / 7 + 1e-9, "\\n")
c <- r7_shape(56L, "controlled", v)
cat("starts", isTRUE(all.equal(c[1], 0.5)), "\\n")
cat("decays", abs(c[56] - e[56]) < 0.5 * exp(-55 / 14) + 1e-9, "\\n")
""",
                tmp_path,
            )
        )
    )
    assert set(out.values()) == {"TRUE"}


@needs_rscript
def test_the_delay_walk_moves_smoothly_between_weeks(tmp_path):
    out = values_of(
        values(
            run_r7(
                """
walk <- c(0, 0.3, 0.1, 0.4)
x <- daily_walk(0:21, walk)
cat("knots", isTRUE(all.equal(x[c(1, 8, 15, 22)], walk)), "\\n")
cat("step", isTRUE(all.equal(max(abs(diff(x))), 0.3 / 7)), "\\n")
""",
                tmp_path,
            )
        )
    )
    assert set(out.values()) == {"TRUE"}


@needs_rscript
def test_onsets_are_ascertained_infections(tmp_path):
    out = values_of(
        values(
            run_r7(
                """
set.seed(6)
d <- draw_design(9L)
d$strata <- "places"; d$n_places <- 2L; d$n_ages <- 1L
d$seeding <- "start"; d$mixing <- "none"; d$window <- "established"
v <- draw_cell_values(d)
v$quiet_places <- integer(0)
strata <- data.table(.group = 1:2, place = c("P01", "P02"), age = "A1")
gt <- gt_pmf_r7(v$gt_mean, d$gt_shape)
run <- r7_onsets(100L, gt, strata, d, v)
a <- run$values$ascertainment
cat("range", all(a > 0 & a < 1) && length(a) == 2, "\\n")
cat("scaled", isTRUE(all.equal(
  run$onsets, sweep(run$infections, 2, a, `*`))), "\\n")
""",
                tmp_path,
            )
        )
    )
    assert out == {"range": "TRUE", "scaled": "TRUE"}


@needs_rscript
def test_revisions_change_releases_but_not_the_final_counts(tmp_path):
    out = values_of(
        values(
            run_r7(
                """
set.seed(7)
days <- as.IDate("2026-01-01") + 0:59
pp <- CJ(draw = 1L, location = c("P01", "P02"), reference_date = days,
  delay = 0:9)
pp[, report_date := reference_date + delay][, delay := NULL]
pp <- pp[report_date <= max(days)]
pp[, mean := rpois(.N, 20)]
v <- list(revise_over = 0.05, revise_move = 0.05, revise_keep = 14)
got <- revise_counts(pp, v)
final <- function(x) x[, .(n = sum(mean)), keyby = .(location,
  reference_date)]
cat("final", isTRUE(all.equal(final(got), final(pp))), "\\n")
setorder(got, location, reference_date, report_date)
got[, cum := cumsum(mean), by = .(location, reference_date)]
cat("decreases", any(got$mean < 0), "\\n")
cat("never_negative", all(got$cum >= 0), "\\n")
cat("not_after_report", all(got$reference_date <= got$report_date), "\\n")
""",
                tmp_path,
            )
        )
    )
    assert set(out.values()) == {"TRUE"}


def test_every_r6_uniform_timing_has_an_r7_distribution():
    text = SIMULATOR.read_text()
    r6 = text[text.index("setting_distributions <- list(") :]
    r6 = r6[: r6.index("\n)\n")]
    uniform = set(re.findall(r'(\w+) = list\("uniform"', r6))
    r7 = text[text.index("r7_distributions <- list(") :]
    r7 = r7[: r7.index("\n)\n")]
    named = set(re.findall(r'(\w+) = list\("', r7))
    assert uniform - {"phase"} <= named


@needs_rscript
def test_r7_delay_settings_have_plausible_tails(tmp_path):
    out = values_of(
        values(
            run_r7(
                """
set.seed(8)
q <- function(k) quantile(draw_setting(k, 1e5), 0.999)
cat("delay_sdlog", q("delay_sdlog") < 2.5, "\\n")
cat("dow_sd", q("dow_sd") < 1, "\\n")
cat("reference_dow_sd", q("reference_dow_sd") < 1, "\\n")
cat("delay_place_sd", q("delay_place_sd") < 1.2, "\\n")
cat("delay_mean", median(draw_setting("delay_mean", 1e5)) < 5.5, "\\n")
""",
                tmp_path,
            )
        )
    )
    assert set(out.values()) == {"TRUE"}


@needs_rscript
def test_onsets_follow_infections_by_the_incubation_delay(tmp_path):
    out = values_of(
        values(
            run_r7(
                """
pmf <- incubation_pmf(5)
cat("mean", round(sum((seq_along(pmf) - 1) * pmf), 1), "\\n")
x <- matrix(0, 40, 2); x[10, ] <- c(100, 50)
y <- convolve_rows(x, list(pmf, pmf))
cat("kept", isTRUE(all.equal(colSums(y), c(100, 50) * 1, tolerance = 1e-6)),
  "\\n")
cat("none_before", all(y[1:9, ] == 0), "\\n")
""",
                tmp_path,
            )
        )
    )
    assert 4.4 <= float(out["mean"]) <= 5.1
    assert out["kept"] == "TRUE"
    assert out["none_before"] == "TRUE"


@needs_rscript
@pytest.mark.parametrize("structure", ["shared", "pooled", "independent"])
def test_layer_values_vary_across_strata_as_their_structure_says(
    structure, tmp_path
):
    out = values_of(
        values(
            run_r7(
                f"""
set.seed(9)
a <- stratum_values("{structure}", 6L, "ascertainment", probability = TRUE)
m <- stratum_values("{structure}", 6L, "incubation_mean")
cat("n", length(a) == 6 && length(m) == 6, "\\n")
cat("range", all(a > 0 & a < 1) && all(m > 0), "\\n")
cat("same", length(unique(a)) == 1 && length(unique(m)) == 1, "\\n")
""",
                tmp_path,
            )
        )
    )
    assert out["n"] == "TRUE"
    assert out["range"] == "TRUE"
    assert out["same"] == ("TRUE" if structure == "shared" else "FALSE")


@needs_rscript
def test_r7_has_a_structure_factor_for_observation_and_reporting(tmp_path):
    out = values_of(
        values(
            run_r7(
                """
f <- grid_factors()
cat("obs", paste(f$observation_strata, collapse = ";"), "\\n")
cat("rep", paste(f$reporting_strata, collapse = ";"), "\\n")
""",
                tmp_path,
            )
        )
    )
    assert out["obs"] == "shared;pooled;independent"
    assert out["rep"] == "shared;pooled;independent"
