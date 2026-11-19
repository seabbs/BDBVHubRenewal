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
        "early": "0.3",
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
log_r <- matrix(log(c(2.5, 1.5)), n, 2, byrow = TRUE)
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
    # R 2.5 infects most of a pool; R 1.5 over 300 days well over half.
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
first <- run$onsets[w, , drop = FALSE]
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
