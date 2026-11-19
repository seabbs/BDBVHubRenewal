"""Sampling distributions of the simulation settings and their check."""

import re
import shutil
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).parents[1]
SIMULATOR = ROOT / "scripts" / "finetune_simulate.R"

# The simulator's functions and settings, without epinowcast, as
# scripts/sim_settings_check.R loads them.
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
"""

needs_rscript = pytest.mark.skipif(
    shutil.which("Rscript") is None, reason="needs R"
)


def run_light(script: str, tmp_path: Path) -> str:
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


def _draw_values() -> str:
    text = SIMULATOR.read_text()
    body = text[text.index("draw_values <- function(design) {") :]
    return body[: body.index("\n}\n")]


def test_every_continuous_draw_comes_from_a_sampling_distribution():
    body = _draw_values()
    # Only the Bernoulli choice of quiet places draws a raw uniform, and
    # strata's dispersion is log-normal around the outbreak's, with its
    # drawn log sd.
    raw = re.findall(r"\b(runif|rnorm|rlnorm|log_uniform)\(", body)
    assert raw == ["runif", "rnorm"]
    assert "runif(design$n_places) < v$quiet" in body
    assert "v$sqrt_phi * exp(rnorm(n_s, 0, v$obs_phi_sd))" in body


@needs_rscript
def test_each_distribution_is_named_with_a_reason(tmp_path):
    out = run_light(
        """
for (n in names(setting_distributions)) {
  d <- setting_distributions[[n]]
  cat(n, d[[1]], !is.null(d$why) && nzchar(d$why), "\\n")
}
""",
        tmp_path,
    )
    rows = [line.split() for line in out.splitlines()]
    assert len(rows) >= 40
    allowed = {"lognormal", "halfnormal", "normal", "beta", "uniform"}
    assert {r[1] for r in rows} <= allowed
    assert all(r[2] == "TRUE" for r in rows)


@needs_rscript
def test_draws_follow_their_distribution_and_truncation(tmp_path):
    out = run_light(
        """
set.seed(1)
phi <- draw_setting("sqrt_phi", 20000)
size <- draw_setting("size_per_stratum", 20000)
mean <- draw_setting("delay_mean", 20000)
cat("phi", min(phi), quantile(phi, 0.95), "\\n")
cat("size", min(size), median(size), max(size), "\\n")
cat("mean", min(mean), max(mean), "\\n")
cat("unknown", tryCatch(draw_setting("nope"), error = function(e) "error"),
  "\\n")
""",
        tmp_path,
    )
    rows = {x.split()[0]: x.split()[1:] for x in out.splitlines() if x}
    lo, q95 = map(float, rows["phi"])
    # Half-normal with scale 0.25: 95% of mass below about 0.49.
    assert lo >= 0
    assert 0.46 < q95 < 0.52
    lo, med, hi = map(float, rows["size"])
    assert lo >= 10
    assert hi <= 5e5
    assert 2600 < med < 3400
    lo, hi = map(float, rows["mean"])
    assert lo >= 1
    assert hi <= 21
    assert rows["unknown"] == ["error"]


@needs_rscript
def test_half_the_cells_have_no_reporting_shock(tmp_path):
    out = run_light(
        """
cells <- lapply(0:(block - 1L), cell_for)
sh <- table(vapply(cells, `[[`, "", "shocks")) / block
cat(names(sh), "\\n", sh, "\\n")
cat(grid_name, "\\n")
""",
        tmp_path,
    )
    names, shares, grid = out.splitlines()[:3]
    got = dict(zip(names.split(), map(float, shares.split()), strict=True))
    assert got == {"backlog": 0.25, "none": 0.5, "outage": 0.25}
    assert grid.strip() == "r6"


@needs_rscript
def test_the_check_writes_a_summary_per_draw(tmp_path):
    out = tmp_path / "sim.csv"
    subprocess.run(
        ["Rscript", "scripts/sim_settings_check.R", "40", str(out)],
        cwd=ROOT,
        check=True,
        capture_output=True,
        timeout=300,
    )
    sim = pd.read_csv(out)
    assert len(sim) == 40
    for col in (
        "max_growth",
        "peak_per_stratum",
        "delay_mean",
        "delay_q95",
        "reported_7",
        "cv_at_peak",
        "onset_dow_ratio",
        "shocks",
    ):
        assert col in sim
    assert sim["max_growth"].abs().max() <= 0.2
    assert sim["reported_28"].between(0, 1).all()


def test_the_comparison_tabulates_simulated_against_real(tmp_path):
    sys.path.insert(0, str(ROOT / "scripts"))
    import sim_settings_compare as cmp

    sim = pd.DataFrame(
        {
            "max_growth": [0.01, 0.05, 0.1],
            "doubling_days": [69.0, 14.0, 7.0],
            "peak_per_stratum": [10.0, 600.0, 1000.0],
            "total_per_stratum": [100.0, 1e4, 1e5],
            "delay_mean": [2.0, 6.0, 10.0],
            "delay_q95": [5, 20, 40],
            "reported_7": [0.5, 0.7, 0.9],
            "reported_14": [0.7, 0.9, 1.0],
            "reported_28": [0.9, 1.0, 1.0],
            "cv_at_peak": [0.5, 0.05, 0.03],
            "onset_dow_ratio": [1.0, 2.0, 3.0],
            "shock_mult": [2.0, 3.0, 4.0],
            "shocks": ["none", "none", "backlog"],
            "delay_change": ["static", "drift", "static"],
        }
    )
    real = pd.DataFrame(
        {
            "source": ["Germany", "BDBV"],
            "max_growth": [0.04, 0.2],
            "peak_daily": [1000.0, 80.0],
            "total": [1e5, 1e4],
            "delay_mean": [6.0, 7.0],
            "delay_q95": [30.0, 20.0],
            "reported_7": [0.7, 0.6],
            "reported_14": [0.9, 0.8],
            "reported_28": [0.95, 0.9],
            "cv_large": [0.1, None],
            "onset_dow_ratio": [3.0, None],
        }
    )
    sim.to_csv(tmp_path / "sim.csv", index=False)
    real.to_csv(tmp_path / "real.csv", index=False)
    sys.argv = ["x", str(tmp_path / "sim.csv"), str(tmp_path / "real.csv")]
    from contextlib import redirect_stdout
    from io import StringIO

    buf = StringIO()
    with redirect_stdout(buf):
        cmp.main()
    text = buf.getvalue()
    assert "| fastest daily growth rate |" in text
    assert "0.04 to 0.04" in text
    # Noise at large counts uses only peaks of 500 or more.
    noise = next(ln for ln in text.splitlines() if "relative noise" in ln)
    assert "0.5" not in noise.split("|")[2]
    assert "Shocks: none 67%" in text


@needs_rscript
def test_a_top_up_batch_simulates_the_cells_it_is_given(tmp_path):
    # ENW_CELLS: simulation i is cell i of the list, past any round's block
    # (a top-up seed's position would be refused otherwise); the default
    # path is the round's block as before.
    out = run_light(
        """
cat("given", grid_index(33200, 3L, 2L, 30000L, "17641,17650,21002"), "\\n")
cat("default", grid_index(32100, 21L, 2L, 30000L), "\\n")
cat("short", tryCatch(grid_index(33200, 3L, 1L, 30000L, "1,2"),
  error = function(e) "error"), "\\n")
cat("past", tryCatch(grid_index(33240, 21L, 1L, 30000L),
  error = function(e) "error"), "\\n")
""",
        tmp_path,
    )
    rows = {x.split()[0]: x.split()[1:] for x in out.splitlines() if x}
    assert rows["given"] == ["17650"]
    assert rows["default"] == [str(21 * 840 + 1)]
    assert rows["short"] == ["error"]
    assert rows["past"] == ["error"]
