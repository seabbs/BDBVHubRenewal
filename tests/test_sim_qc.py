"""The deterministic sim-QC check of a round (scripts/sim_qc.py)."""

import json
import math
import os
import subprocess
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(ROOT / "scripts"))
from test_finetune_chronos import _sim_dir  # noqa: E402

import sim_qc  # noqa: E402

LIMITS = {
    "max_cases_per_stratum": 5e5,
    "max_weekly_jump": 1000,
    "max_mean_delay_days": 35,
}


def _table(**changes):
    row = {
        "simulation": "s-0001",
        "shocks": "none",
        "cases_median": 1000.0,
        "strata_with_cases": 2,
        "max_weekly_log_growth": math.log(4),
        "delay_mean_days": 7.0,
    }
    return pd.DataFrame([row | changes])


def test_a_plausible_simulation_is_kept():
    assert sim_qc.check(_table(), LIMITS).empty


def test_each_threshold_excludes_with_its_reason_and_evidence():
    big = sim_qc.check(_table(cases_median=1.2e6), LIMITS)
    assert big["reason"].tolist() == ["cases per stratum above 500,000"]
    assert "cases_per_stratum=600000" in big["evidence"].iloc[0]
    jump = sim_qc.check(_table(max_weekly_log_growth=math.log(2000)), LIMITS)
    assert jump["reason"].tolist() == [
        "weekly cases rise more than 1,000-fold in one week with no "
        "reporting shock"
    ]
    late = sim_qc.check(_table(delay_mean_days=40.0), LIMITS)
    assert late["reason"].tolist() == ["mean reporting delay over 35 days"]


def test_a_jump_with_a_reporting_shock_is_kept():
    # A backlog or an outage can release many reports in one week.
    got = sim_qc.check(
        _table(max_weekly_log_growth=math.log(2000), shocks="backlog"),
        LIMITS,
    )
    assert got.empty


def test_the_round_s_check_is_recorded_without_any_model(tmp_path):
    sims = tmp_path / "r1" / "sims"
    sims.mkdir(parents=True)
    _sim_dir(sims)
    # No claude on the path: the check calls no model.
    env = {**os.environ, "PATH": "/usr/bin:/bin"}
    subprocess.run(
        [
            sys.executable,
            "scripts/sim_qc.py",
            str(tmp_path / "r1"),
            "--max-cases-per-stratum",
            "500000",
            "--max-weekly-jump",
            "1000",
            "--max-mean-delay-days",
            "35",
        ],
        cwd=ROOT,
        env=env,
        check=True,
    )
    qc = tmp_path / "r1" / "qc"
    assert list(pd.read_csv(qc / "excluded.csv").columns) == [
        "simulation",
        "reason",
        "evidence",
    ]
    decisions = json.loads((qc / "decisions.json").read_text())
    assert decisions["limits"] == LIMITS
    assert decisions["n_simulations"] == 4
    assert len(pd.read_csv(qc / "input.csv")) == 4
    assert "of 4 simulations" in (qc / "summary.txt").read_text()
    assert not (qc / "prompt-a.md").exists()
