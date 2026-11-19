"""The deterministic sim-QC check of a round (scripts/sim_qc.py)."""

import json
import math
import subprocess
import sys
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(ROOT / "scripts"))
import sim_qc  # noqa: E402
from test_finetune_chronos import _sim_dir  # noqa: E402

LIMITS = yaml.safe_load((ROOT / "workflow" / "config.yaml").read_text())[
    "simulate"
]["qc"]


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
    cap = LIMITS["max_cases_per_stratum"]
    big = sim_qc.check(_table(cases_median=2.4 * cap), LIMITS)
    assert big["reason"].tolist() == [f"cases per stratum above {cap:,.0f}"]
    assert f"cases_per_stratum={1.2 * cap:.0f}" in big["evidence"].iloc[0]
    fold = LIMITS["max_weekly_jump"]
    jump = sim_qc.check(
        _table(max_weekly_log_growth=math.log(2 * fold)), LIMITS
    )
    assert jump["reason"].tolist() == [
        f"weekly cases rise more than {fold:,.0f}-fold in one week"
    ]
    days = LIMITS["max_mean_delay_days"]
    late = sim_qc.check(_table(delay_mean_days=days + 5), LIMITS)
    assert late["reason"].tolist() == [
        f"mean reporting delay over {days:,.0f} days"
    ]


def test_a_jump_is_excluded_whatever_the_reporting_shock():
    # Growth is by reference date; shocks move report dates only.
    got = sim_qc.check(
        _table(
            max_weekly_log_growth=math.log(2 * LIMITS["max_weekly_jump"]),
            shocks="backlog",
        ),
        LIMITS,
    )
    assert len(got) == 1


def test_the_round_s_check_is_recorded(tmp_path):
    sims = tmp_path / "r1" / "sims"
    sims.mkdir(parents=True)
    _sim_dir(sims)
    flags = [
        x
        for k, v in LIMITS.items()
        for x in ("--" + k.replace("_", "-"), str(v))
    ]
    subprocess.run(
        [sys.executable, "scripts/sim_qc.py", str(tmp_path / "r1"), *flags],
        cwd=ROOT,
        check=True,
    )
    qc = tmp_path / "r1" / "qc"
    assert list(pd.read_csv(qc / "excluded.csv").columns) == [
        "simulation",
        "reason",
        "evidence",
    ]
    decisions = json.loads((qc / "decisions.json").read_text())
    assert decisions["limits"] == {k: float(v) for k, v in LIMITS.items()}
    assert decisions["n_simulations"] == 4
    assert len(pd.read_csv(qc / "input.csv")) == 4
    assert "of 4 simulations" in (qc / "summary.txt").read_text()


def test_every_threshold_is_required(tmp_path):
    out = subprocess.run(
        [sys.executable, "scripts/sim_qc.py", str(tmp_path)],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    assert out.returncode != 0
    assert "--max-cases-per-stratum" in out.stderr
