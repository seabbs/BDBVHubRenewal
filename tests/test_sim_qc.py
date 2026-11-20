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

# The thresholds live in the run's grid file (`qc`), which the workflow
# config names (`simulate.grid_file`).
GRID_FILE = yaml.safe_load((ROOT / "workflow" / "config.yaml").read_text())[
    "simulate"
]["grid_file"]
LIMITS = yaml.safe_load((ROOT / GRID_FILE).read_text())["qc"]


def _table(**changes):
    row = {
        "simulation": "s-0001",
        "shocks": "none",
        "cases_median": 1000.0,
        "strata_with_cases": 2,
        "cases_over_population": 0.1,
        "max_weekly_log_growth": math.log(4),
        "delay_mean_days": 7.0,
    }
    return pd.DataFrame([row | changes])


def test_a_plausible_simulation_is_kept():
    assert sim_qc.check(_table(), LIMITS).empty


def test_there_is_no_fixed_cap_on_cases_per_stratum():
    # #408 item 10: only impossible sizes are rejected.
    assert "max_cases_per_stratum" not in LIMITS
    assert sim_qc.check(_table(cases_median=1e9), LIMITS).empty


def test_more_cases_than_a_stratum_s_population_is_excluded():
    big = sim_qc.check(_table(cases_over_population=1.5), LIMITS)
    assert big["reason"].tolist() == ["cases in a stratum above its population"]
    assert "cases_over_population=1.50" in big["evidence"].iloc[0]
    # A simulation whose settings give no populations is not judged on size.
    assert sim_qc.check(
        _table(cases_over_population=float("nan")), LIMITS
    ).empty


def test_cases_over_population_is_the_largest_stratum_s(tmp_path):
    _sim_dir(tmp_path)
    for path in tmp_path.glob("settings-*.csv"):
        s = pd.read_csv(path)
        s["stratum_location"] = "P01;P02"
        # P02 reports 120 days x 3 delays x its scale (2 to 4).
        s["stratum_population"] = "1e9;1000"
        s.to_csv(path, index=False)
    table = sim_qc.summaries(tmp_path).set_index("simulation")
    # 6001-0001: P02's scale is draw + 1, a median total of 360 x 2.5.
    assert table.loc["6001-0001", "cases_over_population"] == 0.9
    assert table.loc["6001-0002", "cases_over_population"] == 360 * 3.5 / 1000


def test_cases_over_population_is_unknown_without_populations(tmp_path):
    _sim_dir(tmp_path)
    table = sim_qc.summaries(tmp_path)
    assert table["cases_over_population"].isna().all()


def test_each_threshold_excludes_with_its_reason_and_evidence():
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
    assert "--max-weekly-jump" in out.stderr
