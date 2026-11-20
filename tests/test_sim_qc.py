"""The deterministic sim-QC check of a round (scripts/sim_qc.py)."""

import json
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

# The grid sets no sim QC thresholds: the round's QC records each
# simulation's summaries and excludes none.
GRID_FILE = yaml.safe_load((ROOT / "workflow" / "config.yaml").read_text())[
    "simulate"
]["grid_file"]
GRID = yaml.safe_load((ROOT / GRID_FILE).read_text())


def test_the_grid_sets_no_thresholds():
    assert "qc" not in GRID


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


def test_the_round_s_check_is_recorded(tmp_path):
    sims = tmp_path / "r1" / "sims"
    sims.mkdir(parents=True)
    _sim_dir(sims)
    subprocess.run(
        [sys.executable, "scripts/sim_qc.py", str(tmp_path / "r1")],
        cwd=ROOT,
        check=True,
    )
    qc = tmp_path / "r1" / "qc"
    assert list(pd.read_csv(qc / "excluded.csv").columns) == [
        "simulation",
        "reason",
        "evidence",
    ]
    assert pd.read_csv(qc / "excluded.csv").empty
    decisions = json.loads((qc / "decisions.json").read_text())
    assert decisions["excluded"] == []
    assert decisions["n_simulations"] == 4
    assert len(pd.read_csv(qc / "input.csv")) == 4
    assert "of 4 simulations" in (qc / "summary.txt").read_text()
