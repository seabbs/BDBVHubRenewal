"""The Snakemake DAG stays valid and the development target runs."""

import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest
import yaml

pytest.importorskip("snakemake")

PAPER = Path(__file__).parents[1]
SNAKEFILE = PAPER / "workflow" / "Snakefile"
CONFIG = yaml.safe_load((PAPER / "workflow" / "config.yaml").read_text())
RULES = (
    "data",
    "forecasts",
    "table",
    "truth",
    "scores",
    "figures",
    "score_tables",
)


def _snakemake(*args: str) -> subprocess.CompletedProcess:
    cmd = [sys.executable, "-m", "snakemake", "-s", str(SNAKEFILE), *args]
    return subprocess.run(
        cmd, cwd=PAPER, capture_output=True, text=True, check=False
    )


@pytest.mark.parametrize("target", ["development", "paper"])
def test_dry_run(target: str, tmp_path: Path) -> None:
    res = _snakemake(
        target, "-n", "--cores", "1", "--config", f"results={tmp_path}"
    )
    assert res.returncode == 0, res.stderr
    log = res.stdout + res.stderr
    for rule in RULES:
        assert f"rule {rule}" in log


@pytest.fixture(scope="module")
def development(tmp_path_factory: pytest.TempPathFactory) -> Path:
    out = tmp_path_factory.mktemp("results")
    res = _snakemake(
        "development",
        "--cores",
        "2",
        "--config",
        f"results={out}",
        "n_samples=20",
    )
    assert res.returncode == 0, res.stderr
    return out / "development"


def test_development_writes_main_and_si_outputs(development: Path) -> None:
    for study in CONFIG["studies"]:
        assert (
            development / "figures" / "main" / f"{study}-results.png"
        ).is_file()
        assert (
            development / "tables" / "main" / f"{study}-scores.csv"
        ).is_file()
        assert list((development / "figures" / "si").glob(f"{study}-*.png"))
        assert list((development / "tables" / "si").glob(f"{study}-*.csv"))
    assert (development / "tables" / "main" / "synthetic-grid.csv").is_file()


def test_development_never_reads_heldout_data(development: Path) -> None:
    split = pd.Timestamp(CONFIG["heldout"]["germany_from"])
    germany = pd.read_csv(
        development / "forecasts" / "germany.csv.gz",
        parse_dates=["origin_date"],
    )
    assert germany["origin_date"].max() < split
    obs = pd.read_csv(
        development / "data" / "germany-observations.csv.gz",
        parse_dates=["target_date"],
    )
    assert obs["target_date"].max() < split
    synthetic = pd.read_csv(development / "forecasts" / "synthetic.csv.gz")
    held = tuple(f"s{s}-" for s in CONFIG["heldout"]["synthetic_seeds"])
    assert not synthetic["location"].str.startswith(held).any()


def test_development_scores_have_truth(development: Path) -> None:
    for study in CONFIG["studies"]:
        scores = pd.read_csv(development / "scores" / f"{study}.csv.gz")
        assert len(scores) > 0
        assert scores["score"].notna().all()
