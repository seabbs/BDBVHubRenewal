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
    "publish",
    "summary",
    "forecasts",
    "table",
    "truth",
    "scores",
    "figures",
    "score_tables",
)


def _offline(tmp: Path, **extra) -> Path:
    """A config that runs every study on dummy data, writing under `tmp`."""
    path = tmp / "offline.yaml"
    cfg = {
        "results": str(tmp / "results"),
        "build": str(tmp / "build"),
        "germany": {"real": False},
        "ebola": {"hub": None},
        **extra,
    }
    path.write_text(yaml.safe_dump(cfg))
    return path


def _snakemake(*args: str) -> subprocess.CompletedProcess:
    cmd = [sys.executable, "-m", "snakemake", "-s", str(SNAKEFILE), *args]
    return subprocess.run(
        cmd, cwd=PAPER, capture_output=True, text=True, check=False
    )


@pytest.mark.parametrize("target", ["development", "paper"])
def test_dry_run(target: str, tmp_path: Path) -> None:
    res = _snakemake(
        target,
        "-n",
        "--cores",
        "1",
        "--configfile",
        str(PAPER / "workflow" / "config.yaml"),
        str(_offline(tmp_path)),
    )
    assert res.returncode == 0, res.stderr
    log = res.stdout + res.stderr
    for rule in RULES:
        assert f"rule {rule}" in log


@pytest.fixture(scope="module")
def development(tmp_path_factory: pytest.TempPathFactory) -> Path:
    tmp = tmp_path_factory.mktemp("run")
    res = _snakemake(
        "development",
        "--cores",
        "2",
        "--configfile",
        str(PAPER / "workflow" / "config.yaml"),
        str(_offline(tmp, n_samples=20)),
    )
    assert res.returncode == 0, res.stderr
    return tmp / "results" / "development"


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


def test_development_publishes_for_the_paper(development: Path) -> None:
    build = development.parents[1] / "build"
    assert (build / "figures" / "main" / "germany-results.png").is_file()
    assert (build / "tables" / "main" / "germany-scores.md").is_file()
    results = yaml.safe_load((build / "results.yml").read_text())["results"]
    assert results["mode"] == "development"
    assert results["germany"]["source"] == "dummy"
    assert results["germany"]["label"]


def test_dry_run_with_real_data_runs_the_benchmark(tmp_path: Path) -> None:
    cfg = tmp_path / "real.yaml"
    cfg.write_text(
        yaml.safe_dump(
            {
                "results": str(tmp_path / "results"),
                "build": str(tmp_path / "build"),
                "germany": {
                    "data": str(tmp_path / "germany"),
                    "results": str(tmp_path / "germany-results"),
                },
                "ebola": {"hub": str(tmp_path / "hub")},
            }
        )
    )
    base = str(PAPER / "workflow" / "config.yaml")
    dev = _snakemake("development", "-n", "--configfile", base, str(cfg))
    assert dev.returncode == 0, dev.stderr
    log = dev.stdout + dev.stderr
    for rule in ("germany_fetch", "germany_development", "ebola_hub"):
        assert f"rule {rule}" in log
    assert "rule germany_heldout" not in log
    paper = _snakemake("paper", "-n", "--configfile", base, str(cfg))
    assert "rule germany_heldout" in paper.stdout + paper.stderr
