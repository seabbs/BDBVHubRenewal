"""The Snakemake DAG stays valid and the development target runs."""

import re
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest
import yaml

from paperpipe.config import last_release, run_studies

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


DEV_STUDIES = run_studies(CONFIG["studies"], "development")


def test_development_writes_main_and_si_outputs(development: Path) -> None:
    for study in DEV_STUDIES:
        assert (
            development / "figures" / "main" / f"{study}-results.png"
        ).is_file()
        assert (
            development / "tables" / "main" / f"{study}-scores.csv"
        ).is_file()
        assert list((development / "figures" / "si").glob(f"{study}-*.png"))
        assert list((development / "tables" / "si").glob(f"{study}-*.csv"))
    assert (development / "tables" / "main" / "synthetic-grid.csv").is_file()
    joint = development / "tables" / "main" / "synthetic-joint.md"
    assert "rel_energy_log" in joint.read_text()
    design = development / "tables" / "si" / "grid-epidemic.md"
    assert "Generative process" in design.read_text()
    assert (development / "figures" / "si" / "sim-examples.png").is_file()


def test_development_never_reads_heldout_data(development: Path) -> None:
    split = pd.Timestamp(CONFIG["heldout"]["germany_from"])
    germany = pd.read_csv(
        development / "forecasts" / "germany.csv.gz",
        parse_dates=["origin_date"],
    )
    assert germany["origin_date"].max() < split
    obs = pd.read_csv(
        development / "data" / "germany-observations.csv.gz",
        parse_dates=["target_date", "as_of_date"],
    )
    assert obs["target_date"].max() < split
    delay = CONFIG["truth_delay"]["germany"]
    assert obs["as_of_date"].max() <= last_release(split, delay)
    synthetic = pd.read_csv(development / "forecasts" / "synthetic.csv.gz")
    held = tuple(f"s{s}-" for s in CONFIG["heldout"]["synthetic_seeds"])
    assert not synthetic["location"].str.startswith(held).any()
    # The Ebola study is held out entirely.
    assert DEV_STUDIES == ("synthetic", "germany")
    assert not list(development.rglob("ebola*"))


def test_development_scores_have_truth(development: Path) -> None:
    for study in DEV_STUDIES:
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
    # No Ebola results, and placeholders for the tables the paper
    # includes inside the hidden Ebola sections.
    assert "ebola" not in results
    held = build / "tables" / "main" / "ebola-scores.md"
    assert "Held out" in held.read_text()


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
    for rule in ("germany_fetch", "germany_development"):
        assert f"rule {rule}" in log
    for rule in ("germany_heldout", "ebola_hub"):
        assert f"rule {rule}" not in log
    paper = _snakemake("paper", "-n", "--configfile", base, str(cfg))
    for rule in ("germany_heldout", "ebola_hub"):
        assert f"rule {rule}" in paper.stdout + paper.stderr


def test_only_the_paper_runs_ebola_scoring_every_round(
    tmp_path: Path,
) -> None:
    first = CONFIG["heldout"]["ebola_from"]
    base = str(PAPER / "workflow" / "config.yaml")
    for target in ("development", "paper"):
        res = _snakemake(
            target, "-n", "-p", "--configfile", base, str(_offline(tmp_path))
        )
        assert res.returncode == 0, res.stderr
        ebola = [
            line
            for line in (res.stdout + res.stderr).splitlines()
            if "paperpipe forecasts ebola" in line
            or "paperpipe figures ebola" in line
        ]
        if target == "development":
            assert not ebola
            continue
        assert ebola
        for line in ebola:
            assert f"--split {first}" in line, line


def test_benchmark_commands_run_from_the_repository_root(
    tmp_path: Path,
) -> None:
    # The models read config/ relative to the working directory
    # (config/finetune.toml), so a local run from paper/ must cd first.
    cfg = tmp_path / "real.yaml"
    cfg.write_text(yaml.safe_dump({"results": str(tmp_path / "results")}))
    base = str(PAPER / "workflow" / "config.yaml")
    paper = _snakemake(
        "paper", "-n", "-p", "--forceall", "--configfile", base, str(cfg)
    )
    assert paper.returncode == 0, paper.stderr
    commands = [
        line.strip().removeprefix("Shell command:").strip()
        for line in (paper.stdout + paper.stderr).splitlines()
        if "germany-backtest" in line
    ]
    assert len(commands) == 2
    for line in commands:
        # After the redirect into the rule's log.
        line = re.sub(r"^exec >\S+ 2>&1; ", "", line)
        assert line.startswith("cd .. && ")
        assert "--config config/germany.toml" in line
        assert "--data data/raw/germany" in line


def test_the_synthetic_runner_runs_once_the_simulations_exist(
    tmp_path: Path,
) -> None:
    sims = tmp_path / "sims"
    sims.mkdir()
    seeds = CONFIG["development"]["synthetic_seeds"]
    extra = {
        "synthetic": {
            **CONFIG["synthetic"],
            "simulations": {"development": str(sims), "paper": str(sims)},
            "forecasts": {
                "development": str(tmp_path / "out" / "synthetic-dev.csv.gz"),
                "paper": str(tmp_path / "out" / "synthetic.csv.gz"),
            },
        }
    }
    base = str(PAPER / "workflow" / "config.yaml")
    absent = _snakemake(
        "development",
        "-n",
        "--configfile",
        base,
        str(_offline(tmp_path, **extra)),
    )
    assert absent.returncode == 0, absent.stderr
    assert "rule synthetic_forecasts" not in absent.stdout + absent.stderr
    for seed in seeds:
        (sims / f"counts-{seed}.csv.gz").write_text("")
        (sims / f"settings-{seed}.csv").write_text("")
    present = _snakemake(
        "development",
        "-n",
        "-p",
        "--configfile",
        base,
        str(_offline(tmp_path, **extra)),
    )
    assert present.returncode == 0, present.stderr
    log = present.stdout + present.stderr
    assert "rule synthetic_forecasts_development" in log
    assert "--seeds " + " ".join(str(s) for s in seeds) in log


def test_the_paper_runs_its_ebola_models_at_its_own_draws(
    tmp_path: Path,
) -> None:
    cfg = tmp_path / "real.yaml"
    cfg.write_text(
        yaml.safe_dump(
            {
                "results": str(tmp_path / "results"),
                "build": str(tmp_path / "build"),
                "germany": {"real": False},
                "ebola": {**CONFIG["ebola"], "hub": str(tmp_path / "hub")},
            }
        )
    )
    base = str(PAPER / "workflow" / "config.yaml")
    res = _snakemake("paper", "-n", "-p", "--configfile", base, str(cfg))
    assert res.returncode == 0, res.stderr
    log = res.stdout + res.stderr
    assert "rule ebola_backtest_draws" in log
    # The draws of the models the backtest runs itself, `[samples]
    # backtest`; the stored fits keep `[forecast] n_samples`.
    assert f"backtest = {CONFIG['ebola']['n_samples']}" in log
    assert CONFIG["ebola"]["n_samples"] == 1000


def test_the_workflow_pins_and_records_the_fine_tuned_checkpoint(
    tmp_path: Path,
) -> None:
    base = str(PAPER / "workflow" / "config.yaml")
    sims = tmp_path / "sims"
    sims.mkdir()
    for seed in CONFIG["development"]["synthetic_seeds"]:
        (sims / f"counts-{seed}.csv.gz").write_text("")
        (sims / f"settings-{seed}.csv").write_text("")
    synthetic = {
        **CONFIG["synthetic"],
        "simulations": {"development": str(sims), "paper": str(sims)},
        "forecasts": {
            "development": str(tmp_path / "out" / "synthetic-dev.csv.gz"),
            "paper": str(tmp_path / "out" / "synthetic.csv.gz"),
        },
    }
    # Until the final run publishes a checkpoint, the fine-tuned models
    # are pending: not run, and no checkpoint recorded.
    assert CONFIG["finetuned"] is None
    pending = _snakemake(
        "development",
        "-n",
        "-p",
        "--configfile",
        base,
        str(_offline(tmp_path, synthetic=synthetic)),
    )
    assert pending.returncode == 0, pending.stderr
    log = pending.stdout + pending.stderr
    assert "--finetuned" not in log
    runner = [line for line in log.splitlines() if "run_synthetic.py" in line]
    assert runner
    assert all("_ft" not in line for line in runner)
    pinned = _snakemake(
        "development",
        "-n",
        "-p",
        "--configfile",
        base,
        str(_offline(tmp_path, synthetic=synthetic, finetuned="abcdef123456")),
    )
    assert pinned.returncode == 0, pinned.stderr
    assert "--finetuned abcdef123456" in pinned.stdout + pinned.stderr
