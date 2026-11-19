"""The workflow's fine-tune settings are the agreed design (17 Nov 2026)."""

import shlex
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).parents[2]
CONFIG = yaml.safe_load((ROOT / "workflow" / "config.yaml").read_text())

DESIGN = {
    "holdout": 90,
    "outbreaks": 8400,
    "prefixes": [250, 500, 750, 1000, 2000, 4000, 8400],
    "grid": "r6",
    "draws": 1,
    "cuts": 32,
    "learning_rate": 1e-5,
    "bias_lr": 1e-3,
    "series_bias": True,
    "univariate_share": 0.2,
    "chunk": 250,
    "subset_seed": 1,
    # Held-out evaluation, which never changes the steps trained.
    "eval_every": 200,
    "by_source_every": 200,
}


def test_the_run_is_ten_grid_blocks():
    assert DESIGN["outbreaks"] == 10 * 840
    # The rate falls to zero over the simulations, not a step count.
    assert "lr_steps" not in CONFIG["finetune"]


@pytest.mark.parametrize("key", sorted(DESIGN))
def test_finetune_config_is_the_agreed_design(key):
    assert CONFIG["finetune"][key] == DESIGN[key]


def test_rounds_are_simulated_with_one_draw():
    assert CONFIG["simulate"]["draws"] == 1
    assert CONFIG["simulate"]["seed_base"] == 30000


def test_a_round_is_one_grid_block_of_840():
    sim = CONFIG["simulate"]
    assert sim["batches"] * sim["batch"] == 840
    assert (sim["batches"], sim["batch"]) == (40, 21)
    assert "held_batches" not in sim
    # Ten training rounds, after the held-out round, are the 8,400.
    # The run's rounds are its grid's.
    assert CONFIG["finetune"]["rounds"] is None
    assert CONFIG["finetune"]["validation_round"] is None
    for grid in CONFIG["grid_rounds"].values():
        assert len(grid["rounds"]) * 840 == CONFIG["finetune"]["outbreaks"]
        assert grid["validation_round"] not in grid["rounds"]
        assert grid["test_round"] not in (
            *grid["rounds"],
            grid["validation_round"],
        )


def _trainer_args(tmp_path, finetune=None):
    """The trainer's arguments in the dry run of segment n250."""
    pytest.importorskip("snakemake")
    snakemake = Path(sys.executable).parent / "snakemake"
    local = tmp_path / "local.env"
    local.write_text(f"ROOT={tmp_path}\nHOST_NAME=test-host\n")
    target = tmp_path / "checkpoints" / "slots" / "train" / "n250"
    # The real validation blocks are made from data a test run lacks.
    real = tmp_path / "real-validation.pkl"
    real.write_text("")
    overlay = tmp_path / "overlay.yaml"
    overlay.write_text(
        yaml.safe_dump(
            {"finetune": {"real_validation": str(real), **(finetune or {})}}
        )
    )
    out = subprocess.run(
        [
            str(snakemake),
            str(target / "settings.json"),
            "-s",
            "workflow/Snakefile",
            "-n",
            "-p",
            "--cores",
            "1",
            "--workflow-profile",
            "workflow/profiles/ci",
            "--configfile",
            str(overlay),
        ],
        cwd=ROOT,
        env={
            "PATH": str(snakemake.parent) + ":/usr/bin:/bin",
            "HOME": str(tmp_path),
            "LOCAL_CONFIG": str(local),
        },
        capture_output=True,
        text=True,
    )
    assert out.returncode == 0, out.stderr + out.stdout
    text = out.stdout.replace("\\\n", " ")
    line = next(
        ln for ln in text.splitlines() if "scripts/finetune_stream.py" in ln
    )
    words = shlex.split(line.split("scripts/finetune_stream.py", 1)[1])
    # Up to the log redirect, `>> log` or `>>log` (as snakefmt writes it).
    end = next(i for i, w in enumerate(words) if w.startswith(">>"))
    return words[:end]


def test_the_trainer_gets_the_agreed_design(tmp_path):
    pytest.importorskip("pandas")
    sys.path.insert(0, str(ROOT / "scripts"))
    import finetune_stream as fs

    args = fs.build_parser().parse_args(_trainer_args(tmp_path))
    assert args.until == 250
    # Main's default is the live r6 run: held-out round 20, seeds 32000
    # to 32039 (grid_rounds).
    assert args.grid == "r6"
    assert args.held_out_seeds == [32000, 32039]
    assert args.holdout == 90
    assert args.outbreaks == 8400
    assert args.prefixes == [250, 500, 750, 1000, 2000, 4000, 8400]
    assert args.stop_from == 8401
    assert args.draws_per_simulation == 1
    assert args.blocks_per_simulation == 32
    assert args.subset_seed == 1
    assert args.learning_rate == 1e-5
    assert args.series_bias
    assert args.bias_lr == 1e-3
    assert args.sources == CONFIG["grid_rounds"]["r6"]["sources"]
    assert args.univariate_share == 0.2
    assert args.chunk == 250
    # The held-out evaluations' cadence comes from the config.
    assert args.eval_every == CONFIG["finetune"]["eval_every"]
    assert args.by_source_every == CONFIG["finetune"]["by_source_every"]
    assert args.eval == CONFIG["finetune"]["eval"] == "local"


def test_an_r6_run_holds_out_round_20_and_trains_on_21_to_30(tmp_path):
    pytest.importorskip("pandas")
    sys.path.insert(0, str(ROOT / "scripts"))
    import finetune_stream as fs

    args = fs.build_parser().parse_args(_trainer_args(tmp_path, {"grid": "r6"}))
    assert args.grid == "r6"
    assert "daily+rolling7" in args.sources
    # Round 20, every batch: seeds 32000 to 32039.
    assert args.held_out_seeds == [32000, 32039]


def test_each_grid_s_sources_are_its_block_design():
    sys.path.insert(0, str(ROOT / "scripts"))
    import finetune_chronos as ft

    for grid, entry in CONFIG["grid_rounds"].items():
        assert entry["sources"] == list(ft.DESIGNS[grid]["source_shares"])


def test_development_scores_no_bdbv_blocks(tmp_path):
    # BDBV is held out for the paper: the shipped config gives the
    # trainer no real validation blocks and the check no BDBV loss.
    config = yaml.safe_load(
        (Path(__file__).parents[2] / "workflow" / "config.yaml").read_text()
    )["finetune"]
    assert config["real_validation"] is None
    assert "zero_shot_real" not in config
    args = _trainer_args(tmp_path, {"real_validation": None})
    assert "--real-validation" not in args
