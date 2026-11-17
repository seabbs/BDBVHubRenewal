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
    "rounds": list(range(10, 20)),
    "held_round": 9,
    "holdout": 90,
    "outbreaks": 8400,
    "prefixes": [250, 500, 750, 1000, 2000, 4000, 8400],
    "grid": "r5",
    "draws": 1,
    "cuts": 32,
    "learning_rate": 1e-5,
    "bias_lr": 1e-3,
    "series_bias": True,
    "lr_steps": 38325,
    "sources": ["daily", "weekly", "3day", "daily+weekly", "daily+3day"],
    "univariate_share": 0.2,
    "chunk": 250,
    "subset_seed": 1,
}


def test_the_run_is_ten_grid_blocks_at_36500_steps_per_8000():
    # Ten of r5's balanced blocks of 840, and the learning rate reaching
    # zero there at the trainer's 36,500 steps per 8,000 simulations.
    assert DESIGN["outbreaks"] == 10 * 840
    assert DESIGN["lr_steps"] == DESIGN["outbreaks"] * 36500 // 8000


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
    rounds = CONFIG["finetune"]["rounds"]
    assert len(rounds) * 840 == CONFIG["finetune"]["outbreaks"]
    assert CONFIG["finetune"]["held_round"] not in rounds


def _trainer_args(tmp_path):
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
        yaml.safe_dump({"finetune": {"real_validation": str(real)}})
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
    return words[: words.index(">>")]


def test_the_trainer_gets_the_agreed_design(tmp_path):
    pytest.importorskip("pandas")
    sys.path.insert(0, str(ROOT / "scripts"))
    import finetune_stream as fs

    args = fs.build_parser().parse_args(_trainer_args(tmp_path))
    assert args.until == 250
    assert args.grid == "r5"
    # The held-out round 9, every batch: seeds 30900 to 30939.
    assert args.held_out_seeds == [30900, 30939]
    assert args.holdout == 90
    assert args.outbreaks == 8400
    assert args.prefixes == [250, 500, 750, 1000, 2000, 4000, 8400]
    assert args.stop_from == 8401
    assert args.draws_per_simulation == 1
    assert args.blocks_per_simulation == 32
    assert args.subset_seed == 1
    assert args.learning_rate == 1e-5
    assert args.lr_steps == 38325
    assert args.series_bias
    assert args.bias_lr == 1e-3
    assert args.sources == DESIGN["sources"]
    assert args.univariate_share == 0.2
    assert args.chunk == 250
