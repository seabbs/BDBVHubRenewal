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
    "outbreaks": 3200,
    "prefixes": [100, 200, 400, 800, 1600, 3200],
    "loss_horizon_days": 7,
    "quiet_min_cases": 10,
    "draws": 1,
    "cuts": 14,
    "learning_rate": 1e-5,
    "bias_lr": 1e-3,
    "series_bias": True,
    "univariate_share": 0.2,
    "chunk": 100,
    "subset_seed": 1,
    # Held-out evaluation, which never changes the steps trained.
    "eval_every": 200,
    "by_source_every": 200,
}


def test_every_setting_trains_on_one_fixed_count():
    # A fixed number of simulations, not a time budget, ending a chunk.
    assert DESIGN["outbreaks"] == DESIGN["prefixes"][-1] == 3200
    # Checkpoints double from 100, each ending a chunk.
    assert DESIGN["prefixes"] == [100 * 2**i for i in range(6)]
    assert all(p % DESIGN["chunk"] == 0 for p in DESIGN["prefixes"])
    assert "hours" not in CONFIG["finetune"]
    # The rate falls to zero over the simulations, not a step count.
    assert "lr_steps" not in CONFIG["finetune"]


@pytest.mark.parametrize("key", sorted(DESIGN))
def test_finetune_config_is_the_agreed_design(key):
    assert CONFIG["finetune"][key] == DESIGN[key]


def test_rounds_are_simulated_with_one_draw():
    assert CONFIG["simulate"]["draws"] == 1
    assert CONFIG["simulate"]["seed_base"] == 30000


def test_a_round_is_800_simulations():
    sim = CONFIG["simulate"]
    assert (sim["batches"], sim["batch"]) == (40, 20)
    assert "held_batches" not in sim
    # QC failures are redrawn in their batch, so a round is always 800 and
    # the grid's training rounds hold the run's outbreaks exactly. The
    # run's rounds are the grid's.
    assert CONFIG["finetune"]["rounds"] is None
    assert CONFIG["finetune"]["validation_round"] is None
    grid = CONFIG["grid_rounds"]
    assert len(grid["rounds"]) * 800 == CONFIG["finetune"]["outbreaks"]
    assert grid["validation_round"] not in grid["rounds"]
    assert grid["test_round"] not in (
        *grid["rounds"],
        grid["validation_round"],
    )


def _trainer_args(tmp_path, finetune=None, machine=""):
    """The trainer's arguments in the dry run of segment n100."""
    pytest.importorskip("snakemake")
    snakemake = Path(sys.executable).parent / "snakemake"
    local = tmp_path / "local.env"
    local.write_text(f"ROOT={tmp_path}\nHOST_NAME=test-host\n{machine}")
    target = tmp_path / "checkpoints" / "slots" / "train" / "n100"
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
    assert args.until == 100
    # Main's default: held-out round 33, seeds 33300 to 33339
    # (grid_rounds), on the simulator's grid file.
    assert args.grid_file == Path(CONFIG["simulate"]["grid_file"])
    assert args.held_out_seeds == [33300, 33339]
    assert args.holdout == 90
    assert args.outbreaks == 3200
    assert args.prefixes == [100, 200, 400, 800, 1600, 3200]
    assert args.stop_from == 3201
    assert args.loss_horizon_days == 7
    assert args.quiet_min_cases == 10
    assert args.draws_per_simulation == 1
    assert args.blocks_per_simulation == DESIGN["cuts"]
    assert args.subset_seed == 1
    assert args.learning_rate == 1e-5
    assert args.series_bias
    assert args.bias_lr == 1e-3
    assert args.sources == CONFIG["grid_rounds"]["sources"]
    assert args.univariate_share == 0.2
    assert args.chunk == 100
    # The held-out evaluations' cadence comes from the config.
    assert args.eval_every == CONFIG["finetune"]["eval_every"]
    assert args.by_source_every == CONFIG["finetune"]["by_source_every"]
    # Evaluation runs off the trainer, on the evaluate rules' machine.
    assert args.eval == CONFIG["finetune"]["eval"] == "offload"


def test_the_loss_horizon_comes_from_the_config(tmp_path):
    pytest.importorskip("pandas")
    sys.path.insert(0, str(ROOT / "scripts"))
    import finetune_stream as fs

    args = fs.build_parser().parse_args(
        _trainer_args(tmp_path, {"loss_horizon_days": None})
    )
    assert args.loss_horizon_days is None
    args = fs.build_parser().parse_args(
        _trainer_args(tmp_path, {"loss_horizon_days": 7})
    )
    assert args.loss_horizon_days == 7


def test_quiet_cut_offs_come_from_the_config(tmp_path):
    pytest.importorskip("pandas")
    sys.path.insert(0, str(ROOT / "scripts"))
    import finetune_stream as fs

    args = fs.build_parser().parse_args(
        _trainer_args(tmp_path, {"quiet_min_cases": 10})
    )
    assert args.quiet_min_cases == 10


def test_the_grid_s_sources_are_the_block_design():
    sys.path.insert(0, str(ROOT / "scripts"))
    import finetune_chronos as ft

    sources = CONFIG["grid_rounds"]["sources"]
    assert sources == list(ft.BLOCK_DESIGN["source_shares"])


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


def test_a_machine_sets_the_checkpoint_threshold(tmp_path):
    pytest.importorskip("pandas")
    sys.path.insert(0, str(ROOT / "scripts"))
    import finetune_stream as fs

    args = fs.build_parser().parse_args(_trainer_args(tmp_path))
    assert args.checkpoint_above is None
    args = fs.build_parser().parse_args(
        _trainer_args(tmp_path, machine="CHECKPOINT_ABOVE=0\n")
    )
    assert args.checkpoint_above == 0
