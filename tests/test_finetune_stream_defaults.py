"""The trainer's command-line rates are the workflow's and the SI's."""

import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import finetune_stream as fs  # noqa: E402


def test_the_cli_rates_match_the_workflow_config():
    argv = "sims out --train-seeds 1 2 --held-out-seeds 3 4"
    args = fs.build_parser().parse_args(argv.split())
    config = yaml.safe_load((ROOT / "workflow" / "config.yaml").read_text())
    assert args.learning_rate == 1e-5
    assert args.learning_rate == config["finetune"]["learning_rate"]
    assert args.bias_lr == config["finetune"]["bias_lr"]


def test_the_loss_horizon_enters_the_settings():
    # Unset, only the prediction length is stated.
    full = fs.horizon_settings(None)
    assert "loss_horizon_days" not in full
    assert full["prediction_length"] == (
        f"each block's horizon in whole patches of {fs.ft.OUTPUT_PATCH} steps"
    )
    week = fs.horizon_settings(7)
    assert week["loss_horizon_days"] == 7
    assert "7 days after the cut-off" in week["prediction_length"]


def test_the_loss_horizon_defaults_to_the_full_horizon():
    argv = ["sims", "out", "--held-out-seeds", "1", "2"]
    assert fs.build_parser().parse_args(argv).loss_horizon_days is None
    args = fs.build_parser().parse_args([*argv, "--loss-horizon-days", "7"])
    assert args.loss_horizon_days == 7


def test_dropping_quiet_cut_offs_is_off_until_set():
    argv = ["sims", "out", "--held-out-seeds", "1", "2"]
    assert fs.build_parser().parse_args(argv).quiet_min_cases is None
    args = fs.build_parser().parse_args([*argv, "--quiet-min-cases", "10"])
    assert args.quiet_min_cases == 10
    assert fs.quiet_settings(None) == {}
    rule = fs.quiet_settings(10)["drop_quiet"]
    assert "fewer than 10 cases" in rule
    assert f"{fs.ft.QUIET_WINDOW_DAYS} days" in rule


def test_the_checkpoint_threshold_is_a_machine_setting_outside_the_hash():
    # Recomputing activations in backward changes memory, not the steps,
    # so a small GPU may checkpoint more without a new hash.
    import finetune_verify

    argv = ["sims", "out", "--held-out-seeds", "1", "2"]
    args = fs.build_parser().parse_args(argv)
    assert args.checkpoint_above is None
    assert fs.checkpoint_threshold(args) == fs.ft.checkpoint_above(
        args.memory_gb
    )
    args = fs.build_parser().parse_args([*argv, "--checkpoint-above", "0"])
    assert fs.checkpoint_threshold(args) == 0
    assert "checkpoint_above" in finetune_verify.NOT_HASHED
