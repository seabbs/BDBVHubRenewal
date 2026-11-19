"""The trainer's command-line rates are the workflow's and the SI's."""

import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import finetune_stream as fs  # noqa: E402


def test_the_cli_rates_match_the_workflow_config():
    argv = "sims out --grid r6 --train-seeds 1 2 --held-out-seeds 3 4"
    args = fs.build_parser().parse_args(argv.split())
    config = yaml.safe_load((ROOT / "workflow" / "config.yaml").read_text())
    assert args.learning_rate == 1e-5
    assert args.learning_rate == config["finetune"]["learning_rate"]
    assert args.bias_lr == config["finetune"]["bias_lr"]


def test_the_loss_horizon_enters_the_settings():
    full = fs.horizon_settings(None)
    assert full["loss_horizon_days"] is None
    assert "whole patches" in full["prediction_length"]
    week = fs.horizon_settings(7)
    assert week["loss_horizon_days"] == 7
    assert "7 days after the cut-off" in week["prediction_length"]
    assert full != week


def test_the_loss_horizon_defaults_to_the_full_horizon():
    args = fs.build_parser().parse_args(
        ["sims", "out", "--held-out-seeds", "1", "2", "--grid", "r6"]
    )
    assert args.loss_horizon_days is None
    args = fs.build_parser().parse_args(
        [
            "sims",
            "out",
            "--held-out-seeds",
            "1",
            "2",
            "--grid",
            "r6",
            "--loss-horizon-days",
            "7",
        ]
    )
    assert args.loss_horizon_days == 7
