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
