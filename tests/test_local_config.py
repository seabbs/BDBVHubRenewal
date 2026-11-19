"""Machine settings come from an untracked local config, not the code."""

import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).parents[1]


def test_the_local_config_is_not_tracked():
    ignored = subprocess.run(
        ["git", "check-ignore", "config/local.env"], cwd=ROOT, check=False
    )
    assert ignored.returncode == 0
    assert (ROOT / "config" / "local.env.example").exists()


def test_the_trainer_takes_its_device_from_the_machine():
    import sys

    # torch_device imports torch, which CI's test env does not install.
    pytest.importorskip("torch")

    sys.path.insert(0, str(ROOT / "scripts"))
    import finetune_stream as fs

    assert fs.torch_device("cpu") == "cpu"
    assert fs.torch_device("auto") in {"cuda", "mps", "cpu"}
