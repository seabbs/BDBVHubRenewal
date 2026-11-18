"""Machine settings come from an untracked local config, not the code."""

import os
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).parents[1]
CONFIG = ROOT / "scripts" / "local_config.sh"
LOCAL = (
    "REMOTE",
    "REMOTE_ROOT",
    "REMOTE_MAILBOX",
    "ROOT",
    "MAILBOX",
    "HOST_NAME",
    "INBOX",
    "LOCAL_CONFIG",
)


def _env(tmp_path, **extra):
    env = {k: v for k, v in os.environ.items() if k not in LOCAL}
    return {**env, "HOME": str(tmp_path), **extra}


def _show(tmp_path, **extra):
    out = subprocess.run(
        [
            "bash",
            "-c",
            f'source "{CONFIG}"; '
            'printf "%s\\n" "$REMOTE" "$REMOTE_ROOT" "$REMOTE_MAILBOX" '
            '"$ROOT" "$MAILBOX"',
        ],
        env=_env(tmp_path, **extra),
        capture_output=True,
        text=True,
        check=True,
    )
    return out.stdout.split("\n")[:5]


def test_with_no_local_config_everything_is_on_this_machine(tmp_path):
    remote, remote_root, remote_mailbox, root, mailbox = _show(
        tmp_path, LOCAL_CONFIG=str(tmp_path / "none.env")
    )
    assert remote == ""
    assert remote_root == root == f"{tmp_path}/bdbv-rounds"
    assert remote_mailbox == mailbox == f"{tmp_path}/bdbv-mailbox.md"


def test_the_local_config_is_read_and_the_environment_wins(tmp_path):
    config = tmp_path / "local.env"
    config.write_text(
        "# a remote\nREMOTE=sims-host\nREMOTE_ROOT=rounds-dir\n"
        "REMOTE_MAILBOX=box.md\n"
    )
    remote, remote_root, remote_mailbox, _, _ = _show(
        tmp_path, LOCAL_CONFIG=str(config), REMOTE_ROOT="mine"
    )
    assert (remote, remote_root, remote_mailbox) == (
        "sims-host",
        "mine",
        "box.md",
    )


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
