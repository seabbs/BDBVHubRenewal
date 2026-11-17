"""Machine settings come from an untracked local config, not the code."""

import os
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).parents[1]
CONFIG = ROOT / "scripts" / "local_config.sh"
CONTINUOUS = ROOT / "scripts" / "finetune_continuous.sh"
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


def test_with_no_remote_rounds_stream_in_from_this_machine(tmp_path):
    rounds = tmp_path / "bdbv-rounds" / "rounds" / "r1"
    (rounds / "sims").mkdir(parents=True)
    (rounds / "manifest.csv").write_text("round,seed,kind\n1,20001,train\n")
    (rounds / "sims" / "settings-20001.csv").write_text("x")
    # A running batch's temporary files are not simulations.
    (rounds / "sims" / "tmp" / "20002").mkdir(parents=True)
    (rounds / "sims" / "tmp" / "20002" / "standata.json").write_text("x")
    out = subprocess.run(
        [
            "bash",
            "-c",
            f'source "{CONTINUOUS}"; HERE={tmp_path}/local; '
            "sync_rounds; cat $HERE/seeds.txt",
        ],
        cwd=tmp_path,
        env=_env(tmp_path, LOCAL_CONFIG=str(tmp_path / "none.env")),
        capture_output=True,
        text=True,
        check=True,
    )
    assert out.stdout.split() == ["20001"]
    assert (tmp_path / "local" / "sims" / "settings-20001.csv").exists()
    assert not (tmp_path / "local" / "sims" / "tmp").exists()


def test_no_fine_tuning_or_simulation_code_names_a_host_or_home():
    files = subprocess.run(
        [
            "git",
            "ls-files",
            "scripts/finetune_*",
            "scripts/sim_*",
            "scripts/local_config.sh",
            "src/bdbvhubrenewal/forecasters",
            "config/finetune*",
            "config/local.env.example",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.split()
    found = subprocess.run(
        ["grep", "-n", "-i", "-E", r"archie|/home/[a-z]|/Users/[a-z]", *files],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    ).stdout
    assert found == ""


def test_the_trainer_takes_its_device_from_the_machine():
    import sys

    # pick_device imports torch, which CI's test env does not install.
    pytest.importorskip("torch")

    sys.path.insert(0, str(ROOT / "scripts"))
    import finetune_stream as fs

    assert fs.pick_device("cpu") == "cpu"
    assert fs.pick_device("auto") in {"cuda", "mps", "cpu"}
