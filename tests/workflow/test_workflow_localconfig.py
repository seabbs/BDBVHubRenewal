"""The workflow reads machine settings as scripts/local_config.sh does."""

import os
import subprocess
from pathlib import Path

import pytest
from flowlib import localconfig

ROOT = Path(__file__).parents[2]
SHELL = ROOT / "scripts" / "local_config.sh"


def test_with_no_file_everything_is_on_this_machine(tmp_path):
    env = {"HOME": str(tmp_path)}
    cfg = localconfig.read(tmp_path / "none.env", env)
    assert cfg["ROOT"] == f"{tmp_path}/bdbv-rounds"
    assert cfg["MAILBOX"] == f"{tmp_path}/bdbv-mailbox.md"
    assert cfg["INBOX"] == ""
    assert cfg["JOB_PREFIX"] == ""
    assert cfg["HOST_NAME"]


def test_the_file_is_read_and_the_environment_wins(tmp_path):
    path = tmp_path / "local.env"
    path.write_text(
        "# rounds\n\nROOT=$HOME/rounds\nMAILBOX=${HOME}/box.md\n"
        "HOST_NAME=sims-host\nlower=ignored\nnot a line\n"
    )
    env = {"HOME": str(tmp_path), "HOST_NAME": "from-env"}
    cfg = localconfig.read(path, env)
    assert cfg["ROOT"] == f"{tmp_path}/rounds"
    assert cfg["MAILBOX"] == f"{tmp_path}/box.md"
    assert cfg["HOST_NAME"] == "from-env"
    assert "lower" not in cfg


def test_later_lines_see_earlier_ones_and_quotes_are_dropped(tmp_path):
    path = tmp_path / "local.env"
    path.write_text(
        "ROOT=/data/rounds\nRUN=$ROOT/checkpoints/slots\n"
        "JOB_PREFIX='systemd-run --user --scope -p MemoryMax={mem}'\n"
    )
    cfg = localconfig.read(path, {"HOME": str(tmp_path)})
    assert cfg["RUN"] == "/data/rounds/checkpoints/slots"
    assert cfg["JOB_PREFIX"] == (
        "systemd-run --user --scope -p MemoryMax={mem}"
    )


def test_workflow_paths_default_under_the_rounds_folder(tmp_path):
    path = tmp_path / "local.env"
    path.write_text("ROOT=/data/rounds\n")
    cfg = localconfig.read(path, {"HOME": str(tmp_path)})
    assert cfg["RUN"] == "/data/rounds/checkpoints/slots"
    assert cfg["SCORE_CACHE"] == "/data/rounds/scores"
    assert cfg["NOSLOTS_RUN"] == "/data/rounds/checkpoints"
    assert cfg["GERMANY_BLOCKS"] == "/data/rounds/germany/germany.pkl"


def test_the_path_comes_from_local_config_when_set(tmp_path):
    path = tmp_path / "other.env"
    path.write_text("ROOT=/elsewhere\n")
    env = {"HOME": str(tmp_path), "LOCAL_CONFIG": str(path)}
    assert localconfig.read(None, env)["ROOT"] == "/elsewhere"


@pytest.mark.skipif(not SHELL.exists(), reason="needs scripts/local_config.sh")
def test_same_values_as_the_shell_loader(tmp_path):
    path = tmp_path / "local.env"
    path.write_text(
        "ROOT=$HOME/r\nMAILBOX=${HOME}/m.md\nHOST_NAME=h\nINBOX=i.md\n"
    )
    env = {
        "PATH": os.environ["PATH"],
        "HOME": str(tmp_path),
        "LOCAL_CONFIG": str(path),
    }
    keys = ("ROOT", "MAILBOX", "HOST_NAME", "INBOX")
    out = subprocess.run(
        [
            "bash",
            "-c",
            f'source "{SHELL}"; ' + "; ".join(f'echo "${k}"' for k in keys),
        ],
        env=env,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.split("\n")
    cfg = localconfig.read(None, env)
    assert [cfg[k] for k in keys] == out[: len(keys)]
