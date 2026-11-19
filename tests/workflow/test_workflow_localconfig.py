"""The workflow reads the machine settings from the local config."""

import os
import subprocess
from pathlib import Path

import pytest
from flowlib import localconfig

ROOT = Path(__file__).parents[2]


def test_with_no_file_everything_is_on_this_machine(tmp_path):
    env = {"HOME": str(tmp_path)}
    cfg = localconfig.read(tmp_path / "none.env", env)
    assert cfg["ROOT"] == f"{tmp_path}/bdbv-rounds"
    assert cfg["MAILBOX"] == ""
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


def test_a_line_may_use_a_later_one(tmp_path):
    path = tmp_path / "local.env"
    path.write_text("SCORE_CACHE=$ROOT/cache\nRUN=${ROOT}/run\nROOT=/data\n")
    cfg = localconfig.read(path, {"HOME": str(tmp_path)})
    assert cfg["SCORE_CACHE"] == "/data/cache"
    assert cfg["RUN"] == "/data/run"


def test_a_line_may_use_a_default(tmp_path):
    path = tmp_path / "local.env"
    path.write_text("SCORE_CACHE=$RUN/cache\n")
    cfg = localconfig.read(path, {"HOME": str(tmp_path)})
    assert (
        cfg["SCORE_CACHE"] == f"{tmp_path}/bdbv-rounds/checkpoints/slots/cache"
    )


def test_a_setting_in_the_environment_is_seen_by_the_file(tmp_path):
    path = tmp_path / "local.env"
    path.write_text("SCORE_CACHE=$ROOT/cache\nROOT=/data\n")
    cfg = localconfig.read(path, {"HOME": str(tmp_path), "ROOT": "/env"})
    assert cfg["ROOT"] == "/env"
    assert cfg["SCORE_CACHE"] == "/env/cache"


def test_a_line_that_uses_itself_takes_the_environment_s_value(tmp_path):
    # As the shell reads `ROOT=$ROOT/x`: unset, it is empty; set, the
    # environment wins over the line.
    path = tmp_path / "local.env"
    path.write_text("ROOT=$ROOT/x\nPYTHONPATH=$PYTHONPATH:/x\n")
    env = {"HOME": str(tmp_path), "PYTHONPATH": "/lib"}
    cfg = localconfig.read(path, env)
    assert cfg["ROOT"] == "/x"
    assert cfg["PYTHONPATH"] == "/lib"


def test_settings_that_use_each_other_are_an_error(tmp_path):
    path = tmp_path / "local.env"
    path.write_text("ROOT=$RUN/x\nRUN=$ROOT/y\n")
    with pytest.raises(ValueError, match="ROOT"):
        localconfig.read(path, {"HOME": str(tmp_path)})


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


def test_paper_logs_default_under_home(tmp_path):
    cfg = localconfig.read(tmp_path / "none.env", {"HOME": str(tmp_path)})
    assert cfg["PAPER_LOGS"] == f"{tmp_path}/paper-jobs"


def test_expand_fills_settings_into_a_path(tmp_path):
    path = tmp_path / "local.env"
    path.write_text("ROOT=/data/rounds\n")
    cfg = localconfig.read(path, {"HOME": str(tmp_path)})
    out = localconfig.expand("$ROOT/rounds/r1/sims", cfg)
    assert out == "/data/rounds/rounds/r1/sims"
    assert localconfig.expand("../output/x", cfg) == "../output/x"


def test_the_command_line_prints_one_setting(tmp_path):
    path = tmp_path / "local.env"
    path.write_text("MAILBOX=/data/box.md\n")
    env = {
        "PATH": os.environ["PATH"],
        "HOME": str(tmp_path),
        "LOCAL_CONFIG": str(path),
    }
    script = ROOT / "workflow" / "flowlib" / "localconfig.py"
    out = subprocess.run(
        ["python3", str(script), "MAILBOX"],
        env=env,
        capture_output=True,
        text=True,
        check=True,
    )
    assert out.stdout == "/data/box.md\n"


def test_paper_cores_come_from_the_local_config():
    taskfile = (ROOT / "paper" / "Taskfile.yml").read_text()
    assert "localconfig.py CORES" in taskfile
    script = (ROOT / "scripts" / "paper.sh").read_text()
    assert "localconfig.py CORES" in script


@pytest.mark.parametrize("taskfile", ["Taskfile.yml", "paper/Taskfile.yml"])
def test_task_leaves_the_local_config_to_localconfig(taskfile):
    # Task's dotenv expands `$HOME` against the file alone, so
    # ROOT=$HOME/rounds reached every job as /rounds, and the environment
    # wins over the file.
    text = (ROOT / taskfile).read_text()
    assert "dotenv:" not in text


def test_the_workflow_s_profile_comes_from_the_local_config():
    text = (ROOT / "Taskfile.yml").read_text()
    assert "localconfig.py SNAKEMAKE_PROFILE" in text


def test_a_host_s_seeds_profile_and_rounds_host_are_local(tmp_path):
    cfg = localconfig.read(tmp_path / "none.env", {"HOME": str(tmp_path)})
    for key in ("SIM_SEEDS", "SNAKEMAKE_PROFILE", "REMOTE", "REMOTE_ROOT"):
        assert cfg[key] == ""
