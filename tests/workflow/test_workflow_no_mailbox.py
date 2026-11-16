"""The workflow runs without a mailbox; mailbox lines are an opt-in extra."""

import os
import subprocess
import sys
from pathlib import Path

import pytest
from flowlib import localconfig, notify

ROOT = Path(__file__).parents[2]
NOTIFY_SH = ROOT / "workflow" / "scripts" / "notify.sh"
SNAKEMAKE = Path(sys.executable).parent / "snakemake"


def _bare_env(tmp_path, **extra):
    """An environment with no MAILBOX, NOTIFY or local config."""
    env = {
        k: v
        for k, v in os.environ.items()
        if k not in ("MAILBOX", "NOTIFY", "LOCAL_CONFIG")
    }
    return {
        **env,
        "HOME": str(tmp_path),
        "LOCAL_CONFIG": str(tmp_path / "none.env"),
        **extra,
    }


def test_there_is_no_mailbox_by_default(tmp_path):
    cfg = localconfig.read(tmp_path / "none.env", {"HOME": str(tmp_path)})
    assert cfg["MAILBOX"] == ""
    assert cfg["NOTIFY"] == ""


def test_notify_needs_both_switch_and_mailbox(tmp_path):
    notify.post("hi", {"NOTIFY": "1", "MAILBOX": "", "HOST_NAME": "h"})
    assert list(tmp_path.iterdir()) == []


def test_a_mailbox_that_cannot_be_written_never_fails_the_caller(tmp_path):
    cfg = {
        "NOTIFY": "1",
        "MAILBOX": str(tmp_path / "no" / "such" / "box.md"),
        "HOST_NAME": "h",
    }
    (tmp_path / "no").write_text("a file, not a folder")
    notify.post("hi", cfg)


def _say(tmp_path, **env):
    return subprocess.run(
        ["bash", "-c", f'source "{NOTIFY_SH}"; say "a line"; echo rc=$?'],
        env=_bare_env(tmp_path, **env),
        capture_output=True,
        text=True,
    )


def test_the_shell_say_prints_and_writes_no_mailbox_by_default(tmp_path):
    box = tmp_path / "box.md"
    out = _say(tmp_path, MAILBOX=str(box))
    assert "a line" in out.stdout
    assert "rc=0" in out.stdout
    assert not box.exists()


def test_the_shell_say_appends_when_switched_on(tmp_path):
    box = tmp_path / "box.md"
    _say(tmp_path, MAILBOX=str(box), NOTIFY="1")
    assert box.read_text().rstrip().endswith("a line")


def test_the_shell_say_never_fails(tmp_path):
    (tmp_path / "no").write_text("x")
    out = _say(tmp_path, MAILBOX=str(tmp_path / "no" / "box.md"), NOTIFY="1")
    assert "rc=0" in out.stdout


@pytest.mark.parametrize(
    "rel",
    [
        "paper/scripts/gather_heldout.sh",
        "tools/paper_local.sh",
        "tools/run_target.sh",
    ],
)
def test_scripts_report_through_the_opt_in_helper(rel):
    text = (ROOT / rel).read_text()
    assert "workflow/scripts/notify.sh" in text
    assert '>> "$mb"' not in text


def test_detached_job_helpers_are_optional_tools():
    assert not (ROOT / "paper" / "scripts" / "run_target.sh").exists()
    assert not (ROOT / "paper" / "scripts" / "paper_local.sh").exists()
    assert "optional" in (ROOT / "tools" / "README.md").read_text().lower()


pytest.importorskip("snakemake")


def test_a_rule_that_notifies_runs_with_no_mailbox(tmp_path):
    n1000 = tmp_path / "checkpoints" / "slots" / "train" / "n1000"
    n1000.mkdir(parents=True)
    (n1000 / "settings.json").write_text(
        '{"held_out_loss": 0.4, "held_out_loss_start": 0.5,'
        ' "real_validation_loss": 3.0}'
    )
    local = tmp_path / "local.env"
    local.write_text(f"ROOT={tmp_path}\n")
    out = subprocess.run(
        [
            str(SNAKEMAKE),
            str(tmp_path / "checkpoints/slots/reports/n1000-check.txt"),
            "--cores",
            "1",
            "--workflow-profile",
            "workflow/profiles/ci",
        ],
        cwd=ROOT,
        env=_bare_env(tmp_path, LOCAL_CONFIG=str(local)),
        capture_output=True,
        text=True,
    )
    assert out.returncode == 0, out.stderr[-2000:]
    assert not list(tmp_path.glob("*mailbox*"))


def test_the_paper_targets_build_with_no_mailbox(tmp_path):
    out = subprocess.run(
        [
            str(SNAKEMAKE),
            "-s",
            "workflow/Snakefile",
            "development",
            "paper",
            "-n",
            "--cores",
            "1",
        ],
        cwd=ROOT / "paper",
        env=_bare_env(tmp_path),
        capture_output=True,
        text=True,
    )
    assert out.returncode == 0, out.stderr[-2000:]
