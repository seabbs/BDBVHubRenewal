"""The job watcher writes progress and the job's end to the mailbox."""

import subprocess
from pathlib import Path

SCRIPT = Path(__file__).parents[1] / "scripts" / "job_watch.sh"


def _watch(tmp_path: Path, body: str, relaunch: str = "") -> str:
    log = tmp_path / "job.log"
    mailbox = tmp_path / "mailbox.md"
    job = subprocess.Popen(
        ["bash", "-c", body],
        stdout=log.open("w"),
        stderr=subprocess.STDOUT,
    )
    subprocess.run(
        [
            "bash",
            str(SCRIPT),
            "job",
            str(job.pid),
            str(log),
            "step [0-9]+",
            "1",
            relaunch,
        ],
        env={
            "MAILBOX": str(mailbox),
            "WATCH_SECONDS": "1",
            "PATH": "/usr/bin:/bin",
        },
        check=False,
        timeout=60,
    )
    job.wait()
    return mailbox.read_text()


def test_a_finished_job_is_reported_with_its_progress(tmp_path):
    text = _watch(
        tmp_path, "echo step 1; sleep 2; echo step 2; sleep 2; echo finished"
    )
    assert "job: progress: step" in text
    assert "job: finished" in text
    assert "FAILED" not in text


def test_a_job_that_dies_is_reported_and_relaunched_once(tmp_path):
    marker = tmp_path / "again"
    text = _watch(
        tmp_path,
        "echo step 1; sleep 1; exit 1",
        f"touch {marker}; echo finished",
    )
    assert "job: FAILED" in text
    assert "relaunched once" in text
    assert "finished after relaunch" in text
    assert marker.exists()
