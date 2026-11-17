"""Mailbox lines and the evidence branch guard."""

import pytest
from flowlib import notify


def test_lines_are_posted_only_when_asked(tmp_path):
    box = tmp_path / "box.md"
    cfg = {"MAILBOX": str(box), "HOST_NAME": "h", "NOTIFY": ""}
    notify.post("round 4 ready", cfg)
    assert not box.exists()
    notify.post("round 4 ready", {**cfg, "NOTIFY": "1"})
    line = box.read_text()
    assert line.endswith(" h workflow: round 4 ready\n")
    assert line[:4].isdigit()


def test_land_refuses_main_and_other_branches():
    notify.check_branch("checkpoint-scoring", "checkpoint-scoring")
    with pytest.raises(SystemExit, match="main"):
        notify.check_branch("main", "main")
    with pytest.raises(SystemExit, match="checkpoint-scoring"):
        notify.check_branch("snakemake-workflow", "checkpoint-scoring")
