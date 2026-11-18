"""`task paper`: the paper run on a shared machine, niced and reported."""

import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parents[2] / "workflow" / "scripts"))

import paper_run


@pytest.fixture
def run(tmp_path, monkeypatch):
    """Record the commands and mailbox lines of one paper run."""
    calls, lines, codes = [], [], {}

    def fake(cmd, **kw):
        calls.append(cmd)
        log = kw.get("stdout")
        if log is not None and codes.get("fail"):
            log.write("Error in rule render: quarto failed\n")
        return subprocess.CompletedProcess(cmd, codes.get("rc", 0))

    cfg = {"PAPER_LOGS": str(tmp_path / "logs"), "CORES": "6"}
    monkeypatch.setattr(paper_run.subprocess, "run", fake)
    monkeypatch.setattr(paper_run.localconfig, "read", lambda: cfg)
    monkeypatch.setattr(paper_run, "say", lambda text, _cfg: lines.append(text))
    monkeypatch.setattr(paper_run, "missing", lambda: codes.get("missing", []))
    monkeypatch.setattr(paper_run, "pin", lambda: "slots-abc")
    return calls, lines, codes


def test_development_runs_the_paper_niced_on_the_machine_s_cores(run):
    calls, lines, _ = run
    assert paper_run.main(["development"]) == 0
    (cmd,) = calls
    assert cmd[:3] == ["nice", "-n", "19"]
    assert cmd[3:] == ["scripts/paper.sh", "development"]
    assert lines[0].startswith("started")
    assert lines[-1].startswith("done")


@pytest.mark.usefixtures("run")
def test_the_cores_come_from_the_machine_s_budget(monkeypatch):
    seen = {}

    def fake(cmd, **kw):
        seen.update(kw["env"])
        return subprocess.CompletedProcess(cmd, 0)

    monkeypatch.setattr(paper_run.subprocess, "run", fake)
    paper_run.main(["development"])
    assert seen["CORES"] == "6"


def test_the_paper_waits_for_its_held_out_inputs_without_failing(run):
    calls, lines, codes = run
    codes["missing"] = ["paper/sims/READY"]
    assert paper_run.main(["paper"]) == 0
    assert calls == []
    assert lines == ["waiting for: paper/sims/READY"]


def test_the_paper_pins_the_final_run_s_newest_checkpoint(run):
    _, lines, _ = run
    assert paper_run.main(["paper"]) == 0
    assert "slots-abc" in lines[0]


def test_a_failed_run_is_reported_with_its_first_error(run):
    _, lines, codes = run
    codes["rc"], codes["fail"] = 2, True
    assert paper_run.main(["development"]) == 2
    assert lines[-1].startswith("FAILED")
    assert "quarto failed" in lines[-1]


def test_single_targets_run_the_paper_workflow(run):
    calls, _, _ = run
    assert paper_run.main(["--targets", "figures/a.svg", "tables"]) == 0
    (cmd,) = calls
    assert cmd[:3] == ["nice", "-n", "19"]
    assert "workflow/Snakefile" in cmd
    assert cmd[cmd.index("--cores") + 1] == "6"
    assert "figures/a.svg" in cmd
    assert "tables" in cmd


@pytest.mark.usefixtures("run")
def test_unknown_targets_are_an_error():
    with pytest.raises(SystemExit):
        paper_run.main(["nope"])
