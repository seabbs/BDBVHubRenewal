"""A coordinator run placing a job on a worker with its own root and checkout.

The worker is this machine (`local`) with another root and checkout, so
the push, the run, the pull and the status are the ones an ssh worker
gets, without a second machine.
"""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).parents[2]
SCRIPTS = REPO / "workflow" / "scripts" / "remote"

SNAKEFILE = """
import os
ROOT = os.environ["ROOT"]

rule all:
    input: f"{ROOT}/b.txt"

rule a:
    output: f"{ROOT}/a.txt"
    localrule: True
    shell: "echo from-coordinator > {output}"

rule b:
    input: f"{ROOT}/a.txt"
    output: f"{ROOT}/b.txt"
    log: f"{ROOT}/logs/b.log"
    params: copy_back=[f"{ROOT}/extra.txt"]
    threads: 1
    resources: mem_mb=10
    shell:
        "echo ran-in $(pwd) > {log}; "
        "cat {input} > {output}; echo $ROOT >> {output}; "
        "echo extra > {params.copy_back}"
"""


def _git(where, *args):
    subprocess.run(
        ["git", "-C", str(where), *args],
        check=True,
        capture_output=True,
        env={
            **os.environ,
            "GIT_AUTHOR_NAME": "t",
            "GIT_AUTHOR_EMAIL": "t@t",
            "GIT_COMMITTER_NAME": "t",
            "GIT_COMMITTER_EMAIL": "t@t",
        },
    )


@pytest.fixture
def setup(tmp_path):
    coord = tmp_path / "coord"
    coord.mkdir()
    (coord / "Snakefile").write_text(SNAKEFILE)
    _git(coord, "init", "-q")
    _git(coord, "add", "Snakefile")
    _git(coord, "commit", "-q", "-m", "x")
    worker = tmp_path / "worker"
    _git(tmp_path, "clone", "-q", str(coord), str(worker))
    roots = tmp_path / "croot", tmp_path / "wroot"
    for r in roots:
        r.mkdir()
    state = tmp_path / "state"
    state.mkdir()
    return coord, worker, roots, state


def _run(coord, worker_spec, croot, state):
    local = coord.parent / "local.env"
    local.write_text(f"COMPUTE=ssh\nROOT={croot}\nWORKERS={worker_spec}\n")
    (state / "budgets.json").write_text(
        json.dumps({"w": {"cores": 2, "mem_mb": 1000, "gpu": 1}})
    )
    py = sys.executable
    env = {
        **os.environ,
        "LOCAL_CONFIG": str(local),
        "ROOT": str(croot),
        "REMOTE_STATE": str(state),
        "REMOTE_POLL": "1",
        "REMOTE_SNAKEMAKE": f"{py} -m snakemake",
    }
    return subprocess.run(
        [
            py,
            "-m",
            "snakemake",
            "--executor",
            "cluster-generic",
            "--cluster-generic-submit-cmd",
            f"{py} {SCRIPTS / 'submit.py'}",
            "--cluster-generic-status-cmd",
            f"{py} {SCRIPTS / 'status.py'}",
            "--cluster-generic-cancel-cmd",
            f"{py} {SCRIPTS / 'cancel.py'}",
            "--jobs",
            "1",
            "--latency-wait",
            "5",
            "--max-status-checks-per-second",
            "2",
        ],
        cwd=coord,
        env=env,
        capture_output=True,
        text=True,
        timeout=300,
    )


def test_a_job_runs_on_the_worker_and_comes_back(setup):
    coord, worker, (croot, wroot), state = setup
    got = _run(coord, f"w=local:{wroot}:{worker}", croot, state)
    assert got.returncode == 0, got.stderr
    # The input was pushed, the job ran in the worker's checkout on its
    # root, and the outputs, the log and the files to copy back came back.
    assert (wroot / "a.txt").read_text() == "from-coordinator\n"
    assert (croot / "b.txt").read_text() == f"from-coordinator\n{wroot}\n"
    assert str(worker) in (croot / "logs" / "b.log").read_text()
    assert (croot / "extra.txt").read_text() == "extra\n"


def test_a_worker_on_another_commit_fails_the_job(setup):
    coord, worker, (croot, wroot), state = setup
    (coord / "new.txt").write_text("x")
    _git(coord, "add", "new.txt")
    _git(coord, "commit", "-q", "-m", "y")
    got = _run(coord, f"w=local:{wroot}:{worker}", croot, state)
    assert got.returncode != 0
    assert not (croot / "b.txt").exists()
    logs = "".join(p.read_text() for p in state.glob("*.log"))
    assert "commit" in logs


def test_an_unreachable_worker_fails_at_submit(setup):
    coord, _, (croot, _), state = setup
    got = _run(coord, "w=nohost.invalid:/x:/y", croot, state)
    assert got.returncode != 0
    assert not (croot / "b.txt").exists()
    assert "nohost.invalid" in got.stderr + got.stdout
