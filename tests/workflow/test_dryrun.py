"""Dry runs of the workflow's targets against a temporary machine."""

import gzip
import re
import subprocess
import sys
from pathlib import Path

import pytest

pytest.importorskip("snakemake")

ROOT = Path(__file__).parents[2]
SNAKEMAKE = Path(sys.executable).parent / "snakemake"


def dry(tmp_path, *targets, config=()):
    """Return the jobs per rule of a dry run of `targets`."""
    local = tmp_path / "local.env"
    if not local.exists():
        local.write_text(f"ROOT={tmp_path}\nHOST_NAME=test-host\n")
    out = subprocess.run(
        [
            str(SNAKEMAKE),
            *targets,
            "-s",
            "workflow/Snakefile",
            "-n",
            "--quiet",
            "rules",
            "--cores",
            "1",
            "--workflow-profile",
            "workflow/profiles/ci",
            *(["--config", *config] if config else []),
        ],
        cwd=ROOT,
        env={
            "PATH": str(SNAKEMAKE.parent) + ":/usr/bin:/bin",
            "HOME": str(tmp_path),
            "LOCAL_CONFIG": str(local),
        },
        capture_output=True,
        text=True,
    )
    assert out.returncode == 0, out.stderr + out.stdout
    jobs = dict(re.findall(r"^(\w+)\s+(\d+)$", out.stdout, re.M))
    return {k: int(v) for k, v in jobs.items() if k != "total"}


def test_simulate_runs_every_batch_of_a_round(tmp_path):
    jobs = dry(tmp_path, "simulate", config=["rounds=[4]"])
    assert jobs["simulate_batch"] == 44
    assert jobs["batch_record"] == 44
    assert jobs["round_manifest"] == 1
    assert jobs["sim_qc"] == 1
    assert jobs["round_ready"] == 1


def test_simulate_never_releases(tmp_path):
    jobs = dry(tmp_path, "simulate", config=["rounds=[4]"])
    assert not any(k.startswith("release") for k in jobs)


def test_release_is_its_own_target(tmp_path):
    jobs = dry(tmp_path, "release", config=["rounds=[4]"])
    assert jobs["release_batch"] == 44


def test_batches_already_simulated_are_kept(tmp_path):
    sims = tmp_path / "rounds" / "r4" / "sims"
    sims.mkdir(parents=True)
    for seed in (30400, 30401):
        with gzip.open(sims / f"counts-{seed}.csv.gz", "wt") as f:
            f.write("x\n")
        (sims / f"settings-{seed}.csv").write_text("outbreak\n")
    jobs = dry(tmp_path, "simulate", config=["rounds=[4]"])
    assert jobs["simulate_batch"] == 42
