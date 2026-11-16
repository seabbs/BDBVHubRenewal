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


FT = [
    "rounds=[1]",
    "finetune={'prefixes': [250, 500], 'outbreaks': 500, 'rounds': [1], "
    "'held_round': 1, 'sims_from': 'local'}",
]


def test_finetune_trains_one_segment_per_prefix(tmp_path):
    jobs = dry(tmp_path, "finetune", config=FT)
    assert jobs["train_segment"] == 2
    # 500 outbreaks are 20 batches, one spare, and 4 held-out batches.
    assert jobs["fetch_batch"] == 25
    # On one machine the batches are simulated here.
    assert jobs["simulate_batch"] == 25


def test_finetune_takes_batches_from_releases(tmp_path):
    config = [FT[0], FT[1].replace("'local'", "'release'")]
    jobs = dry(tmp_path, "finetune", config=config)
    assert jobs["fetch_batch"] == 25
    assert "simulate_batch" not in jobs


def test_finetune_never_publishes(tmp_path):
    jobs = dry(tmp_path, "finetune", config=FT)
    assert not any(k.startswith("publish") for k in jobs)


def test_finished_checkpoints_are_kept(tmp_path):
    n250 = tmp_path / "checkpoints" / "slots" / "train" / "n250"
    (n250 / "finetuned-ckpt").mkdir(parents=True)
    (n250 / "state").mkdir()
    (n250 / "settings.json").write_text("{}")
    (n250 / "finetuned-ckpt" / "adapter_model.safetensors").write_text("")
    (n250 / "state" / "state.json").write_text('{"done": 250}')
    jobs = dry(tmp_path, "finetune", config=FT)
    assert jobs["train_segment"] == 1
