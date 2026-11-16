"""Dry runs of the workflow's targets against a temporary machine."""

import gzip
import re
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

pytest.importorskip("snakemake")

ROOT = Path(__file__).parents[2]
SNAKEMAKE = Path(sys.executable).parent / "snakemake"


def dry(tmp_path, *targets, config=None):
    """Return the jobs per rule of a dry run of `targets`.

    `config` is merged into workflow/config.yaml.
    """
    local = tmp_path / "local.env"
    if not local.exists():
        local.write_text(f"ROOT={tmp_path}\nHOST_NAME=test-host\n")
    overlay = tmp_path / "overlay.yaml"
    overlay.write_text(yaml.safe_dump(config or {}))
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
            *(["--configfile", str(overlay)] if config else []),
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
    jobs = dry(tmp_path, "simulate", config={"rounds": [4]})
    assert jobs["simulate_batch"] == 44
    assert jobs["batch_record"] == 44
    assert jobs["round_manifest"] == 1
    assert jobs["sim_qc"] == 1
    assert jobs["round_ready"] == 1


def test_simulate_never_releases(tmp_path):
    jobs = dry(tmp_path, "simulate", config={"rounds": [4]})
    assert not any(k.startswith("release") for k in jobs)


def test_release_is_its_own_target(tmp_path):
    jobs = dry(tmp_path, "release", config={"rounds": [4]})
    assert jobs["release_batch"] == 44


def test_batches_already_simulated_are_kept(tmp_path):
    sims = tmp_path / "rounds" / "r4" / "sims"
    sims.mkdir(parents=True)
    for seed in (30400, 30401):
        with gzip.open(sims / f"counts-{seed}.csv.gz", "wt") as f:
            f.write("x\n")
        (sims / f"settings-{seed}.csv").write_text("outbreak\n")
    jobs = dry(tmp_path, "simulate", config={"rounds": [4]})
    assert jobs["simulate_batch"] == 42


FT = {
    "rounds": [1],
    "finetune": {
        "prefixes": [250, 500],
        "outbreaks": 500,
        "rounds": [1],
        "held_round": 1,
        "sims_from": "local",
        "real_validation": None,
    },
}


def test_finetune_trains_one_segment_per_prefix(tmp_path):
    jobs = dry(tmp_path, "finetune", config=FT)
    assert jobs["train_segment"] == 2
    # 500 outbreaks are 20 batches, one spare, and 4 held-out batches.
    assert jobs["fetch_batch"] == 25
    # On one machine the batches are simulated here.
    assert jobs["simulate_batch"] == 25


def test_finetune_takes_batches_from_releases(tmp_path):
    config = {**FT, "finetune": {**FT["finetune"], "sims_from": "release"}}
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


def _checkpoint(tmp_path, n):
    where = tmp_path / "checkpoints" / "slots" / "train" / f"n{n}"
    (where / "finetuned-ckpt").mkdir(parents=True)
    (where / "state").mkdir()
    (where / "settings.json").write_text("{}")
    (where / "finetuned-ckpt" / "adapter_model.safetensors").write_text("")
    (where / "state" / "state.json").write_text(f'{{"done": {n}}}')


def test_publish_verifies_then_releases_each_checkpoint(tmp_path):
    for n in (250, 500):
        _checkpoint(tmp_path, n)
    jobs = dry(tmp_path, "publish", config=FT)
    assert jobs["verify_checkpoint"] == 2
    assert jobs["publish_checkpoint"] == 2
    assert "train_segment" not in jobs


def test_verify_alone_never_releases(tmp_path):
    _checkpoint(tmp_path, 250)
    jobs = dry(
        tmp_path,
        f"{tmp_path}/checkpoints/slots/train/n250/verified.txt",
        config=FT,
    )
    assert jobs == {"verify_checkpoint": 1}


def test_score_runs_three_units_a_checkpoint_and_the_reference(tmp_path):
    config = {"score": {"labels": ["slots-aaa", "bbb"]}}
    jobs = dry(tmp_path, "score", config=config)
    assert jobs["score_bdbv"] == 3
    assert jobs["score_germany"] == 3
    assert jobs["score_heldout"] == 2
    assert jobs["score_table"] == 1
    assert "land" not in jobs


def test_score_takes_checkpoints_from_the_release_list(tmp_path):
    cache = tmp_path / "scores"
    cache.mkdir()
    (cache / "releases.json").write_text(
        '[{"tagName": "chronos-slots-aaa",'
        ' "publishedAt": "2026-11-16T12:00:00Z"}]'
    )
    jobs = dry(tmp_path, "score")
    assert jobs["score_heldout"] == 1
    assert "release_list" not in jobs


def test_finished_units_are_kept(tmp_path):
    (tmp_path / "scores" / "bdbv").mkdir(parents=True)
    (tmp_path / "scores" / "bdbv" / "reference.done").write_text("")
    config = {"score": {"labels": ["aaa"]}}
    jobs = dry(tmp_path, "score", config=config)
    assert jobs["score_bdbv"] == 1


def test_score_runs_the_follow_ups_of_slots_checkpoints_only(tmp_path):
    config = {"score": {"labels": ["slots-aaa", "bbb"]}}
    jobs = dry(tmp_path, "score", config=config)
    assert jobs["slots_check_shard"] == 2
    assert jobs["slots_check_summary"] == 1
    assert jobs["agg_fit"] == 2
    assert jobs["agg_rows"] == 1
    assert jobs["agg_sanity"] == 1
    assert jobs["fetch_adapter"] == 1
