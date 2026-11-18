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
    assert jobs["simulate_batch"] == 40
    assert jobs["batch_record"] == 40
    assert jobs["round_manifest"] == 1
    assert jobs["sim_qc"] == 1
    assert jobs["round_ready"] == 1


def test_simulate_never_releases(tmp_path):
    jobs = dry(tmp_path, "simulate", config={"rounds": [4]})
    assert not any(k.startswith("release") for k in jobs)


def test_release_is_its_own_target(tmp_path):
    jobs = dry(tmp_path, "release", config={"rounds": [4]})
    assert jobs["release_batch"] == 40


def test_batches_already_simulated_are_kept(tmp_path):
    sims = tmp_path / "rounds" / "r4" / "sims"
    sims.mkdir(parents=True)
    for seed in (30400, 30401):
        with gzip.open(sims / f"counts-{seed}.csv.gz", "wt") as f:
            f.write("x\n")
        (sims / f"settings-{seed}.csv").write_text("outbreak\n")
    jobs = dry(tmp_path, "simulate", config={"rounds": [4]})
    assert jobs["simulate_batch"] == 38


SPLIT = {"rounds": [10]}


def _host(tmp_path, seeds):
    (tmp_path / "local.env").write_text(
        f"ROOT={tmp_path}\nHOST_NAME=test-host\nSIM_SEEDS={seeds}\n"
    )


def test_a_round_shared_with_another_host_waits_for_its_seeds(tmp_path):
    _host(tmp_path, '"31000-31004 31006 31008-31039"')
    jobs = dry(tmp_path, "simulate", config=SPLIT)
    assert jobs["simulate_batch"] == 38
    assert jobs["await_batch"] == 2
    assert jobs["batch_record"] == 40


def test_a_second_host_simulates_and_pushes_only_its_seeds(tmp_path):
    _host(tmp_path, '"31005 31007"')
    jobs = dry(tmp_path, "push", config=SPLIT)
    assert jobs["simulate_batch"] == 2
    assert jobs["push_batch"] == 2
    assert "batch_record" not in jobs
    assert "await_batch" not in jobs


def test_a_host_can_take_whole_rounds(tmp_path):
    _host(tmp_path, "r10")
    jobs = dry(tmp_path, "push", config={"rounds": [10, 11]})
    assert jobs["push_batch"] == 40


def test_no_host_is_named_in_the_committed_workflow():
    profiles = (ROOT / "workflow" / "profiles").iterdir()
    assert sorted(p.name for p in profiles) == ["ci", "default"]
    for path in [
        ROOT / "workflow" / "config.yaml",
        ROOT / "workflow" / "README.md",
        ROOT / "config" / "local.env.example",
        ROOT / "config" / "profile.example" / "config.yaml",
        *(ROOT / "workflow" / "rules").glob("*.smk"),
    ]:
        assert "archie" not in path.read_text().lower(), path


FT = {
    "rounds": [1, 2],
    "finetune": {
        "prefixes": [250, 500],
        "outbreaks": 500,
        "rounds": [2],
        "held_round": 1,
        "sims_from": "local",
        "real_validation": None,
    },
}


def test_finetune_trains_one_segment_per_prefix(tmp_path):
    jobs = dry(tmp_path, "finetune", config=FT)
    assert jobs["train_segment"] == 2
    # 500 outbreaks are 24 batches of 21, one spare, and the held-out
    # round's 40 batches.
    assert jobs["fetch_batch"] == 65
    # On one machine the batches are simulated here.
    assert jobs["simulate_batch"] == 65


def test_finetune_takes_batches_from_releases(tmp_path):
    config = {**FT, "finetune": {**FT["finetune"], "sims_from": "release"}}
    jobs = dry(tmp_path, "finetune", config=config)
    assert jobs["fetch_batch"] == 65
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


def test_the_live_run_switches_over_without_retraining(tmp_path):
    # The laptop's slots run: n250 to n750 saved without state of their
    # own, the trainer's state part way to n1000, and a seed list that
    # reaches 1,500 outbreaks so far.
    train = tmp_path / "checkpoints" / "slots" / "train"
    for n in (250, 500, 750):
        where = train / f"n{n}"
        (where / "finetuned-ckpt").mkdir(parents=True)
        (where / "settings.json").write_text("{}")
        (where / "finetuned-ckpt" / "adapter_model.safetensors").write_text("")
    (train / "state").mkdir()
    (train / "state" / "state.json").write_text('{"done": 875}')
    seeds = tmp_path / "seeds.txt"
    seeds.write_text("".join(f"{30100 + i}\n" for i in range(60)))
    config = {
        **FT,
        "finetune": {
            **FT["finetune"],
            "prefixes": [250, 500, 750, 1000, 2000],
            "outbreaks": 2000,
            "seed_file": str(seeds),
        },
    }
    jobs = dry(tmp_path, "finetune", config=config)
    # Only n1000: n2000 needs more seeds than the list has.
    assert jobs["train_segment"] == 1


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


def test_german_forecasts_are_off_by_default(tmp_path):
    config = {"score": {"labels": ["slots-aaa", "bbb"]}}
    jobs = dry(tmp_path, "score", config=config)
    assert "score_germany_forecast" not in jobs
    assert "score_germany_forecast" not in dry(
        tmp_path, "germany_forecast", config=config
    )


def test_german_forecasts_run_the_reference_and_one_checkpoint(tmp_path):
    forecast = {"enabled": True, "label": "slots-aaa"}
    config = {
        "score": {"labels": ["slots-aaa", "bbb"], "germany_forecast": forecast}
    }
    jobs = dry(tmp_path, "germany_forecast", config=config)
    assert jobs["score_germany_forecast"] == 2
    # Each after its own German nowcasts.
    assert jobs["score_germany"] == 2
    assert dry(tmp_path, "score", config=config)["score_germany_forecast"] == 2


def test_german_forecasts_of_the_reference_alone(tmp_path):
    forecast = {"enabled": True, "label": None}
    config = {"score": {"labels": ["bbb"], "germany_forecast": forecast}}
    jobs = dry(tmp_path, "germany_forecast", config=config)
    assert jobs["score_germany_forecast"] == 1


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
    data = tmp_path / "target.csv"
    data.write_text("x\n")
    (tmp_path / "scores" / "bdbv").mkdir(parents=True)
    (tmp_path / "scores" / "bdbv" / "reference.done").write_text("")
    config = {"score": {"labels": ["aaa"], "data": str(data)}}
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


def _local(tmp_path, *lines):
    (tmp_path / "local.env").write_text(
        "\n".join([f"ROOT={tmp_path}", "HOST_NAME=test-host", *lines]) + "\n"
    )


def test_publish_takes_only_checkpoints_already_saved(tmp_path):
    _checkpoint(tmp_path, 250)
    jobs = dry(tmp_path, "publish", config=FT)
    assert jobs["publish_checkpoint"] == 1
    assert "train_segment" not in jobs


def test_a_machine_can_publish_as_it_trains(tmp_path):
    _local(tmp_path, "PUBLISH=1")
    jobs = dry(tmp_path, "finetune", config=FT)
    assert jobs["publish_checkpoint"] == 2


def test_a_machine_runs_only_its_own_targets(tmp_path):
    _local(tmp_path, "TARGETS=score")
    config = {**FT, "score": {"labels": ["aaa"]}}
    jobs = dry(tmp_path, "all", config=config)
    assert "train_segment" not in jobs
    assert "simulate_batch" not in jobs
    assert jobs["score_table"] == 1


def test_where_batches_come_from_is_the_machine_s(tmp_path):
    _local(tmp_path, "SIMS_FROM=release")
    jobs = dry(tmp_path, "finetune", config=FT)
    assert "simulate_batch" not in jobs


def test_scores_follow_the_data(tmp_path):
    jobs = dry(tmp_path, "score", config={"score": {"labels": ["aaa"]}})
    assert jobs["bdbv_data"] == 1


def test_release_carries_each_round_s_exclusions(tmp_path):
    jobs = dry(tmp_path, "release", config={"rounds": [4]})
    assert jobs["release_qc"] == 1


def test_at_most_two_simulations_run_at_once(tmp_path):
    local = tmp_path / "local.env"
    local.write_text(f"ROOT={tmp_path}\n")
    out = subprocess.run(
        [
            str(SNAKEMAKE),
            "simulate",
            "-n",
            "-p",
            "--cores",
            "8",
            "--workflow-profile",
            "workflow/profiles/default",
            "--config",
            "rounds=[4]",
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
    assert "sim_slots=1" in out.stdout


def test_a_failed_run_posts_a_mailbox_line(tmp_path):
    # A real run of one local rule that fails: n1000_check on settings
    # with no losses.
    box = tmp_path / "box.md"
    _local(tmp_path, "NOTIFY=1", f"MAILBOX={box}")
    n1000 = tmp_path / "checkpoints" / "slots" / "train" / "n1000"
    n1000.mkdir(parents=True)
    (n1000 / "settings.json").write_text("{}")
    target = tmp_path / "checkpoints" / "slots" / "reports" / "n1000-check.txt"
    out = subprocess.run(
        [
            str(SNAKEMAKE),
            str(target),
            "--cores",
            "1",
            "--workflow-profile",
            "workflow/profiles/ci",
        ],
        cwd=ROOT,
        env={
            "PATH": str(SNAKEMAKE.parent) + ":/usr/bin:/bin",
            "HOME": str(tmp_path),
            "LOCAL_CONFIG": str(tmp_path / "local.env"),
        },
        capture_output=True,
        text=True,
    )
    assert out.returncode != 0
    line = box.read_text()
    assert "FAILED" in line
    assert ".snakemake/log/" in line


def test_the_check_follows_the_configured_prefix(tmp_path):
    config = {
        **FT,
        "finetune": {**FT["finetune"], "check_at": 500},
    }
    jobs = dry(tmp_path, "finetune", config=config)
    assert jobs["checkpoint_check"] == 1


def test_the_dashboard_render_is_its_own_target(tmp_path):
    assert dry(tmp_path, "dashboard") == {"dashboard": 1}
    jobs = dry(tmp_path, "all", config={**FT, "score": {"labels": ["aaa"]}})
    assert "dashboard" not in jobs


def _scored(tmp_path):
    import os
    import time

    data = tmp_path / "target.csv"
    data.write_text("x\n")
    cache = tmp_path / "scores"
    for study in ("bdbv", "germany", "heldout"):
        (cache / study).mkdir(parents=True, exist_ok=True)
    stamps = [
        cache / "bdbv" / "reference.done",
        cache / "germany" / "reference.done",
        cache / "bdbv" / "aaa.done",
        cache / "germany" / "aaa.done",
        cache / "heldout" / "aaa.done",
    ]
    for f in stamps:
        f.write_text("")
    (cache / "tables").mkdir()
    for f in ("checkpoint-scores.csv", "checkpoints.csv", "landed"):
        (cache / "tables" / f).write_text("")
    old = time.time() - 3600
    os.utime(data, (old, old))
    for f in stamps:
        os.utime(f, (old + 60, old + 60))
    return {
        "score": {"labels": ["aaa"], "data": str(data)},
        "slots_check": {"run": False},
        "agg": {"run": False},
    }, stamps


def test_land_runs_only_when_the_tables_changed(tmp_path):
    config, stamps = _scored(tmp_path)
    assert dry(tmp_path, "land", config=config) == {}
    stamps[2].write_text("")  # a unit finished again: newer than the table
    jobs = dry(tmp_path, "land", config=config)
    assert jobs["score_table"] == 1
    assert jobs["land"] == 1
