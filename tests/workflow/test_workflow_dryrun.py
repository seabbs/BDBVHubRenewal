"""Dry runs of the workflow's targets against a temporary machine."""

import gzip
import json
import re
import subprocess
import sys
from pathlib import Path

import pytest
import yaml
from flowlib import simulator

pytest.importorskip("snakemake")

ROOT = Path(__file__).parents[2]
SNAKEMAKE = Path(sys.executable).parent / "snakemake"


def dry(tmp_path, *targets, config=None, extra=()):
    """Return the jobs per rule of a dry run of `targets`.

    `config` is merged into workflow/config.yaml; `extra` are more
    Snakemake options.
    """
    out = dry_run(tmp_path, *targets, config=config, extra=extra)
    assert out.returncode == 0, out.stderr + out.stdout
    jobs = dict(re.findall(r"^(\w+)\s+(\d+)$", out.stdout, re.M))
    return {k: int(v) for k, v in jobs.items() if k != "total"}


def dry_run(tmp_path, *targets, config=None, extra=()):
    """Return the finished process of a dry run of `targets`.

    Programs in <tmp_path>/bin (a stand-in `gh`) come first on PATH.
    """
    local = tmp_path / "local.env"
    if not local.exists():
        local.write_text(f"ROOT={tmp_path}\nHOST_NAME=test-host\n")
    overlay = tmp_path / "overlay.yaml"
    overlay.write_text(yaml.safe_dump(config or {}))
    return subprocess.run(
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
            *extra,
        ],
        cwd=ROOT,
        env={
            "PATH": f"{tmp_path}/bin:{SNAKEMAKE.parent}:/usr/bin:/bin",
            "HOME": str(tmp_path),
            "LOCAL_CONFIG": str(local),
        },
        capture_output=True,
        text=True,
    )


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


def _enw_lib(tmp_path):
    """The pinned epinowcast's library under the machine's ROOT."""
    return tmp_path / "rlib" / f"epinowcast-{simulator.epinowcast_sha()}"


def test_simulate_installs_the_pinned_epinowcast_first(tmp_path):
    # Without it the simulator would load whatever epinowcast R finds.
    jobs = dry(tmp_path, "simulate", config={"rounds": [4]})
    assert jobs["epinowcast_lib"] == 1


def test_the_pinned_epinowcast_is_installed_once_a_machine(tmp_path):
    lib = _enw_lib(tmp_path)
    lib.mkdir(parents=True)
    (lib / ".installed").write_text("")
    jobs = dry(tmp_path, "simulate", config={"rounds": [4]})
    assert "epinowcast_lib" not in jobs
    assert jobs["simulate_batch"] == 40


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


def test_only_generic_profiles_ship():
    profiles = (ROOT / "workflow" / "profiles").iterdir()
    # ssh names no machine: its workers are in the local config.
    assert sorted(p.name for p in profiles) == ["ci", "default", "ssh"]


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


RELEASE = {**FT, "finetune": {**FT["finetune"], "sims_from": "release"}}


def _released(tmp_path, seeds):
    """Release `seeds`: a stand-in `gh` lists them as sims-r<N> assets."""
    bin_ = tmp_path / "bin"
    bin_.mkdir(exist_ok=True)
    rounds = {}
    for s in seeds:
        rounds.setdefault((s - 30000) // 100, []).append(s)
    for r, got in rounds.items():
        names = [f"counts-{s}.csv.gz" for s in got]
        names += [f"settings-{s}.csv" for s in got]
        (tmp_path / f"sims-r{r}.json").write_text(
            json.dumps({"assets": [{"name": n} for n in names]})
        )
    gh = bin_ / "gh"
    # `gh release view sims-r<N> ...`: the round's assets, or not found.
    gh.write_text(
        "#!/bin/sh\n"
        f'f="{tmp_path}/$3.json"\n'
        '[ -f "$f" ] && exec cat "$f"\n'
        "echo 'release not found' >&2; exit 1\n"
    )
    gh.chmod(0o755)


def _stale_list(tmp_path, seeds):
    """A batch list an earlier run wrote."""
    where = tmp_path / "checkpoints" / "slots"
    where.mkdir(parents=True, exist_ok=True)
    (where / "available-batches.txt").write_text(
        "".join(f"{s}\n" for s in seeds)
    )


def test_finetune_takes_batches_from_releases(tmp_path):
    _released(tmp_path, [*range(30100, 30140), *range(30200, 30240)])
    jobs = dry(tmp_path, "finetune", config=RELEASE)
    assert jobs["fetch_batch"] == 65
    assert "simulate_batch" not in jobs


def test_from_releases_only_released_batches_are_trained_on(tmp_path):
    # The held-out round and 13 batches of round 2: 273 outbreaks, enough
    # for n250 and its spare batch but not n500.
    _released(tmp_path, [*range(30100, 30140), *range(30200, 30213)])
    jobs = dry(tmp_path, "finetune", config=RELEASE)
    assert jobs["train_segment"] == 1
    # 12 batches hold 252 outbreaks, then one spare, then the held-out.
    assert jobs["fetch_batch"] == 12 + 1 + 40


def test_from_releases_nothing_trains_before_the_held_out_round_is_in(tmp_path):
    _released(tmp_path, [*range(30100, 30139), *range(30200, 30240)])
    jobs = dry(tmp_path, "finetune", config=RELEASE)
    assert "train_segment" not in jobs


def test_from_releases_each_run_lists_them_as_it_plans(tmp_path):
    # No listing job: the run lists the releases itself, and records it.
    _released(tmp_path, [*range(30100, 30140)])
    jobs = dry(tmp_path, "finetune", config=RELEASE)
    assert "available_batches" not in jobs
    record = tmp_path / "checkpoints" / "slots" / "available-batches.txt"
    assert record.read_text().split() == [str(s) for s in range(30100, 30140)]


REMOTE = {**FT, "finetune": {**FT["finetune"], "sims_from": "remote"}}


def test_from_the_rounds_host_only_listed_batches_are_trained_on(tmp_path):
    # A machine without the rounds streams them from the rounds host's
    # manifests, as they are simulated, before any release.
    _rounds_host(tmp_path, {1: range(30100, 30140), 2: range(30200, 30213)})
    jobs = dry(tmp_path, "finetune", config=REMOTE)
    assert jobs["train_segment"] == 1
    assert jobs["fetch_batch"] == 12 + 1 + 40
    assert "simulate_batch" not in jobs


def _records(tmp_path, r, seeds):
    """Batch records of round `r` on the rounds host, before its manifest."""
    records = tmp_path / "host" / "rounds" / f"r{r}" / "records"
    records.mkdir(parents=True)
    for s in seeds:
        (records / f"record-{s}.csv").write_text(
            f"{r},{s},block,r5,c,1,21,a,b,,1,2.0,h,t\n"
        )
    return records.parent


def test_from_the_rounds_host_a_round_is_listed_once_its_qc_ran(tmp_path):
    # QC runs once the round is whole, after its manifest: a batch it
    # later excludes must never be trained on, mid-round or in between.
    _rounds_host(tmp_path, {1: range(30100, 30140)})
    r2 = _records(tmp_path, 2, range(30200, 30213))
    jobs = dry(tmp_path, "finetune", config=REMOTE)
    assert "train_segment" not in jobs
    (r2 / "manifest.csv").write_text(
        "header\n"
        + "".join(
            f"2,{s},block,r5,c,1,21,a,b,,1,2.0,h,t\n"
            for s in range(30200, 30213)
        )
    )
    jobs = dry(tmp_path, "finetune", config=REMOTE)
    assert "train_segment" not in jobs
    (r2 / "qc").mkdir()
    (r2 / "qc" / "excluded.csv").write_text("simulation,reason,evidence\n")
    jobs = dry(tmp_path, "finetune", config=REMOTE)
    assert jobs["train_segment"] == 1
    assert jobs["fetch_batch"] == 12 + 1 + 40


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


def priorities(tmp_path, *targets, config=None):
    """Return each rule's job priority in a dry run (0 when unset)."""
    local = tmp_path / "local.env"
    local.write_text(f"ROOT={tmp_path}\nHOST_NAME=test-host\n")
    overlay = tmp_path / "overlay.yaml"
    overlay.write_text(yaml.safe_dump(config or {}))
    out = subprocess.run(
        [str(SNAKEMAKE), *targets, "-s", "workflow/Snakefile", "-n",
         "--cores", "1", "--workflow-profile", "workflow/profiles/ci",
         "--configfile", str(overlay)],
        cwd=ROOT,
        env={"PATH": str(SNAKEMAKE.parent) + ":/usr/bin:/bin",
             "HOME": str(tmp_path), "LOCAL_CONFIG": str(local)},
        capture_output=True,
        text=True,
    )  # fmt: skip
    assert out.returncode == 0, out.stderr + out.stdout
    found = {}
    for block in re.split(r"\n(?=rule \w+:)", out.stdout):
        rule = re.match(r"rule (\w+):", block)
        if rule:
            got = re.search(r"priority: ([\d.]+)", block)
            found[rule[1]] = float(got[1]) if got else 0
    return found


def batch_priorities(tmp_path, *targets, config=None):
    """Return each simulate_batch job's priority by seed in a dry run."""
    local = tmp_path / "local.env"
    local.write_text(f"ROOT={tmp_path}\nHOST_NAME=test-host\n")
    overlay = tmp_path / "overlay.yaml"
    overlay.write_text(yaml.safe_dump(config or {}))
    out = subprocess.run(
        [str(SNAKEMAKE), *targets, "-s", "workflow/Snakefile", "-n",
         "--cores", "1", "--workflow-profile", "workflow/profiles/ci",
         "--configfile", str(overlay)],
        cwd=ROOT,
        env={"PATH": str(SNAKEMAKE.parent) + ":/usr/bin:/bin",
             "HOME": str(tmp_path), "LOCAL_CONFIG": str(local)},
        capture_output=True,
        text=True,
    )  # fmt: skip
    assert out.returncode == 0, out.stderr + out.stdout
    found = {}
    for block in re.split(r"\n(?=rule \w+:)", out.stdout):
        if block.startswith("rule simulate_batch:"):
            seed = int(re.search(r"seed=(\d+)", block)[1])
            found[seed] = float(re.search(r"priority: ([\d.]+)", block)[1])
    return found


def test_a_run_over_rounds_simulates_them_in_seed_order(tmp_path):
    # Training reads batches in seed order, so the first round must not
    # wait on batches of later rounds.
    got = batch_priorities(tmp_path, "simulate", config={"rounds": [4, 5]})
    assert len(got) == 80
    seeds = sorted(got)
    assert [got[s] for s in seeds] == sorted(got.values(), reverse=True)
    assert all(15 <= p < 16 for p in got.values())
    assert len(set(got.values())) == 80


def test_the_trainer_goes_first_then_simulation_then_scoring(tmp_path):
    # One pool per machine: when jobs compete for cores or the GPU, the
    # trainer (the run's critical path) goes first, then the simulations
    # it trains on, then checkpoint scoring (score.smk, 10 and below).
    got = priorities(tmp_path, "finetune", config=FT)
    assert got["train_segment"] == 20
    assert 15 <= got["simulate_batch"] < 16


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


def test_score_computes_nothing_on_bdbv_held_out_for_the_paper(tmp_path):
    # BDBV is held out entirely (Sam, 2026-11-19 11:20): no BDBV unit,
    # zone unit or data refresh is a scoring job, and `zones` is gone.
    config = {"score": {"labels": ["slots-aaa", "bbb"]}}
    jobs = dry(tmp_path, "score", config=config)
    assert not {"score_bdbv", "score_zones", "bdbv_data"} & set(jobs)
    rules = (ROOT / "workflow" / "rules" / "score.smk").read_text()
    assert not re.search(
        r"^rule (zones|score_bdbv|score_zones|bdbv_data):", rules, re.M
    )


def test_score_runs_each_unit_for_a_checkpoint_and_the_reference(tmp_path):
    config = {"score": {"labels": ["slots-aaa", "bbb"]}}
    jobs = dry(tmp_path, "score", config=config)
    assert jobs["score_germany"] == 3
    assert jobs["score_release"] == 2
    assert jobs["score_table"] == 1
    assert "land" not in jobs


def test_units_wait_below_the_load_ceiling(tmp_path):
    blocks = score_jobs(
        tmp_path, {"score": {"labels": ["aaa"]}}, "MAX_LOAD=12\n"
    )
    units = [b for b in blocks if re.match(r"rule score_germany:", b)]
    assert units
    for block in units:
        assert f"wait_for_room.sh 20 16 {tmp_path}/scores 12 " in block


def test_units_run_the_shared_checkpoint_evaluation(tmp_path):
    blocks = score_jobs(tmp_path, {"score": {"labels": ["aaa"]}})
    units = [b for b in blocks if re.match(r"(local)?rule score_\w+:", b)]
    assert units
    for block in units:
        assert "scripts/checkpoint_scores.py" not in block
    shells = "".join(units)
    assert "-m bdbvhubrenewal.evaluation.checkpoint germany" in shells


def test_units_make_the_rows_folder_a_worker_brings_back(tmp_path):
    blocks = score_jobs(tmp_path, {"score": {"labels": ["aaa"]}})
    shells = "".join(blocks)
    assert f"mkdir -p {tmp_path}/scores/germany/aaa/" in shells


def test_skipped_checkpoints_are_not_scored(tmp_path):
    config = {"score": {"labels": None, "skip": ["slots-aaa"]}}
    cache = tmp_path / "scores"
    cache.mkdir()
    at = "2026-11-16T12:00:00Z"
    (cache / "releases.json").write_text(
        f'[{{"tagName": "chronos-slots-aaa", "publishedAt": "{at}"}},'
        f' {{"tagName": "chronos-ft-bbb", "publishedAt": "{at}"}}]'
    )
    jobs = dry(tmp_path, "score", config=config)
    assert jobs["score_release"] == 1
    assert jobs["score_germany"] == 2


def test_a_checkpoint_released_after_the_tables_is_scored(tmp_path):
    # With mtime triggers alone, a new label's units under tables already
    # built were "missing, no update": every tick did nothing (r5 n4000,
    # 2026-11-19).
    cache = tmp_path / "scores"
    for unit, labels in (
        ("germany", ["reference", "slots-aaa"]),
        ("heldout", ["slots-aaa"]),
    ):
        (cache / unit).mkdir(parents=True)
        for label in labels:
            (cache / unit / f"{label}.done").write_text("")
    at = "2026-11-17T12:00:00Z"
    (cache / "releases.json").write_text(
        f'[{{"tagName": "chronos-slots-aaa", "publishedAt": "{at}"}},'
        f' {{"tagName": "chronos-slots-bbb", "publishedAt": "{at}"}}]'
    )
    (cache / "tables").mkdir()
    for name in ("checkpoint-scores.csv", "checkpoints.csv"):
        (cache / "tables" / name).write_text("x\n")
    jobs = dry(tmp_path, "score")
    assert jobs["score_germany"] == 1
    assert jobs["score_release"] == 1


def test_score_takes_checkpoints_from_the_release_list(tmp_path):
    cache = tmp_path / "scores"
    cache.mkdir()
    (cache / "releases.json").write_text(
        '[{"tagName": "chronos-slots-aaa",'
        ' "publishedAt": "2026-11-16T12:00:00Z"}]'
    )
    jobs = dry(tmp_path, "score")
    assert jobs["score_release"] == 1
    assert "release_list" not in jobs


def test_finished_units_are_kept(tmp_path):
    (tmp_path / "scores" / "germany").mkdir(parents=True)
    (tmp_path / "scores" / "germany" / "reference.done").write_text("")
    config = {"score": {"labels": ["aaa"]}}
    jobs = dry(tmp_path, "score", config=config)
    assert jobs["score_germany"] == 1


def test_score_runs_no_follow_ups_by_default(tmp_path):
    # Their scripts are not on main (test_every_script_a_rule_runs_exists).
    config = {"score": {"labels": ["slots-aaa", "bbb"]}}
    jobs = dry(tmp_path, "score", config=config)
    assert not {"slots_check_shard", "agg_fit", "fetch_adapter"} & set(jobs)


def test_score_runs_the_follow_ups_of_slots_checkpoints_only(tmp_path):
    config = {
        "score": {"labels": ["slots-aaa", "bbb"]},
        "slots_check": {"run": True},
        "agg": {"run": True},
    }
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

    cache = tmp_path / "scores"
    for study in ("germany", "heldout"):
        (cache / study).mkdir(parents=True, exist_ok=True)
    stamps = [
        cache / "germany" / "reference.done",
        cache / "germany" / "aaa.done",
        cache / "heldout" / "aaa.done",
    ]
    for f in stamps:
        f.write_text("")
    (cache / "tables").mkdir()
    for f in ("checkpoint-scores.csv", "checkpoints.csv", "landed"):
        (cache / "tables" / f).write_text("")
    old = time.time() - 3600
    for f in stamps:
        os.utime(f, (old + 60, old + 60))
    return {
        "score": {"labels": ["aaa"]},
        "slots_check": {"run": False},
        "agg": {"run": False},
    }, stamps


def test_land_runs_only_when_the_tables_changed(tmp_path):
    config, stamps = _scored(tmp_path)
    assert dry(tmp_path, "land", config=config) == {}
    stamps[1].write_text("")  # a unit finished again: newer than the table
    jobs = dry(tmp_path, "land", config=config)
    assert jobs["score_table"] == 1
    assert jobs["land"] == 1


def test_the_rounds_host_s_manifests_and_exclusions_are_mirrored(tmp_path):
    # The rounds host is this machine here (REMOTE empty), its rounds in
    # REMOTE_ROOT: round 1 whole, round 2 so far one batch.
    _rounds_host(tmp_path, {1: range(30100, 30140), 2: [30200]})
    host = tmp_path / "host"
    (host / "rounds" / "r1" / "qc" / "excluded.csv").write_text(
        "simulation\n30101-3\n"
    )
    dry(tmp_path, "finetune", config=REMOTE)
    run = tmp_path / "checkpoints" / "slots"
    listed = (run / "available-batches.txt").read_text().split()
    assert listed == [str(s) for s in [*range(30100, 30140), 30200]]
    assert (run / "rounds" / "r1" / "qc" / "excluded.csv").exists()
    assert (run / "rounds" / "r2" / "manifest.csv").exists()


NOBIAS = {**FT, "finetune": {**FT["finetune"], "series_bias": False}}


def test_a_no_bias_control_refuses_the_slots_run_s_folder(tmp_path):
    local = tmp_path / "local.env"
    local.write_text(f"ROOT={tmp_path}\nHOST_NAME=test-host\n")
    overlay = tmp_path / "overlay.yaml"
    overlay.write_text(yaml.safe_dump(NOBIAS))
    out = subprocess.run(
        [str(SNAKEMAKE), "finetune", "-s", "workflow/Snakefile", "-n",
         "--cores", "1", "--workflow-profile", "workflow/profiles/ci",
         "--configfile", str(overlay)],
        cwd=ROOT,
        env={"PATH": str(SNAKEMAKE.parent) + ":/usr/bin:/bin",
             "HOME": str(tmp_path), "LOCAL_CONFIG": str(local)},
        capture_output=True,
        text=True,
    )  # fmt: skip
    assert out.returncode != 0
    assert "RUN" in out.stderr + out.stdout


def test_a_no_bias_control_trains_in_its_own_folder(tmp_path):
    (tmp_path / "local.env").write_text(
        f"ROOT={tmp_path}\nHOST_NAME=test-host\nRUN={tmp_path}/nobias\n"
    )
    jobs = dry(tmp_path, "finetune", config=NOBIAS)
    assert jobs["train_segment"] == 2


def test_a_tick_trains_only_the_next_segment(tmp_path):
    # So each run ends after one segment and the next run plans afresh:
    # new rounds, new releases, a checkpoint to score.
    jobs = dry(tmp_path, "finetune", config={**FT, "next_segment_only": True})
    assert jobs["train_segment"] == 1


def test_a_tick_after_n250_trains_n500(tmp_path):
    _checkpoint(tmp_path, 250)
    jobs = dry(tmp_path, "finetune", config={**FT, "next_segment_only": True})
    assert jobs["train_segment"] == 1


def test_a_tick_does_not_train_on_to_the_n1000_check(tmp_path):
    # The n1000 report needs n1000, so it must not pull the chain there.
    config = {
        **FT,
        "next_segment_only": True,
        "rounds": [1, 2, 3],
        "finetune": {
            **FT["finetune"],
            "prefixes": [250, 500, 1000],
            "outbreaks": 1000,
            "rounds": [2, 3],
        },
    }
    jobs = dry(tmp_path, "finetune", config=config)
    assert jobs["train_segment"] == 1
    assert "checkpoint_check" not in jobs


def _rounds_host(tmp_path, rounds):
    """A rounds host on this machine with `rounds` {round: seeds} listed."""
    host = tmp_path / "host"
    for r, seeds in rounds.items():
        where = host / "rounds" / f"r{r}"
        (where / "qc").mkdir(parents=True)
        rows = "".join(
            f"{r},{s},block,r5,c,1,21,a,b,,1,2.0,h,t\n" for s in seeds
        )
        (where / "manifest.csv").write_text("header\n" + rows)
        (where / "qc" / "excluded.csv").write_text(
            "simulation,reason,evidence\n"
        )
    local = tmp_path / "local.env"
    local.write_text(
        f"ROOT={tmp_path}\nHOST_NAME=test-host\nREMOTE_ROOT={host}\n"
    )
    return local


def test_a_fresh_listing_keeps_the_checkpoints_saved(tmp_path):
    # Each run lists the batches anew; the newer list must not make a
    # saved (protected) checkpoint look outdated.
    _rounds_host(tmp_path, {1: range(30100, 30140), 2: range(30200, 30240)})
    _checkpoint(tmp_path, 250)
    dry(tmp_path, "finetune", config=REMOTE)
    jobs = dry(
        tmp_path, "finetune", config={**REMOTE, "next_segment_only": True}
    )
    assert jobs["train_segment"] == 1


def test_each_run_lists_the_batches_afresh(tmp_path):
    # A list written by an earlier run, when round 2 had 13 batches, must
    # not cap a later run once the rounds host has the whole round.
    _rounds_host(tmp_path, {1: range(30100, 30140), 2: range(30200, 30240)})
    _stale_list(tmp_path, [*range(30100, 30140), *range(30200, 30213)])
    jobs = dry(tmp_path, "finetune", config=REMOTE)
    assert jobs["train_segment"] == 2


def test_a_no_bias_config_still_simulates_in_the_default_folder(tmp_path):
    # The guard is the trainer's: other targets run as before.
    jobs = dry(tmp_path, "simulate", config={**NOBIAS, "rounds": [4]})
    assert jobs["simulate_batch"] == 40


def test_an_unreachable_rounds_host_is_said(tmp_path):
    # A run that cannot copy a round's manifest says so, not silence.
    (tmp_path / "local.env").write_text(
        f"ROOT={tmp_path}\nHOST_NAME=test-host\n"
        f"REMOTE_ROOT={tmp_path}/no-such-host\n"
    )
    out = dry_run(tmp_path, "finetune", config=REMOTE)
    assert "manifest.csv" in out.stderr
    assert "could not copy" in out.stderr


def score_jobs(tmp_path, config, local_extra="", target="score"):
    """Return each score rule's shell command, priority and resources."""
    local = tmp_path / "local.env"
    local.write_text(
        f"ROOT={tmp_path}\nHOST_NAME=test-host\nDEVICE=cuda\n{local_extra}"
    )
    overlay = tmp_path / "overlay.yaml"
    overlay.write_text(yaml.safe_dump(config))
    out = subprocess.run(
        [str(SNAKEMAKE), target, "-s", "workflow/Snakefile", "-n", "-p",
         "--cores", "8", "--workflow-profile", "workflow/profiles/ci",
         "--configfile", str(overlay)],
        cwd=ROOT,
        env={"PATH": str(SNAKEMAKE.parent) + ":/usr/bin:/bin",
             "HOME": str(tmp_path), "LOCAL_CONFIG": str(local)},
        capture_output=True,
        text=True,
    )  # fmt: skip
    assert out.returncode == 0, out.stderr + out.stdout
    return re.split(r"\n(?=(?:local)?rule \w+:)", out.stdout)


def test_score_units_run_torch_on_the_machine_device(tmp_path):
    config = {"score": {"labels": ["aaa", "bbb"]}}
    blocks = score_jobs(tmp_path, config)
    units = [b for b in blocks if re.match(r"rule score_germany:", b)]
    assert len(units) == 3
    for block in units:
        assert "DEVICE=cuda" in block
        assert "OMP_NUM_THREADS=1 " not in block
        assert "CUDA_VISIBLE_DEVICES" not in block
        assert re.search(r"resources: .*gpu=1", block)
        # A few threads, not the machine: simulations keep their cores.
        assert re.search(r"threads: 2\b", block)
        assert "--xla_cpu_multi_thread_eigen=false" in block


def test_score_threads_are_a_machine_setting(tmp_path):
    config = {"score": {"labels": ["aaa"]}}
    blocks = score_jobs(tmp_path, config, "SCORE_THREADS=3\n")
    units = [b for b in blocks if re.match(r"rule score_germany:", b)]
    assert units
    assert all(re.search(r"threads: 3\b", b) for b in units)


def test_newer_checkpoints_are_scored_first(tmp_path):
    config = {"score": {"labels": ["aaa", "bbb"]}}
    blocks = score_jobs(tmp_path, config)
    got = {}
    for block in blocks:
        rule = re.match(r"rule (score_germany):", block)
        if rule:
            label = re.search(r"wildcards: label=(\S+)", block)[1]
            got[label] = float(re.search(r"priority: ([\d.]+)", block)[1])
    assert got["bbb"] > got["aaa"] > got["reference"]


def test_one_unit_runs_before_the_release_list_exists(tmp_path):
    # A unit's priority reads the list if it is there, else takes its band's
    # base: it never waits on the release_list checkpoint.
    jobs = dry(tmp_path, str(tmp_path / "scores" / "germany" / "aaa.done"))
    assert jobs["score_germany"] == 1


# Each grid has its own rounds (flowlib.grids).


def test_r6_release_of_an_r5_round_is_refused(tmp_path):
    with pytest.raises(AssertionError, match="round 9 is grid r5's"):
        dry(tmp_path, "release", config={"rounds": [9]})


def test_r6_simulates_its_held_out_round_20(tmp_path):
    jobs = dry(tmp_path, "simulate", config={"rounds": [20]})
    assert jobs["simulate_batch"] == 40


def test_the_sim_qc_checks_fixed_thresholds_from_the_config(tmp_path):
    # The deterministic check, with simulate.qc's thresholds; no model.
    blocks = score_jobs(tmp_path, {"rounds": [4]}, target="simulate")
    qc = next(b for b in blocks if re.match(r"(local)?rule sim_qc:", b))
    assert "scripts/sim_qc.py" in qc
    assert "--max-cases-per-stratum 500000" in qc
    assert "--max-weekly-jump 1000" in qc
    assert "--max-mean-delay-days 35" in qc
    assert "claude" not in qc
