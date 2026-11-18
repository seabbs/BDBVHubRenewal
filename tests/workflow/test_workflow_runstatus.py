"""What a machine's workflow run is doing, from its Snakemake log."""

import json

from flowlib import runstatus

LOG = """\
Building DAG of jobs...
Provided resources: gpu=Resource("gpu", 1), mem_mb=Resource("mem_mb", 16384)

[Wed Nov 18 01:11:45 2026]
localrule release_list:
    output: cache/releases.json
    jobid: 0
    reason: Forced execution
    resources: tmpdir=/tmp, mem_mb=2048, mem=2.05 GB

[Wed Nov 18 01:11:46 2026]
Finished jobid: 0 (Rule: release_list)
1 of 4 steps (25%) done

[Wed Nov 18 01:11:46 2026]
localrule score_bdbv:
    output: cache/bdbv/slots-f23ac5d03d45.done
    jobid: 18
    wildcards: label=slots-f23ac5d03d45
    resources: tmpdir=/tmp, mem_mb=6144, mem=6.14 GB, gpu=1

[Wed Nov 18 01:11:47 2026]
rule simulate_batch:
    output: rounds/r4/sims/counts-30408.csv.gz
    jobid: 21
    wildcards: round=4, seed=30408
    resources: tmpdir=/tmp, mem_mb=16384, sim_slots=1

[Wed Nov 18 01:20:01 2026]
Error in rule simulate_batch:
    jobid: 21
    output: rounds/r4/sims/counts-30408.csv.gz

[Wed Nov 18 01:20:02 2026]
rule simulate_batch:
    output: rounds/r4/sims/counts-30408.csv.gz
    jobid: 21
    wildcards: round=4, seed=30408
    resources: tmpdir=/tmp, mem_mb=20480, sim_slots=1

[Wed Nov 18 01:21:00 2026]
rule simulate_batch:
    output: rounds/r4/sims/counts-30409.csv.gz
    jobid: 22
    wildcards: round=4, seed=30409
    resources: tmpdir=/tmp, mem_mb=16384, sim_slots=1

[Wed Nov 18 01:30:00 2026]
Error in rule simulate_batch:
    jobid: 22
"""


def jobs_by_id(text):
    return {j["jobid"]: j for j in runstatus.parse_log(text)}


def test_a_job_started_and_not_finished_is_running():
    jobs = jobs_by_id(LOG)
    assert jobs[18]["state"] == "running"
    assert jobs[18]["rule"] == "score_bdbv"
    assert jobs[18]["wildcards"] == "label=slots-f23ac5d03d45"
    assert jobs[18]["started"] == "2026-11-18T01:11:46"


def test_a_finished_job_is_done_with_its_times():
    job = jobs_by_id(LOG)[0]
    assert job["state"] == "done"
    assert job["ended"] == "2026-11-18T01:11:46"


def test_a_retried_job_runs_again_with_its_retry_resources():
    # simulate_batch retries once with more memory: not a failure.
    job = jobs_by_id(LOG)[21]
    assert job["state"] == "running"
    assert job["resources"]["mem_mb"] == "20480"
    assert job["started"] == "2026-11-18T01:20:02"


def test_a_job_that_errs_without_a_retry_has_failed():
    job = jobs_by_id(LOG)[22]
    assert job["state"] == "failed"
    assert job["ended"] == "2026-11-18T01:30:00"


def test_only_the_resources_that_place_a_job_are_kept():
    assert jobs_by_id(LOG)[18]["resources"] == {"mem_mb": "6144", "gpu": "1"}


def test_the_status_says_who_and_what_and_when(tmp_path):
    log = tmp_path / "run.snakemake.log"
    log.write_text(LOG)
    got = runstatus.status(
        {"HOST_NAME": "host-a", "STATUS_EVERY": "60"},
        step="simulate finetune score",
        log=log,
        alive=True,
        now="2026-11-18T01:31:00",
    )
    assert got["host"] == "host-a"
    assert got["step"] == "simulate finetune score"
    assert got["alive"] is True
    assert got["every"] == 60
    assert got["updated"] == "2026-11-18T01:31:00"
    assert [j["jobid"] for j in got["running"]] == [18, 21]
    assert [j["jobid"] for j in got["recent"]] == [22, 0]


def test_a_status_without_a_run_says_why():
    got = runstatus.status(
        {"HOST_NAME": "host-a"},
        step="skipped: another tick is running",
        log=None,
        alive=False,
        now="2026-11-18T01:31:00",
    )
    assert got["running"] == []
    assert got["step"] == "skipped: another tick is running"


def test_a_status_file_is_replaced_whole(tmp_path):
    cfg = {"HOST_NAME": "host-a", "ROOT": str(tmp_path)}
    got = runstatus.status(cfg, step="idle", log=None, alive=False)
    path = runstatus.write(got, cfg)
    assert path == tmp_path / "status" / "host-a.json"
    assert json.loads(path.read_text())["host"] == "host-a"
    assert not list(path.parent.glob("*.tmp"))


def test_the_newest_log_of_a_run_is_found(tmp_path):
    logs = tmp_path / ".snakemake" / "log"
    logs.mkdir(parents=True)
    (logs / "2026-11-18T010000.000000.snakemake.log").write_text("a")
    (logs / "2026-11-18T020000.000000.snakemake.log").write_text("b")
    assert runstatus.newest_log(tmp_path).name.startswith("2026-11-18T02")
    assert runstatus.newest_log(tmp_path / "none") is None
