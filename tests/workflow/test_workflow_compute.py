"""Where jobs run: one coordinator run placing jobs on its workers."""

import json
import os
import subprocess
from pathlib import Path

import pytest
from flowlib import compute, machine

ROOT = Path(__file__).parents[2]

CFG = {
    "COMPUTE": "ssh",
    "ROOT": "/srv/rounds",
    "RUN": "/srv/rounds/checkpoints/r6",
    "SCORE_CACHE": "/srv/rounds/scores",
    "WORKERS": "archie=local laptop=mac:/Users/me/rounds:/Users/me/repo",
    "GROUPS": "train=laptop score=archie",
}


def test_single_is_the_default_and_has_no_workers():
    assert compute.mode({}) == "single"
    assert compute.workers({}) == {}


def test_workers_are_named_hosts_with_their_root_and_checkout():
    got = compute.workers(CFG, checkout="/home/me/repo")
    assert list(got) == ["archie", "laptop"]
    assert got["archie"] == compute.Worker(
        "archie", "local", "/srv/rounds", "/home/me/repo"
    )
    assert got["laptop"] == compute.Worker(
        "laptop", "mac", "/Users/me/rounds", "/Users/me/repo"
    )


def test_a_worker_needs_a_root_and_checkout_unless_local():
    with pytest.raises(ValueError, match="laptop"):
        compute.workers({**CFG, "WORKERS": "laptop=mac"}, checkout="/r")


def test_groups_name_a_worker_or_any():
    got = compute.groups(CFG, ["archie", "laptop"])
    assert got == {"train": "laptop", "score": "archie"}
    with pytest.raises(ValueError, match="nowhere"):
        compute.groups({**CFG, "GROUPS": "train=nowhere"}, ["archie"])
    with pytest.raises(ValueError, match="group"):
        compute.groups({**CFG, "GROUPS": "trains=laptop"}, ["laptop"])


def test_every_heavy_rule_has_a_group():
    assert compute.group_of("train_segment") == "train"
    assert compute.group_of("simulate_batch") == "simulate"
    assert compute.group_of("score_germany") == "score"
    assert compute.group_of("an_unlisted_rule") == "any"
    assert set(compute.RULE_GROUPS.values()) <= set(compute.GROUPS)


def test_paths_under_the_root_or_checkout_move_to_the_worker():
    laptop = compute.workers(CFG, checkout="/home/me/repo")["laptop"]
    at = compute.mapper("/srv/rounds", "/home/me/repo", laptop)
    assert (
        at("/srv/rounds/checkpoints/r6/x")
        == "/Users/me/rounds/checkpoints/r6/x"
    )
    assert at("output/foo.txt") == "/Users/me/repo/output/foo.txt"
    assert at("/home/me/repo/data/a.csv") == "/Users/me/repo/data/a.csv"
    with pytest.raises(ValueError, match="/etc/passwd"):
        at("/etc/passwd")


def test_a_job_elsewhere_takes_the_run_settings_but_not_the_machine_ones():
    laptop = compute.workers(CFG, checkout="/home/me/repo")["laptop"]
    cfg = {
        **CFG,
        "SIMS_FROM": "local",
        "DEVICE": "cuda",
        "HEADROOM_GB": "8",
        "HOST_NAME": "archie",
        "HF_HOME": "/srv/hf",
    }
    env = compute.job_env(cfg, laptop)
    assert env["ROOT"] == "/Users/me/rounds"
    assert env["RUN"] == "/Users/me/rounds/checkpoints/r6"
    assert env["SCORE_CACHE"] == "/Users/me/rounds/scores"
    assert env["SIMS_FROM"] == "local"
    assert env["COMPUTE"] == "single"
    for key in ("DEVICE", "HEADROOM_GB", "HOST_NAME", "HF_HOME", "WORKERS"):
        assert key not in env


def test_a_job_fits_a_worker_with_room_left():
    budget = {"cores": 8, "mem_mb": 16000, "gpu": 1}
    need = {"cores": 2, "mem_mb": 4000, "gpu": 1}
    assert compute.fits(need, budget, [])
    assert not compute.fits(need, budget, [{"cores": 1, "mem_mb": 1, "gpu": 1}])
    assert not compute.fits({**need, "mem_mb": 20000}, budget, [])
    assert compute.fits({**need, "gpu": 0}, budget, [need])


def test_placement_takes_the_group_worker_or_the_first_that_fits():
    budgets = {
        "archie": {"cores": 4, "mem_mb": 8000, "gpu": 1},
        "laptop": {"cores": 8, "mem_mb": 16000, "gpu": 1},
    }
    need = {"cores": 2, "mem_mb": 4000, "gpu": 0}
    busy = {"archie": [{"cores": 4, "mem_mb": 0, "gpu": 0}], "laptop": []}
    assert compute.place("laptop", need, budgets, busy) == "laptop"
    assert compute.place("archie", need, budgets, busy) is None
    assert compute.place("any", need, budgets, busy) == "laptop"
    idle = {"archie": [], "laptop": []}
    assert compute.place("any", need, budgets, idle) == "archie"


def test_single_keeps_this_machine_and_the_default_profile():
    cfg = {"CORES": "8", "MEM_MB": "32768"}
    got = compute.args(cfg, path="")
    assert got[:2] == ["--workflow-profile", "workflow/profiles/default"]
    assert got[2:] == machine.args(cfg, path="")


def test_ssh_pools_the_workers_and_submits_through_the_scripts():
    budgets = {
        "archie": {"cores": 12, "mem_mb": 50000, "gpu": 1},
        "laptop": {"cores": 10, "mem_mb": 24000, "gpu": 1},
    }
    got = compute.args({**CFG, "SIM_SLOTS": "6"}, budgets=budgets)
    assert got[:2] == ["--workflow-profile", "workflow/profiles/ssh"]
    assert got[got.index("--jobs") + 1] == "22"
    res = got[got.index("--resources") + 1 :]
    assert "mem_mb=74000" in res
    assert "gpu=2" in res
    assert "sim_slots=6" in res


def test_the_ssh_profile_names_the_scripts():
    text = (ROOT / "workflow" / "profiles" / "ssh" / "config.yaml").read_text()
    for name in ("submit", "status", "cancel"):
        script = f"workflow/scripts/remote/{name}.py"
        assert script in text
        assert (ROOT / script).is_file()


def test_the_budget_prints_as_json(tmp_path):
    local = tmp_path / "local.env"
    local.write_text("CORES=2\nMEM_MB=4096\n")
    env = {**os.environ, "LOCAL_CONFIG": str(local)}
    script = ROOT / "workflow" / "flowlib" / "machine.py"
    out = subprocess.run(
        ["python3", str(script), "budget"],
        env=env,
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    got = json.loads(out)
    assert got["cores"] == 2
    assert got["mem_mb"] == 4096
    assert got["gpu"] >= 1


def test_the_jobscript_gives_the_job_and_its_snakemake_arguments():
    script = (
        "#!/bin/sh\n"
        '# properties = {"rule": "b", "input": ["out/a-1.txt"], '
        '"output": ["out/b-1.txt"], "log": [], "threads": 2, '
        '"resources": {"mem_mb": 100, "gpu": 1}, "params": {}, "jobid": 1}\n'
        "cd /home/me/repo && /venv/bin/python -m snakemake "
        "--snakefile '/home/me/repo/Snakefile' --target-jobs 'b:x=1' "
        "--allowed-rules b --cores 'all' --attempt 1 --force-use-threads "
        "--wait-for-files '/home/me/repo/.snakemake/tmp.1' 'out/a-1.txt' "
        "--force --nolock --config 'rounds=[21]' --latency-wait 5 "
        "--scheduler-solver-path '/venv/bin' --mode 'remote' && touch x || "
        "(touch y; exit 1)\n"
    )
    job, argv = compute.read_jobscript(script)
    assert job["rule"] == "b"
    assert job["output"] == ["out/b-1.txt"]
    assert "--target-jobs" in argv
    assert "b:x=1" in argv
    assert argv[argv.index("--cores") + 1] == "2"
    assert "rounds=[21]" in argv
    for gone in ("--snakefile", "--wait-for-files", "--scheduler-solver-path"):
        assert gone not in argv
    assert "/home/me/repo/.snakemake/tmp.1" not in argv
