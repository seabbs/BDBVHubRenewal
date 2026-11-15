"""The unattended fine-tuning driver's fallback when the remote is down."""

import subprocess
from pathlib import Path

DRIVER = Path(__file__).parents[1] / "scripts" / "finetune_driver.sh"

ENV = """\
SIM_DIR=sims
TRAIN_FIRST=5001
TRAIN_LAST=5010
HELD_FIRST=5901
HELD_LAST=5903
LOCAL_FIRST=5009
LOCAL_LAST=5010
LOCAL_SLOTS=2
LOCAL_SLOTS_TRAINING=1
LOCAL_MIN_FREE_MEM_GB=12
REMOTE=nowhere
REMOTE_DIR=x
SIM_REMOTE=y
REMOTE_SLOTS=4
REMOTE_MEM_KB=1
REMOTE_MIN_FREE_GB=15
REMOTE_RETRY=600
REMOTE_GRACE=1800
FIRST_CHUNK_SEEDS=2
OUT=ckpt
OUTBREAKS=10
REMOTE_RAM_GB=62
GRID=g
MPS_HIGH_WATERMARK=0.8
GRID=g
"""


def run(tmp_path: Path, script: str, env: str = ENV) -> str:
    """Run bash with the driver's functions and return its output."""
    (tmp_path / "round.env").write_text(env)
    (tmp_path / "output" / "finetune").mkdir(parents=True, exist_ok=True)
    (tmp_path / "sims").mkdir(exist_ok=True)
    (tmp_path / "sims" / "settings-5902.csv").write_text("outbreak\n")
    out = subprocess.run(
        ["bash", "-c", f'set -- round.env; source "{DRIVER}"; {script}'],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=True,
    )
    return out.stdout


def test_down_remote_hands_held_out_and_first_seeds_to_the_laptop(tmp_path):
    out = run(
        tmp_path,
        "remote() { return 1; }; "
        "remote_ok=$(( $(date +%s) - 1801 )); remote_tried=0; "
        "check_remote; local_seeds | tr '\\n' ' '",
    )
    # Held-out seeds not done first, then the first training seeds, then
    # the laptop's own range.
    assert out.split() == ["5901", "5903", "5001", "5002", "5009", "5010"]


def test_remote_short_of_disk_counts_as_down(tmp_path):
    out = run(
        tmp_path,
        "remote() { echo 8G; }; "
        "remote_ok=$(( $(date +%s) - 1801 )); remote_tried=0; "
        "check_remote; cat sims/local-seeds.txt | tr '\\n' ' '",
    )
    assert out.split() == ["5901", "5903", "5001", "5002"]


def test_remote_down_for_less_than_the_grace_keeps_its_seeds(tmp_path):
    out = run(
        tmp_path,
        "remote() { return 1; }; "
        "remote_ok=$(( $(date +%s) - 60 )); remote_tried=0; "
        "check_remote; local_seeds | tr '\\n' ' '",
    )
    assert out.split() == ["5009", "5010"]


def test_short_yield_adds_training_seeds(tmp_path):
    rows = "outbreak\n" + "".join(f"x-{i}\n" for i in range(10))
    (tmp_path / "sims").mkdir()
    for s in range(5001, 5011):
        (tmp_path / "sims" / f"settings-{s}.csv").write_text(rows)
    env = ENV.replace("TRAIN_LAST=5010", "TRAIN_LAST=5020").replace(
        "OUTBREAKS=10", "OUTBREAKS=300"
    )
    (tmp_path / "round.env").write_text(env)
    (tmp_path / "output" / "finetune").mkdir(parents=True, exist_ok=True)
    (tmp_path / "ckpt").mkdir()
    out = subprocess.run(
        [
            "bash",
            "-c",
            f'set -- round.env; source "{DRIVER}"; '
            "remote() { return 1; }; extend_seeds; echo $TRAIN_LAST",
        ],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=True,
    )
    # 10 per seed, 200 projected: 115 more for 5% to spare is 12 seeds.
    assert out.stdout.split() == ["5032"]
    assert (tmp_path / "ckpt" / "train-last.txt").read_text() == "5032\n"


def test_no_remote_runs_every_seed_locally(tmp_path):
    (tmp_path / "round.env").write_text(
        ENV.replace("REMOTE=nowhere", "REMOTE=")
    )
    (tmp_path / "output" / "finetune").mkdir(parents=True)
    (tmp_path / "sims").mkdir()
    out = subprocess.run(
        ["bash", "-c", f'set -- round.env; source "{DRIVER}"; local_seeds'],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=True,
    )
    assert out.stdout.split() == [
        *map(str, range(5901, 5904)),
        *map(str, range(5001, 5011)),
    ]


def test_laptop_takes_unstarted_seeds_in_order(tmp_path):
    (tmp_path / "round.env").write_text(ENV)
    (tmp_path / "output" / "finetune").mkdir(parents=True)
    logs = tmp_path / "sims" / "logs"
    logs.mkdir(parents=True)
    for s in (5001, 5003):
        (logs / f"{s}.log").write_text("")
    out = subprocess.run(
        ["bash", "-c", f'set -- round.env; source "{DRIVER}"; unstarted_seeds'],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=True,
    )
    assert out.stdout.split()[:3] == ["5002", "5004", "5005"]


def test_remote_slots_fit_the_worst_case_memory(tmp_path):
    out = run(
        tmp_path,
        "REMOTE_SLOTS=12; REMOTE_MEM_KB=10485760; REMOTE_RAM_GB=62; "
        "remote_slots",
    )
    # 12 batches of 10 GB do not fit 62 GB less 16 GB: 4 do.
    assert out.split() == ["4"]


def test_status_reports_the_remote_as_last_seen(tmp_path):
    out = run(
        tmp_path,
        "remote() { return 1; }; remote_tried=0; check_remote; "
        "echo $remote_state",
    )
    assert out.split()[-1] == "down"


def test_down_remote_hands_over_held_out_seeds_after_other_claims(tmp_path):
    out = run(
        tmp_path,
        "echo 5005 > sims/local-seeds.txt; remote() { return 1; }; "
        "remote_ok=$(( $(date +%s) - 1801 )); remote_tried=0; "
        "check_remote; local_seeds | tr '\\n' ' '",
    )
    # Unfinished held-out seeds first, whatever was claimed before.
    assert out.split()[:2] == ["5901", "5903"]


def test_seed_extension_reopens_a_finished_remote_scheduler(tmp_path):
    rows = "outbreak\n" + "".join(f"x-{i}\n" for i in range(10))
    (tmp_path / "sims").mkdir()
    for s in range(5001, 5011):
        (tmp_path / "sims" / f"settings-{s}.csv").write_text(rows)
    env = ENV.replace("TRAIN_LAST=5010", "TRAIN_LAST=5020").replace(
        "OUTBREAKS=10", "OUTBREAKS=300"
    )
    (tmp_path / "round.env").write_text(env)
    (tmp_path / "output" / "finetune").mkdir(parents=True, exist_ok=True)
    (tmp_path / "ckpt").mkdir()
    out = subprocess.run(
        [
            "bash",
            "-c",
            f'set -- round.env; source "{DRIVER}"; '
            'remote() { echo "$@" >> calls.txt; }; extend_seeds; cat calls.txt',
        ],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=True,
    )
    assert "remote-done" in out.stdout


def test_extra_seeds_run_after_held_out_and_before_training(tmp_path):
    (tmp_path / "round.env").write_text(
        ENV.replace("REMOTE=nowhere", "REMOTE=") + 'EXTRA_SEEDS="8016 7001"\n'
    )
    (tmp_path / "output" / "finetune").mkdir(parents=True)
    (tmp_path / "sims").mkdir()
    out = subprocess.run(
        ["bash", "-c", f'set -- round.env; source "{DRIVER}"; local_seeds'],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=True,
    )
    assert out.stdout.split()[2:6] == ["5903", "8016", "7001", "5001"]


def test_remote_runs_extra_seeds_after_held_out(tmp_path):
    env = ENV + 'EXTRA_SEEDS="8016 7001"\n'
    out = run(tmp_path, "seeds | tr '\\n' ' '", env)
    # Held-out 5902 is done; the laptop's own range 5009-5010 is skipped.
    assert out.split()[:5] == ["5901", "5903", "8016", "7001", "5001"]


def test_second_trainer_failure_alerts_the_mailbox(tmp_path):
    env = ENV + "MAILBOX=inbox.md\n"
    (tmp_path / "output" / "finetune").mkdir(parents=True)
    (tmp_path / "output" / "finetune" / "trainer.log").write_text("boom\n")
    out = run(
        tmp_path,
        "remote_state=down; failures=0; "
        "trainer_failed 1; cat inbox.md 2> /dev/null; echo ---; "
        "trainer_failed 1; cat inbox.md; cat output/finetune/STATUS",
        env,
    )
    before, after = out.split("---")
    assert before.strip() == ""
    assert "ALERT: trainer failed 2 times" in after
    assert after.count("ALERT") == 2  # one mailbox line, and STATUS


def _simulator(tmp_path: Path) -> str:
    import hashlib

    (tmp_path / "scripts").mkdir()
    (tmp_path / "scripts" / "finetune_simulate.R").write_text("new\n")
    return hashlib.sha256(b"new\n").hexdigest()


def test_stale_remote_simulator_is_replaced_before_batches(tmp_path):
    sha = _simulator(tmp_path)
    out = run(
        tmp_path,
        # The remote holds an old simulator until the copy lands.
        "copied=0; enw_fingerprint() { echo same; }; "
        "remote() { case $1 in sha256sum*) "
        f"(( copied )) && echo '{sha}  f' || echo 'old  f';; "
        "*enw_fingerprint*) echo same;; "
        '*) echo "$1" >> calls;; esac; }; '
        "rsync() { copied=1; echo copied >> calls; }; "
        "remote_code && echo current; cat calls",
    )
    # Old batches stop before the copy, and the copy is checked.
    assert out.startswith("current")
    assert out.index("pkill") < out.index("copied")


def test_remote_simulator_that_stays_stale_blocks_batches(tmp_path):
    _simulator(tmp_path)
    out = run(
        tmp_path,
        "remote() { case $1 in sha256sum*) echo 'old  f';; esac; }; "
        "rsync() { :; }; "
        "remote_scheduler() { echo started; }; "
        "remote_code || echo stale; "
        "remote() { case $1 in sha256sum*) echo 'old  f';; "
        "*df*) echo 100G;; esac; }; "
        "remote_tried=0; check_remote; cat output/finetune/driver.log",
    )
    assert "stale" in out
    assert "started" not in out
    assert "simulator" in out


def test_trainer_does_not_start_from_a_dirty_worktree(tmp_path):
    def git(*args):
        subprocess.run(["git", *args], cwd=tmp_path, check=True)

    git("init", "-q")
    (tmp_path / "code.py").write_text("x = 1\n")
    git("add", "code.py")
    git("-c", "user.name=t", "-c", "user.email=t@t", "commit", "-qm", "c")
    (tmp_path / "code.py").write_text("x = 2\n")
    out = run(
        tmp_path,
        "trainer_args=(); (start_trainer); echo exit $?; "
        "cat output/finetune/driver.log; ls output/finetune",
    )
    assert "exit 1" in out
    assert "uncommitted" in out
    assert "trainer.pid" not in out


def test_remote_with_another_epinowcast_runs_no_batches(tmp_path):
    sha = _simulator(tmp_path)
    out = run(
        tmp_path,
        "enw_fingerprint() { echo laptop; }; "
        "remote() { case $1 in sha256sum*) "
        f"echo '{sha}  f';; *enw_fingerprint*) echo archie;; esac; }}; "
        "rsync() { :; }; "
        "remote_code || echo blocked; cat output/finetune/driver.log",
    )
    assert "blocked" in out
    assert "epinowcast" in out


def test_laptop_first_runs_the_seeds_the_trainer_waits_for(tmp_path):
    (tmp_path / "sims").mkdir()
    rows = "outbreak\n" + "".join(f"x-{i}\n" for i in range(5))
    for s in (5001, 5002, 5004, 5005):
        (tmp_path / "sims" / f"settings-{s}.csv").write_text(rows)
    (tmp_path / "ckpt").mkdir()
    (tmp_path / "ckpt" / "stream-status.json").write_text(
        '{"stage": "waiting for outbreaks 11-20"}'
    )
    # 5003 is missing: with five outbreaks a seed, outbreaks 11-20 need
    # seeds 5001-5004, and 5003 is on no machine's list first.
    out = run(tmp_path, "blocking_seeds | tr '\\n' ' '")
    assert out.split() == ["5003"]


def test_restarting_the_remote_scheduler_spares_its_batches(tmp_path):
    out = run(
        tmp_path,
        'remote() { echo "$@" >> calls; }; restart_remote_scheduler; cat calls',
    )
    assert "pkill -o -f" in out
    assert "finetune_simulate" not in out


def test_status_shows_the_last_chunk_time_split(tmp_path):
    (tmp_path / "ckpt").mkdir()
    (tmp_path / "ckpt" / "stream-status.json").write_text(
        '{"stage": "training", "split": '
        '{"wait_s": 5, "cut_s": 0, "train_s": 3500, "eval_s": 60}}'
    )
    out = run(tmp_path, "remote_state=up; status; cat output/finetune/STATUS")
    assert "last_chunk=wait 5s cut 0s train 3500s eval 60s" in out


def _scheduler(tmp_path: Path) -> str:
    """Return the remote scheduler script the driver writes."""
    run(
        tmp_path,
        "remote() { case $1 in cat*) cat > sched.sh;; *) return 1;; esac; }; "
        "remote_scheduler; true",
        ENV + "REMOTE_RETRY_MEM_KB=2\nREMOTE_MIN_FREE_MEM_GB=1\n"
        "REMOTE_MAX_LOAD=9\n",
    )
    return (tmp_path / "sched.sh").read_text()


def test_remote_retries_an_aborted_seed_with_a_higher_cap(tmp_path):
    script = _scheduler(tmp_path)
    # A batch that R aborted (a segfault under the memory cap) runs again
    # once the first pass is done, under REMOTE_RETRY_MEM_KB.
    assert "R is aborting" in script
    assert "batch $s 1 &" in script
    assert "batch $s 2\n" in script


def test_batches_log_their_peak_memory(tmp_path):
    script = _scheduler(tmp_path)
    assert "/usr/bin/time" in script
    assert "peak_rss_kb" in script


def test_laptop_leaves_unclaimed_seeds_to_an_answering_remote(tmp_path):
    out = run(
        tmp_path,
        "remote_state=up; candidate_seeds | tr '\\n' ' '; echo; "
        "remote_state=down; candidate_seeds | tr '\\n' ' '",
    )
    up, down = out.split("\n")[:2]
    # With the remote up the laptop runs its own range (5009-5010) only;
    # with it down, every unstarted seed too.
    assert up.split() == ["5009", "5010"]
    assert "5001" in down.split()


def test_a_claimed_seed_is_stopped_on_the_remote(tmp_path):
    out = run(
        tmp_path,
        'remote() { echo "$@" >> calls; }; claim 5004; cat calls; '
        "cat sims/local-seeds.txt",
    )
    assert "inetune_simulate.R --args 25 5004 " in out
    assert out.split()[-1] == "5004"


def test_long_wait_is_an_alert_in_status_and_the_mailbox(tmp_path):
    import time

    (tmp_path / "ckpt").mkdir()
    since = time.time() - 31 * 60
    (tmp_path / "ckpt" / "stream-status.json").write_text(
        f'{{"stage": "waiting for outbreaks 1-500", "waiting_since": {since}}}'
    )
    out = run(
        tmp_path,
        "remote_state=up; status; status; cat output/finetune/STATUS; "
        "cat inbox.md",
        ENV + "MAILBOX=inbox.md\n",
    )
    assert "ALERT: trainer waiting 31 min" in out
    # One mailbox line per wait, however often STATUS is written.
    assert out.count("fm-finetune3") == 1


def test_a_second_driver_does_not_start(tmp_path):
    (tmp_path / "output" / "finetune").mkdir(parents=True)
    fake = ["sh", "-c", "sleep 30; true", "finetune_driver.sh"]
    live = subprocess.Popen(fake)
    try:
        (tmp_path / "output" / "finetune" / "driver.pid").write_text(
            f"{live.pid}\n"
        )
        out = run(tmp_path, "another_driver && echo running || echo none")
    finally:
        live.kill()
    assert out.split() == ["running"]
    out = run(tmp_path, "another_driver && echo running || echo none")
    assert out.split() == ["none"]


def test_status_after_a_restart_keeps_step_and_last_checkpoint(tmp_path):
    ckpt = tmp_path / "ckpt"
    (ckpt / "state").mkdir(parents=True)
    (ckpt / "state" / "state.json").write_text('{"step": 5577, "done": 2000}')
    for n, key in ((1000, "aaa"), (2000, "bbb")):
        (ckpt / f"n{n}").mkdir()
        (ckpt / f"n{n}" / "settings.json").write_text(f'{{"hash": "{key}"}}')
    # A restarted trainer's status starts again from nothing.
    (ckpt / "stream-status.json").write_text(
        '{"stage": "waiting for held-out simulations", "step": 0}'
    )
    out = run(tmp_path, "remote_state=up; status; cat output/finetune/STATUS")
    assert "train_step=5577" in out
    assert "last_checkpoint=n2000 bbb" in out


def test_simulation_pauses_far_ahead_of_the_trainer(tmp_path):
    (tmp_path / "sims").mkdir()
    rows = "outbreak\n" + "".join(f"x-{i}\n" for i in range(5))
    for s in range(5001, 5009):
        if s != 5003:
            (tmp_path / "sims" / f"settings-{s}.csv").write_text(rows)
    (tmp_path / "ckpt" / "state").mkdir(parents=True)
    state = tmp_path / "ckpt" / "state" / "state.json"
    status = tmp_path / "ckpt" / "stream-status.json"
    status.write_text('{"stage": "training"}')
    env = ENV + "LEAD_MAX=20\nLEAD_RESUME=10\n"
    script = (
        "throttle; echo $throttled; candidate_seeds | tr '\\n' ' '; echo; "
        "echo '{\"done\": 15}' > ckpt/state/state.json; "
        "throttle; echo $throttled; "
        "echo '{\"done\": 26}' > ckpt/state/state.json; "
        "throttle; echo $throttled"
    )
    state.write_text('{"done": 0}')
    lines = run(tmp_path, "remote_state=up; " + script, env).splitlines()
    # 35 outbreaks done ahead of 0 trained: paused, and the laptop runs
    # only seeds the trainer waits for (none while it trains).
    assert lines[0] == "1"
    assert lines[1].split() == []
    # 20 ahead is between the bounds, so the pause holds; 9 ahead ends it.
    assert lines[2:] == ["1", "0"]


def test_a_checkpoint_that_fails_its_check_is_an_alert(tmp_path):
    for n in (1000, 2000):
        (tmp_path / "ckpt" / f"n{n}").mkdir(parents=True)
        (tmp_path / "ckpt" / f"n{n}" / "settings.json").write_text("{}")
    out = run(
        tmp_path,
        'verify_checkpoint() { [[ "$1" == *n1000 ]]; }; '
        "check_checkpoints; check_checkpoints; ls ckpt/n1000 ckpt/n2000; "
        "cat inbox.md; remote_state=up; status; cat output/finetune/STATUS",
        ENV + "MAILBOX=inbox.md\n",
    )
    assert "verified" in out.split("ckpt/n2000")[0]
    assert "ALERT: checkpoint n2000 does not match its code_commit" in out
    # One mailbox line however often it is checked.
    assert out.count("fm-finetune3") == 1


def test_laptop_runs_next_chunk_seeds_that_aborted_before_they_are_due(
    tmp_path,
):
    sims = tmp_path / "sims"
    (sims / "logs").mkdir(parents=True)
    rows = "outbreak\n" + "".join(f"x-{i}\n" for i in range(5))
    for s in (5001, 5002, 5005):
        (sims / f"settings-{s}.csv").write_text(rows)
    # 5003 aborted on the remote; 5004 is running there.
    (sims / "logs" / "5003.log").write_text("R is aborting now ...\n")
    (sims / "logs" / "5004.log").write_text("running\n")
    (tmp_path / "ckpt" / "state").mkdir(parents=True)
    (tmp_path / "ckpt" / "state" / "state.json").write_text('{"done": 10}')
    (tmp_path / "ckpt" / "stream-status.json").write_text(
        '{"stage": "training"}'
    )
    # Training outbreaks 1-10 with 10 more ahead needs seeds 5001-5004.
    out = run(
        tmp_path,
        "blocking_seeds | tr '\\n' ' '",
        ENV + "LOOKAHEAD_OUTBREAKS=10\n",
    )
    assert out.split() == ["5003"]


def test_r4_training_settings_reach_the_trainer(tmp_path):
    env = ENV + "LR_DECAY=1\nSCALE_FLOOR=1\nREAL_VALIDATION=real.pkl\n"
    out = run(tmp_path, 'build_trainer_args; echo "${trainer_args[@]}"', env)
    assert "--lr-decay 1" in out
    assert "--scale-floor 1" in out
    assert "--real-validation real.pkl" in out
    out = run(tmp_path, 'build_trainer_args; echo "${trainer_args[@]}"')
    assert "--lr-decay" not in out
    assert "--scale-floor" not in out
