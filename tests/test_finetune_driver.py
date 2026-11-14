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
    (tmp_path / "output" / "finetune").mkdir(parents=True)
    (tmp_path / "sims").mkdir()
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
    (tmp_path / "output" / "finetune").mkdir(parents=True)
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
    (tmp_path / "output" / "finetune").mkdir(parents=True)
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
