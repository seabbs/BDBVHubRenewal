"""The rounds driver's batch runner (scripts/sim_rounds.sh)."""

import os
import subprocess
from pathlib import Path

ROOT = Path(__file__).parents[1]
SCRIPT = ROOT / "scripts" / "sim_rounds.sh"

# Stand-ins: systemd-run drops its own options and runs the command;
# Rscript writes a file in TMPDIR, then succeeds or fails as asked.
FAKES = {
    "systemd-run": """#!/bin/bash
while [ "$1" = --user ] || [ "$1" = --scope ] || [ "$1" = -q ]; do shift; done
while [ "$1" = -p ]; do shift 2; done
exec "$@"
""",
    "Rscript": """#!/bin/bash
seed=$3 out=$4
echo standata > "$TMPDIR/standata.json"
echo "$TMPDIR" > "$out/tmpdir-$seed"
[ "$FAKE_FAIL" = 1 ] && exit 1
echo outbreak > "$out/settings-$seed.csv"
echo x | gzip > "$out/counts-$seed.csv.gz"
""",
}


def _run(tmp_path: Path, body: str, **env: str) -> subprocess.CompletedProcess:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir(exist_ok=True)
    for name, text in FAKES.items():
        (bin_dir / name).write_text(text)
        (bin_dir / name).chmod(0o755)
    (tmp_path / "r1" / "sims" / "logs").mkdir(parents=True, exist_ok=True)
    return subprocess.run(
        ["bash", "-c", f'source "{SCRIPT}"; {body}'],
        cwd=ROOT,
        env={
            **os.environ,
            "PATH": f"{bin_dir}:{os.environ['PATH']}",
            "MAILBOX": str(tmp_path / "mailbox.md"),
            **env,
        },
        capture_output=True,
        text=True,
        timeout=60,
    )


def test_a_batch_runs_in_its_own_tmpdir_removed_on_success(tmp_path):
    out = _run(tmp_path, f'run_batch 7 "{tmp_path}/r1" 1G; echo rc=$?')
    assert "rc=0" in out.stdout
    used = Path((tmp_path / "r1" / "sims" / "tmpdir-7").read_text().strip())
    assert used.parent.parent == tmp_path / "r1" / "sims"
    assert not used.exists()


def test_a_failed_batch_also_removes_its_tmpdir(tmp_path):
    out = _run(
        tmp_path, f'run_batch 8 "{tmp_path}/r1" 1G; echo rc=$?', FAKE_FAIL="1"
    )
    assert "rc=1" in out.stdout
    used = Path((tmp_path / "r1" / "sims" / "tmpdir-8").read_text().strip())
    assert not used.exists()


def test_room_needs_the_disk_threshold(tmp_path):
    ok = _run(
        tmp_path,
        "room && echo yes || echo no",
        MIN_DISK_GB="0",
        MIN_FREE_GB="0",
        ROOT=str(tmp_path),
    )
    full = _run(
        tmp_path,
        "room && echo yes || echo no",
        MIN_DISK_GB="100000000",
        MIN_FREE_GB="0",
        ROOT=str(tmp_path),
    )
    assert ok.stdout.strip() == "yes"
    assert full.stdout.strip() == "no"


def test_draws_per_simulation_are_set_by_the_launcher():
    script = (
        Path(__file__).parents[1] / "scripts" / "sim_rounds.sh"
    ).read_text()
    assert "DRAWS=${DRAWS:-1}\n" in script


def test_every_simulation_has_one_observation_draw():
    scripts = Path(__file__).parents[1] / "scripts"
    laptop = (scripts / "sim_rounds_laptop.sh").read_text()
    assert "\nDRAWS=1\n" in laptop
    simulate = (scripts / "finetune_simulate.R").read_text()
    assert 'Sys.getenv("ENW_DRAWS", "1")' in simulate


def test_a_round_is_one_grid_block_of_40_batches_of_21():
    scripts = Path(__file__).parents[1] / "scripts"
    rounds = (scripts / "sim_rounds.sh").read_text()
    assert "BATCH=${BATCH:-21}\n" in rounds
    assert "ROUND_BATCHES=${ROUND_BATCHES:-40}\n" in rounds
    assert "HELD_BATCHES" not in rounds
    assert "ENW_SEED_BASE=$SEED_BASE" in rounds
    laptop = (scripts / "sim_rounds_laptop.sh").read_text()
    assert "BATCH=${BATCH:-21}\n" in laptop
    assert "held" not in laptop
