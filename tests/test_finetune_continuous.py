"""Continuous fine-tuning on the laptop from archie's rounds."""

import subprocess
from pathlib import Path

SCRIPT = Path(__file__).parents[1] / "scripts" / "finetune_continuous.sh"


HEADER = "round,seed,kind,grid\n"


def _round(remote, n, rows):
    """A remote round with these finished manifest rows (seed, kind)."""
    r = remote / f"r{n}"
    (r / "sims").mkdir(parents=True, exist_ok=True)
    (r / "manifest.csv").write_text(
        HEADER + "".join(f"{n},{s},{k},r4\n" for s, k in rows)
    )
    for s, _ in rows:
        (r / "sims" / f"settings-{s}.csv").write_text("x")
    return r


def _sync(tmp_path, remote, first=1):
    out = subprocess.run(
        [
            "bash",
            "-c",
            f'source "{SCRIPT}"; HERE={tmp_path}/local; FIRST_ROUND={first}; '
            f'remote_ls() {{ ls "{remote}"; }}; '
            # GNU and BSD cp agree on a folder's contents only as `src/.`.
            f'pull() {{ mkdir -p "$3"; s="{remote}/$1/$2"; '
            'if [ -d "$s" ]; then cp -R "$s/." "$3"; '
            'else cp "$s" "$3"; fi; }; '
            "sync_rounds; cat $HERE/seeds.txt",
        ],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=True,
    )
    return out.stdout.split()


def _seeds(n):
    """Round n's 40 batch seeds, in seed order."""
    return [str(30000 + 100 * n + i) for i in range(40)]


def test_the_seed_list_is_the_ten_training_rounds_in_seed_order(tmp_path):
    remote = tmp_path / "remote"
    # Batches finish out of order; the list is in seed order regardless,
    # every seed of the ten rounds after the held-out round.
    _round(remote, 1, [(30102, "block"), (30100, "block")])
    _round(remote, 2, [(30201, "block")])
    want = [s for n in range(2, 12) for s in _seeds(n)]
    assert _sync(tmp_path, remote) == want
    _round(remote, 2, [(30201, "block"), (30200, "block")])
    assert _sync(tmp_path, remote) == want
    # The held-out round's batches are pulled too.
    assert (tmp_path / "local" / "sims" / "settings-30200.csv").exists()
    assert (tmp_path / "local" / "sims" / "settings-30102.csv").exists()


def test_the_held_out_seeds_are_the_whole_held_out_round(tmp_path):
    out = _source(tmp_path, "", "held_out_seeds")
    assert out.returncode == 0
    assert out.stdout.split() == ["30900", "30939"]


def test_n1000_must_beat_zero_shot_on_both(tmp_path):
    import json

    cases = {
        "better": (0.20, 0.29, 3.60),
        "worse_sims": (0.30, 0.29, 3.60),
        "worse_real": (0.20, 0.29, 4.50),
    }
    results = {}
    for name, (loss, start, real) in cases.items():
        path = tmp_path / f"{name}.json"
        path.write_text(
            json.dumps(
                {
                    "held_out_loss": loss,
                    "held_out_loss_start": start,
                    "real_validation_loss": real,
                }
            )
        )
        out = subprocess.run(
            ["bash", "-c", f'source "{SCRIPT}"; beats_zero_shot {path}'],
            cwd=tmp_path,
            capture_output=True,
        )
        results[name] = out.returncode == 0
    assert results == {"better": True, "worse_sims": False, "worse_real": False}


def _source(tmp_path, env: str, command: str):
    return subprocess.run(
        ["bash", "-c", f'{env} source "{SCRIPT}"; {command}'],
        cwd=tmp_path,
        capture_output=True,
        text=True,
    )


def test_the_run_trains_the_attention_biases_by_default(tmp_path):
    on = _source(
        tmp_path, "", 'bias_args; echo; remote_checkpoints; echo "$HERE"'
    )
    assert on.returncode == 0
    assert on.stdout.split() == [
        "--series-bias",
        "checkpoints/slots",
        "output/finetune/continuous-r5",
    ]


def test_a_no_bias_control_sets_it_and_its_own_folder(tmp_path):
    off = _source(
        tmp_path,
        f"export SERIES_BIAS=0 HERE={tmp_path}/nobias;",
        "bias_args; echo; remote_checkpoints",
    )
    assert off.returncode == 0
    assert off.stdout.split() == ["checkpoints"]


def test_a_no_bias_run_refuses_the_design_runs_folder(tmp_path):
    out = _source(tmp_path, "export SERIES_BIAS=0;", "bias_args")
    assert out.returncode != 0
    assert "HERE" in out.stderr


def test_the_restart_trains_on_grid_r5_with_one_draw_of_32_blocks(tmp_path):
    out = _source(tmp_path, "", 'echo "$DRAWS $CUTS $CHUNK"')
    assert out.returncode == 0
    assert out.stdout.split() == ["1", "32", "250"]
    script = SCRIPT.read_text()
    assert "--grid r5" in script
    assert "--blocks-per-simulation $((CUTS * DRAWS))" in script


def test_the_restart_streams_only_rounds_from_the_first_r5_round(tmp_path):
    remote = tmp_path / "remote"
    _round(remote, 8, [(30801, "block")])
    _round(remote, 9, [(30900, "block")])
    assert _sync(tmp_path, remote, first=9) == [
        s for n in range(10, 20) for s in _seeds(n)
    ]
    assert not (tmp_path / "local" / "sims" / "settings-30801.csv").exists()
    out = _source(tmp_path, "", 'echo "$FIRST_ROUND"')
    assert out.stdout.split() == ["9"]


def test_the_held_out_default_leaves_room_for_skipped_simulations(tmp_path):
    # The trainer monitors the first 90 simulations of the held-out round,
    # a random sample of its block's cells; the whole round is scored at
    # each checkpoint by the scoring workflow.
    out = _source(tmp_path, "", 'echo "$HOLDOUT"')
    assert out.stdout.split() == ["90"]


def test_the_run_is_ten_grid_blocks(tmp_path):
    out = _source(tmp_path, "", 'echo "$OUTBREAKS $PREFIXES"')
    assert out.stdout.split() == [
        "8400",
        *["250", "500", "750", "1000", "2000", "4000", "8400"],
    ]
    # The rate falls to zero over OUTBREAKS, not a step count.
    assert "LR_STEPS" not in SCRIPT.read_text()
