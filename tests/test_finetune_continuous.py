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


def _sync(tmp_path, remote):
    out = subprocess.run(
        [
            "bash",
            "-c",
            f'source "{SCRIPT}"; HERE={tmp_path}/local; '
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


def test_finished_batches_stream_in_before_their_round_is_ready(tmp_path):
    remote = tmp_path / "remote"
    _round(remote, 1, [(20001, "train"), (20002, "train"), (29901, "held_out")])
    (remote / "r1" / "held-out.txt").write_text("29901 29901\n")
    _round(remote, 2, [(30201, "train")])
    assert _sync(tmp_path, remote) == ["20001", "20002", "30201"]
    # A batch finishing later joins once, after the others.
    _round(remote, 2, [(30201, "train"), (30200, "train")])
    assert _sync(tmp_path, remote) == ["20001", "20002", "30201", "30200"]
    assert (tmp_path / "local" / "sims" / "settings-30200.csv").exists()
    assert (tmp_path / "local" / "sims" / "settings-29901.csv").exists()


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
