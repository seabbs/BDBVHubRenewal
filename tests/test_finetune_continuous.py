"""Continuous fine-tuning on the laptop from archie's rounds."""

import subprocess
from pathlib import Path

SCRIPT = Path(__file__).parents[1] / "scripts" / "finetune_continuous.sh"


def test_each_new_round_is_synced_once_in_order(tmp_path):
    remote = tmp_path / "remote" / "rounds"
    for n, seeds in ((1, "20001\n20002\n"), (2, "21001\n")):
        (remote / f"r{n}" / "sims").mkdir(parents=True)
        (remote / f"r{n}" / "seeds.txt").write_text(seeds)
        (remote / f"r{n}" / "held-out.txt").write_text("29901 29902\n")
        (remote / f"r{n}" / "sims" / f"settings-{n}.csv").write_text("x")
    out = subprocess.run(
        [
            "bash",
            "-c",
            f'source "{SCRIPT}"; HERE={tmp_path}/local; '
            # The remote is a local folder here.
            f'remote_ls() {{ ls "{remote}"; }}; '
            f'pull() {{ cp -R "{remote}/$1" "$2"; }}; '
            "sync_rounds; sync_rounds; cat $HERE/seeds.txt; ls $HERE/sims",
        ],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=True,
    )
    lines = out.stdout.split()
    assert lines[:3] == ["20001", "20002", "21001"]
    assert "settings-1.csv" in lines
    assert "settings-2.csv" in lines


def test_n1000_must_beat_zero_shot_on_both(tmp_path):
    import json

    cases = {
        "better": (0.20, 0.29, 0.50),
        "worse_sims": (0.30, 0.29, 0.50),
        "worse_real": (0.20, 0.29, 0.70),
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
