"""Continuous fine-tuning on the laptop from archie's rounds."""

import subprocess
from pathlib import Path

SCRIPT = Path(__file__).parents[1] / "scripts" / "finetune_continuous.sh"


def test_each_new_round_is_synced_once_in_order(tmp_path):
    remote = tmp_path / "remote" / "rounds"
    for n, seeds in ((1, "20001\n20002\n"), (2, "21001\n"), (3, "22001\n")):
        (remote / f"r{n}" / "sims").mkdir(parents=True)
        (remote / f"r{n}" / "seeds.txt").write_text(seeds)
        (remote / f"r{n}" / "held-out.txt").write_text("29901 29902\n")
        (remote / f"r{n}" / "sims" / f"settings-{n}.csv").write_text("x")
        # Round 3 is still being reviewed: not READY, so not synced.
        if n < 3:
            (remote / f"r{n}" / "READY").write_text("t")
    (remote / "r1" / "qc").mkdir()
    (remote / "r1" / "qc" / "excluded.csv").write_text(
        "simulation,reason,evidence\n20001-0003,r,e\n"
    )
    out = subprocess.run(
        [
            "bash",
            "-c",
            f'source "{SCRIPT}"; HERE={tmp_path}/local; '
            # The remote is a local folder here.
            f'remote_ls() {{ ls "{remote}"; }}; '
            f'pull() {{ cp -R "{remote}/$1" "$2"; }}; '
            "sync_rounds; sync_rounds; cat $HERE/seeds.txt; ls $HERE/sims; "
            "echo excluded; cat $HERE/excluded.txt",
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
    assert "22001" not in lines
    assert "settings-3.csv" not in lines
    assert lines[lines.index("excluded") + 1 :] == ["20001-0003"]
