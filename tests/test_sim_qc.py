"""The recorded sim-QC review of a round (scripts/sim_qc.py)."""

import json
import os
import subprocess
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(Path(__file__).parent))
from test_finetune_chronos import _sim_dir  # noqa: E402


def test_qc_records_the_review_and_keeps_only_known_exclusions(tmp_path):
    sims = tmp_path / "r1" / "sims"
    sims.mkdir(parents=True)
    _sim_dir(sims)
    reply = {
        "excluded": [
            {"simulation": "6001-0001", "reason": "r", "evidence": "e"},
            {"simulation": "9999-0001", "reason": "r", "evidence": "e"},
        ],
        "notes": "fine",
    }
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    fake = bin_dir / "claude"
    fake.write_text(
        "#!/bin/sh\ncat > /dev/null\n"
        f"echo '{json.dumps({'result': json.dumps(reply)})}'\n"
    )
    fake.chmod(0o755)
    env = {**os.environ, "PATH": f"{bin_dir}:{os.environ['PATH']}"}
    subprocess.run(
        [sys.executable, "scripts/sim_qc.py", str(tmp_path / "r1")],
        cwd=ROOT,
        env=env,
        check=True,
    )
    qc = tmp_path / "r1" / "qc"
    excluded = pd.read_csv(qc / "excluded.csv")
    assert excluded["simulation"].tolist() == ["6001-0001"]
    decisions = json.loads((qc / "decisions.json").read_text())
    assert decisions["dropped_unknown_ids"] == ["9999-0001"]
    assert decisions["n_simulations"] == 4
    assert len(decisions["prompt_sha256"]) == 64
    assert (qc / "prompt-a.md").read_text().startswith("# Sim QC prompt A")
    assert len(pd.read_csv(qc / "input.csv")) == 4
    assert "excluded 1 of 4" in (qc / "summary.txt").read_text()
