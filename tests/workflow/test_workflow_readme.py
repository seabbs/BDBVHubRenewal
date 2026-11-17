"""workflow/README.md names every rule, so the walk-through stays whole."""

import re
import subprocess
import sys
from pathlib import Path

import pytest

pytest.importorskip("snakemake")

ROOT = Path(__file__).parents[2]


def test_every_rule_is_in_the_readme(tmp_path):
    out = subprocess.run(
        [str(Path(sys.executable).parent / "snakemake"), "--list-rules"],
        cwd=ROOT,
        env={
            "PATH": "/usr/bin:/bin",
            "HOME": str(tmp_path),
            "LOCAL_CONFIG": str(tmp_path / "none.env"),
        },
        capture_output=True,
        text=True,
        check=True,
    )
    rules = set(out.stdout.split())
    readme = (ROOT / "workflow" / "README.md").read_text()
    named = set(re.findall(r"`([a-z0-9_]+)`", readme))
    assert rules - named == set()
