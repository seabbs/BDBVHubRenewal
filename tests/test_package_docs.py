"""The package pages' examples run as written (WP5)."""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
PAGES = sorted((ROOT / "docs" / "packages").glob("*.md"))
BLOCK = re.compile(r"^```python\n(.*?)^```", re.MULTILINE | re.DOTALL)


def test_every_package_page_has_an_example() -> None:
    examples = {p.stem for p in PAGES if BLOCK.search(p.read_text())}
    assert {"nowcastutils", "nowcasteval", "nowcastbaselines"} <= examples


@pytest.mark.parametrize(
    "page",
    [p for p in PAGES if BLOCK.search(p.read_text())],
    ids=lambda p: p.stem,
)
def test_package_page_examples_run(page: Path, tmp_path: Path) -> None:
    # The blocks of a page run in order and share their names, as a reader
    # runs them, in a new interpreter so that no other page's imports
    # stand in for the page's own.
    code = "\n".join(BLOCK.findall(page.read_text()))
    run = subprocess.run(
        [sys.executable, "-c", code],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
    )
    assert run.returncode == 0, run.stderr
