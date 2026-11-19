"""The package pages' examples run as written (WP5)."""

from __future__ import annotations

import re
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
def test_package_page_examples_run(page: Path) -> None:
    # The blocks of a page run in order and share their names, as a reader
    # runs them.
    namespace: dict[str, object] = {}
    for code in BLOCK.findall(page.read_text()):
        exec(compile(code, str(page), "exec"), namespace)
