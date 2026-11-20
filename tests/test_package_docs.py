"""Each package's docs site follows the template and its examples run."""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
SITES = sorted((ROOT / "packages").glob("*/mkdocs.yml"))
BLOCK = re.compile(r"^```python\n(.*?)^```", re.MULTILINE | re.DOTALL)
TEMPLATE = [
    {"Getting started": "index.md"},
    {"How to": "how-to.md"},
    {"Reference": "reference.md"},
    {"How it is used in the workflow": "workflow.md"},
]


def _config(site: Path) -> dict:
    return yaml.safe_load(site.read_text())


def test_each_documented_package_has_its_own_site() -> None:
    names = {s.parent.name for s in SITES}
    assert {"nowcastutils", "nowcasteval", "nowcastbaselines"} <= names


@pytest.mark.parametrize("site", SITES, ids=lambda s: s.parent.name)
def test_a_package_site_follows_the_template(site: Path) -> None:
    config = _config(site)
    assert config["site_name"] == site.parent.name
    assert config["nav"] == TEMPLATE
    for page in TEMPLATE:
        assert (site.parent / "docs" / next(iter(page.values()))).is_file()


@pytest.mark.parametrize("site", SITES, ids=lambda s: s.parent.name)
def test_the_main_site_links_out_to_each_package_site(site: Path) -> None:
    index = (ROOT / "docs" / "packages" / "index.md").read_text()
    assert f"](../{site.parent.name}/)" in index


@pytest.mark.parametrize("site", SITES, ids=lambda s: s.parent.name)
def test_package_site_examples_run(site: Path, tmp_path: Path) -> None:
    # Every page's blocks run in the nav's order and share their names, as
    # a reader runs them, in a new interpreter so that no other package's
    # imports stand in for the package's own.
    pages = [next(iter(p.values())) for p in TEMPLATE]
    code = "\n".join(
        block
        for page in pages
        for block in BLOCK.findall((site.parent / "docs" / page).read_text())
    )
    assert code, "a package site has no example"
    run = subprocess.run(
        [sys.executable, "-c", code],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
    )
    assert run.returncode == 0, run.stderr
