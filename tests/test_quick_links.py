"""The Quick links on the home page and in the README, grouped by project."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SITE = "https://samabbott.co.uk/bdbv-hub-modelling/"
SITES = {s.parent.name for s in (ROOT / "packages").glob("*/mkdocs.yml")}
GROUPS = [
    ["Fine-tuned foundation", "nowcastfoundation"],
    ["Joint-model-informed renewal", "nowcastrenewal"],
    ["Hub submission", "bdbvhub"],
    ["nowcastutils", "nowcasteval", "nowcastbaselines", "nowcastworkflow"],
]
PACKAGES = [name for group in GROUPS for name in group if name.islower()]
HOME_LINK = re.compile(r"\} ([\w\- ]+)\]\(([^)\s]+)\)\{ \.quick-link \}")
README_LINK = re.compile(r'<a href="([^"]+)">.*?<b>([^<]+)</b></a>')
WIDTH = re.compile(r'width="(\d+)"')


def _home() -> str:
    text = (ROOT / "docs" / "index.md").read_text()
    start = text.index('<div class="quick-links" markdown>')
    return text[start : text.index("\n</div>", start)]


def _readme() -> str:
    text = (ROOT / "README.md").read_text()
    start = text.index("## Quick links")
    return text[start : text.index("</table>", start)]


def _home_groups() -> list[list[tuple[str, str]]]:
    rows = [r for r in _home().split("\n\n") if HOME_LINK.search(r)]
    return [HOME_LINK.findall(r) for r in rows]


def _readme_groups() -> list[list[tuple[str, str]]]:
    rows = _readme().split("<tr>")[1:]
    return [[(n, u) for u, n in README_LINK.findall(r)] for r in rows]


def _target(name: str, home: bool) -> str:
    if name in SITES:
        return f"{name}/" if home else f"{SITE}{name}/"
    anchor = f"packages/index.md#{name}" if home else f"packages/#{name}"
    return anchor if home else f"{SITE}{anchor}"


@pytest.mark.parametrize("where", ["home", "readme"])
def test_quick_links_are_grouped_by_project(where: str) -> None:
    groups = _home_groups() if where == "home" else _readme_groups()
    assert [[n for n, _ in g] for g in groups] == GROUPS


@pytest.mark.parametrize("where", ["home", "readme"])
def test_a_package_links_to_its_own_site_where_it_has_one(where: str) -> None:
    groups = _home_groups() if where == "home" else _readme_groups()
    links = dict(link for group in groups for link in group)
    for name in PACKAGES:
        assert links[name] == _target(name, where == "home"), name


@pytest.mark.parametrize("where", ["home", "readme"])
def test_quick_links_use_compact_marks(where: str) -> None:
    text = _home() if where == "home" else _readme()
    widths = {int(w) for w in WIDTH.findall(text)}
    assert widths
    assert max(widths) <= 48
