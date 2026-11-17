"""The page's scripts and style name no study, place, run or host."""

import re
from pathlib import Path

import pytest

PAGE = Path(__file__).parents[2] / "workflow" / "flowlib" / "dashboard" / "page"
NAMES = re.compile(
    r"""["'](?:germany|ebola|bdbv|DE|CD|archie|rw|zero|complete|zero-shot|"""
    r"""complete-only|randomwalknowcast|slots|no slots|7day|daily|weekly|"""
    r"""national)["']""",
    re.I,
)


@pytest.mark.parametrize(
    "path", sorted(PAGE.glob("*.js")), ids=lambda p: p.name
)
def test_scripts_take_names_from_the_page_data(path):
    assert NAMES.findall(path.read_text()) == []


def test_the_old_single_dataset_scripts_are_gone():
    assert not (PAGE / "germany.js").exists()
    assert not (PAGE / "checkpoints.js").exists()
