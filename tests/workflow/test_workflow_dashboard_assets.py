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


def test_the_nowcasts_view_and_its_controls_expand_as_one():
    html = (PAGE / "nowcasts.html").read_text()
    assert html.lstrip().startswith("<div class=expandable id=gview")
    assert "id=gctl" in html or "class=gctl" in html
    assert html.rstrip().endswith("</div>")


def test_the_expand_script_is_keyboard_accessible():
    js = (PAGE / "expand.js").read_text()
    # A real button (Tab, Enter, Space), a label, Esc to close, the
    # Fullscreen API with the overlay as fallback, and a redraw.
    for needle in (
        'createElement("button")',
        "aria-label",
        '"Escape"',
        "requestFullscreen",
        'new Event("resize")',
        ".focus()",
    ):
        assert needle in js
