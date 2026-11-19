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


HEX = re.compile(r"(?<![\w&])#(?:[0-9a-fA-F]{6}|[0-9a-fA-F]{3})\b")
NEUTRALS = {
    "#fff",
    "#1d2127",
    "#6b7280",
    "#f6f7f9",
    "#d9dde3",
    "#14171c",
    "#e6e8eb",
    "#9aa3af",
    "#1d2128",
    "#343a44",
}
"""The page's neutral surfaces, text and lines (style.css)."""


def test_no_dashboard_file_holds_a_colour_except_neutrals_in_the_style():
    # Every data and accent colour comes from the shared map
    # (bdbvhubrenewal.evaluation.labels): models, relations, layouts,
    # levels, runs, WIS parts and the page's accents.
    found = {}
    for path in sorted(PAGE.parent.rglob("*")):
        if path.suffix not in {".py", ".js", ".css", ".html", ".json"}:
            continue
        hexes = set(HEX.findall(path.read_text()))
        if path.name == "style.css":
            hexes -= NEUTRALS
        if hexes:
            found[str(path.relative_to(PAGE.parent))] = sorted(hexes)
    assert found == {}


def test_the_page_writes_its_accents_from_the_shared_map():
    from flowlib.dashboard import page

    from bdbvhubrenewal.evaluation import labels

    css = page.ui_colours()
    for name, (light, dark) in labels.UI_COLOURS.items():
        assert f"--{name}:{light}" in css
        assert f"--{name}:{dark}" in css
