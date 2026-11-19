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


def test_charts_redraw_to_the_full_screen_size_and_keep_their_cards():
    # Sam 11:10: entering and leaving full screen redraws each chart at
    # the new size, keeping zoom and selections.
    expand = (PAGE / "expand.js").read_text()
    assert "ResizeObserver" in expand
    assert "fullscreenchange" in expand
    assert "window.ftFull=" in expand
    for name, card in (("loss.js", "cards("), ("bias.js", "biaslinecard")):
        js = (PAGE / name).read_text()
        assert 'addEventListener("resize"' in js, name
        assert "ftFull" in js, name
        # Cards are filled in place, not rebuilt, so the open one stays.
        assert card in js, name
    nowcasts = (PAGE / "nowcasts.js").read_text()
    assert "ftFull" in nowcasts


def test_the_nowcasts_have_a_view_of_one_model_over_time():
    # Sam 11:00: several origins of one model on one plot, each with its
    # horizons, against the final truth; how many origins and which model.
    html = (PAGE / "nowcasts.html").read_text()
    for control in ("id=gmode", "id=gmod", "id=gnd", "id=gstep", "id=gsl"):
        assert control in html, control
    js = (PAGE / "nowcasts.js").read_text()
    assert "S.mode==1" in js
    # Over time, the latest release is the truth drawn.
    assert "pts=lat" in js
    # The score card still compares every ticked model.
    assert "wisChart(all[0],ticked())" in js
