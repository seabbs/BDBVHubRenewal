"""The dashboard draws nowcasts with the paper figures' styling."""

from pathlib import Path

import pytest

pytest.importorskip("matplotlib")

from flowlib.dashboard import paperstyle

PLOTS = Path(__file__).parents[2] / "paper" / "paperfigs" / "plots.py"


def test_levels_and_ink_come_from_paperfigs():
    from paperfigs import plots

    assert paperstyle.LEVELS == plots.NOWCAST_LEVELS
    assert paperstyle.INK["truth"] == plots._PANEL_INK["truth"]
    assert paperstyle.INK["data"] == plots._PANEL_INK["data"]


def test_bands_match_the_paper_nowcast_panels():
    # The bands are inline in paperfigs.plots.nowcast_panels; this fails if
    # they change there, so the dashboard follows.
    literal = "bands = (((0.025, 0.975), 0.15), ((0.25, 0.75), 0.3))"
    assert literal in PLOTS.read_text()
    assert paperstyle.BANDS == ((0.025, 0.975, 0.15), (0.25, 0.75, 0.3))


def test_reference_models_take_their_paper_colours():
    from paperfigs import style

    assert (
        paperstyle.model_colour("randomwalknowcast")
        == style.model_style("randomwalknowcast", ["randomwalknowcast"])[0]
    )


def test_the_page_style_has_what_the_page_draws():
    got = paperstyle.page_style()
    assert set(got) == {"levels", "bands", "ink"}
    assert got["levels"] == list(paperstyle.LEVELS)
