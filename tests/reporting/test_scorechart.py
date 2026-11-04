from __future__ import annotations

import pandas as pd

from bdbvhubrenewal.reporting.plots import model_colour
from bdbvhubrenewal.reporting.scorechart import (
    score_chart_data,
    score_chart_html,
)

MODELS = ["randomwalk", "randomwalknowcast", "renewal"]


def _table() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "model": ["renewal", "randomwalk", "randomwalk", "hubteam"],
            "horizon": [0, 1, 0, 0],
            "n": [4, 3, 4, 4],
            "wis": [2.0, 1.5, 1.0, 9.0],
            "coverage_90": [0.5, None, 1.0, 0.2],
        }
    )


def test_score_chart_data_keeps_listed_models_in_their_colours() -> None:
    data = score_chart_data(
        _table(),
        "horizon",
        {"wis": ("WIS", None), "coverage_90": ("90% coverage", 0.9)},
        MODELS,
        default_on=["randomwalk"],
    )
    names = [m["name"] for m in data["models"]]
    assert names == ["randomwalk", "renewal"]
    renewal = data["models"][1]
    assert renewal["light"] == model_colour("renewal", MODELS, "light")
    assert not renewal["on"]
    assert data["x"] == [0, 1]
    assert data["metrics"][1] == {
        "key": "coverage_90",
        "label": "90% coverage",
        "ref": 0.9,
    }
    # Model index, x index, n, then the metrics, with a gap as None.
    assert [0, 1, 3, 1.5, None] in data["rows"]
    assert len(data["rows"]) == 3


def test_score_chart_html_falls_back_to_the_static_figure() -> None:
    html = score_chart_html("../f/a.json", "Scores", static="../f/a.svg")
    assert 'data-src="../f/a.json"' in html
    assert "../f/a-dark.svg#only-dark" in html


def test_score_chart_data_takes_fixed_colours() -> None:
    colours = {
        "randomwalk": ("#111111", "#222222"),
        "renewal": ("#333", "#444"),
    }
    data = score_chart_data(
        _table(), "horizon", {"wis": ("WIS", None)}, MODELS, colours=colours
    )
    assert [(m["light"], m["dark"]) for m in data["models"]] == [
        ("#111111", "#222222"),
        ("#333", "#444"),
    ]
