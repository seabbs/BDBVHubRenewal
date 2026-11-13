import matplotlib as mpl

mpl.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import pytest

from paperfigs.dummy import dummy_study
from paperfigs.plots import (
    PANEL_LETTERS,
    results_panel,
    si_scores_by_location_horizon,
    si_wis_over_time,
)
from paperfigs.schema import STUDIES
from paperfigs.scores import score_forecasts
from paperfigs.style import model_style


@pytest.fixture(scope="module", params=STUDIES)
def study(request):
    fc, obs = dummy_study(request.param, seed=1, n_samples=20)
    return fc, obs, score_forecasts(fc)


def test_model_style_is_fixed_whatever_else_is_shown() -> None:
    a = model_style("randomwalknowcast", ["randomwalknowcast", "x"])
    b = model_style("randomwalknowcast", ["hub_team_a", "randomwalknowcast"])
    assert a == b


def test_baselines_have_their_own_colours() -> None:
    colours = {
        model_style(m, [])[0]
        for m in ("randomwalk", "randomwalknowcast", "hub_baseline")
    }
    assert len(colours) == 3


def test_fine_tuned_and_zero_shot_share_a_colour_not_a_dash() -> None:
    ft = model_style("fm_chronos_daily_ragged_ft", [])
    zs = model_style("fm_chronos_daily_ragged", [])
    assert ft[0] == zs[0]
    assert ft[1] != zs[1]


def test_unknown_models_get_distinct_styles() -> None:
    models = ["new_a", "new_b"]
    assert model_style("new_a", models) != model_style("new_b", models)


def test_results_panel_has_lettered_panels(study) -> None:
    fc, obs, scores = study
    fig = results_panel(fc, obs, scores)
    letters = sorted(
        ax.get_title(loc="left") for ax in fig.axes if ax.get_title(loc="left")
    )
    stratified = fc["stratum"].nunique() > 1
    assert letters == list(PANEL_LETTERS[: 8 if stratified else 7])
    if stratified:
        assert any("by stratum" in ax.get_title() for ax in fig.axes)
    metric = "WIS" if fc["study"].iloc[0] == "germany" else "CRPS"
    titles = " ".join(ax.get_title() for ax in fig.axes)
    assert f"Relative {metric}" in titles
    assert fig.legends
    plt.close(fig)


def test_results_panel_needs_one_study(study) -> None:
    fc, obs, scores = study
    name = fc["study"].iloc[0]
    other = fc.assign(study="germany" if name == "ebola" else "ebola")
    with pytest.raises(ValueError, match="one study"):
        results_panel(pd.concat([fc, other]), obs, scores)


def test_si_figures_render(study) -> None:
    _, _, scores = study
    n_loc = len(scores[["location", "stratum"]].drop_duplicates())
    fig = si_scores_by_location_horizon(scores)
    assert sum(ax.get_visible() for ax in fig.axes) == n_loc
    plt.close(fig)
    fig = si_wis_over_time(scores)
    assert len(fig.axes) == 2
    plt.close(fig)


def test_coverage_panel_follows_the_study(study) -> None:
    fc, obs, scores = study
    fig = results_panel(fc, obs, scores)
    labels = " ".join(ax.get_xlabel() for ax in fig.axes)
    want = "95%" if fc["study"].iloc[0] == "germany" else "90%"
    assert want in labels
    plt.close(fig)
