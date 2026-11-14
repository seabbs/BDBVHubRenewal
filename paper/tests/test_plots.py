import matplotlib as mpl

mpl.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from paperfigs.dummy import dummy_factors, dummy_study
from paperfigs.plots import (
    PANEL_LETTERS,
    results_panel,
    si_scores_by_factor,
    si_scores_by_location_horizon,
    si_wis_over_time,
)
from paperfigs.schema import STUDIES
from paperfigs.scores import score_forecasts
from paperfigs.style import bar_style, model_order, model_style, point_style


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
        for m in ("randomwalk", "randomwalknowcast", "KIT-frozen_baseline")
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


def test_synthetic_panel_shows_factors_and_weekly_horizons() -> None:
    fc, obs = dummy_study("synthetic", seed=1, n_samples=20)
    scores = score_forecasts(fc)
    factors = dummy_factors(fc, seed=1)
    # Rank delay_q98 exactly as delay_mean, so delay_mean is redundant.
    factors["delay_q98"] = factors["delay_mean"] * 10
    fig = results_panel(fc, obs, scores, factors=factors)
    titles = " ".join(ax.get_title() for ax in fig.axes)
    assert "by scenario factor" in titles
    assert "by location" not in titles
    groups = {t.get_text() for ax in fig.axes for t in ax.texts}
    assert "Delay q98" in groups
    assert "Delay mean" not in groups
    labels = " ".join(ax.get_xlabel() for ax in fig.axes)
    assert "Horizon (weeks)" in labels
    plt.close(fig)
    fig = si_scores_by_factor(scores, factors)
    # The SI keeps every factor, delay_mean included.
    assert len([ax for ax in fig.axes if ax.get_title()]) == 6
    plt.close(fig)


@pytest.mark.parametrize("family", ["daily", "weekly"])
def test_chronos_variants_share_a_colour_with_distinct_styles(family) -> None:
    names = [
        f"fm_chronos_{family}_complete",
        f"fm_chronos_{family}_ragged",
        f"fm_chronos_{family}_ragged_ft",
    ]
    styles = [model_style(n, []) for n in names]
    assert len({c for c, _ in styles}) == 1
    assert [d for _, d in styles] == ["dotted", "dashed", "solid"]
    bars = [bar_style(n, []).get("hatch") for n in names]
    assert len(set(bars)) == 3
    points = [tuple(sorted(point_style(n, []).items())) for n in names]
    assert len(set(points)) == 3
    assert model_order(list(reversed(names))) == names


@pytest.fixture(scope="module")
def germany():
    fc, obs = dummy_study("germany", seed=1, n_samples=5)
    return fc, obs, score_forecasts(fc), pd.Timestamp("2022-02-10")


def _texts(fig) -> str:
    return " ".join(t.get_text() for ax in fig.axes for t in ax.texts)


def test_split_draws_both_halves_and_the_line(germany) -> None:
    fc, obs, scores, split = germany
    fig = results_panel(fc, obs, scores, split=split)
    assert "development | held out" in _texts(fig)
    labels = [t.get_text() for t in fig.legends[0].get_texts()]
    assert "Development half" in labels
    assert "Held-out half" in labels
    rel_ax = next(ax for ax in fig.axes if ax.get_title() == "Relative WIS")
    n_models = scores["model"].nunique()
    assert len(rel_ax.patches) == 2 * n_models
    plt.close(fig)


def test_no_split_draws_one_bar_per_model(germany) -> None:
    fc, obs, scores, _ = germany
    fig = results_panel(fc, obs, scores, split=None)
    rel_ax = next(ax for ax in fig.axes if ax.get_title() == "Relative WIS")
    assert len(rel_ax.patches) == scores["model"].nunique()
    assert "development | held out" not in _texts(fig)
    plt.close(fig)


def test_si_figures_split_by_half(germany) -> None:
    _, _, scores, split = germany
    fig = si_wis_over_time(scores, split=split)
    assert "development | held out" in _texts(fig)
    plt.close(fig)
    one = si_scores_by_location_horizon(scores, split=split)
    none = si_scores_by_location_horizon(scores)
    assert len(one.axes[0].lines) > len(none.axes[0].lines)
    plt.close(one)
    plt.close(none)


def test_ribbons_fall_back_to_the_best_chronos_variant() -> None:
    from paperfigs.plots import _ribbon_models

    models = [
        "randomwalknowcast",
        "fm_chronos_daily_complete",
        "fm_chronos_daily_ragged",
        "fm_chronos_weekly_complete",
        "fm_chronos_weekly_ragged",
        "fm_chronos_weekly_ragged_ft",
    ]
    assert _ribbon_models(models, "randomwalknowcast") == [
        "randomwalknowcast",
        "fm_chronos_daily_ragged",
        "fm_chronos_weekly_ragged_ft",
    ]


@pytest.fixture(scope="module")
def gappy():
    fc, obs = dummy_study("ebola", seed=1, n_samples=5)
    origins = sorted(fc["origin_date"].unique())
    drop = (fc["model"] == "randomwalknowcast") & fc["origin_date"].isin(
        origins[:6]
    )
    fc = fc.loc[~drop]
    return fc, obs, score_forecasts(fc), origins[:6]


def test_missing_forecasts_are_labelled(gappy) -> None:
    fc, obs, scores, _ = gappy
    fig = results_panel(fc, obs, scores)
    labels = [
        t.get_text()
        for ax in fig.axes
        for t in ax.get_yticklabels()
        if "randomwalknowcast" in t.get_text()
    ]
    assert any("% made" in label for label in labels)
    assert not any(
        "% made" in t.get_text()
        for ax in fig.axes
        for t in ax.get_yticklabels()
        if t.get_text().startswith("BDBVHub-baseline")
    )
    plt.close(fig)


def test_missing_origins_break_lines_over_time(gappy) -> None:
    _, _, scores, dropped = gappy
    fig = si_wis_over_time(scores)
    ax = fig.axes[0]
    line = next(ln for ln in ax.get_lines() if ln.get_color() == "#eb6834")
    y = np.asarray(line.get_ydata(), dtype=float)
    assert np.isnan(y).sum() >= len(dropped)
    plt.close(fig)


def test_nowcast_panel_marks_a_missing_ribbon(gappy) -> None:
    fc, obs, scores, _ = gappy
    fig = results_panel(
        fc, obs, scores, ribbon_models=["randomwalknowcast"], split=None
    )
    ax = next(a for a in fig.axes if a.get_title() == "Nowcasts over time")
    assert any(ln.get_marker() == "x" for ln in ax.get_lines())
    labels = [t.get_text() for t in fig.legends[0].get_texts()]
    assert "No forecast at origin" in labels
    plt.close(fig)


def test_finetune_size_plots_one_line_per_finetuned_model() -> None:
    from paperfigs.plots import si_scores_by_finetune_size

    fc, _ = dummy_study("synthetic", seed=1, n_samples=10)
    fig = si_scores_by_finetune_size(score_forecasts(fc))
    ax = fig.axes[0]
    # The reference line at 1 spans the axes, from 0 to 1.
    xs = sorted(
        {x for line in ax.get_lines() for x in line.get_xdata()} - {0, 1}
    )
    assert xs[0] == 1000
    assert xs[-1] == 16000
    assert ax.get_xscale() == "log"
    plt.close(fig)


def test_nowcasts_draw_each_dates_latest_count_when_releases_are_partial():
    fc, obs = dummy_study("germany", seed=1, n_samples=5)
    # The last release holds only its last week, as the German data do.
    last = obs["as_of_date"].max()
    obs = obs.loc[
        (obs["as_of_date"] < last)
        | (obs["target_date"] > last - pd.Timedelta(days=7))
    ]
    fig = results_panel(fc, obs, score_forecasts(fc))
    plt.close(fig)
