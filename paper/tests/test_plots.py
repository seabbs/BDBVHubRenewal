import matplotlib as mpl

mpl.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from bdbvhubrenewal.evaluation.labels import model_label

from paperfigs.dummy import DUMMY_CELLS, dummy_factors, dummy_study
from paperfigs.periods import PERIODS
from paperfigs.plots import (
    PANEL_LETTERS,
    results_panel,
    si_scores_by_factor,
    si_scores_by_location_horizon,
    si_wis_over_time,
)
from paperfigs.schema import STUDIES
from paperfigs.scores import relative_scores, score_forecasts
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
    # Sample studies are scored on their quantiles too.
    titles = " ".join(ax.get_title() for ax in fig.axes)
    assert "Relative WIS" in titles
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
    # The SI keeps every grid factor and both delays.
    n = len(DUMMY_CELLS) + 2
    assert len([ax for ax in fig.axes if ax.get_title()]) == n
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
    # Bars are filled with the model's colour, never hatched (Sam, #367).
    bars = [bar_style(n, []) for n in names]
    assert all("hatch" not in b for b in bars)
    assert len({b["color"] for b in bars}) == 1
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
    # Three WIS parts per model and half.
    n_models = scores["model"].nunique()
    assert len(rel_ax.patches) == 3 * 2 * n_models
    plt.close(fig)


def test_no_split_draws_one_bar_per_model(germany) -> None:
    fc, obs, scores, _ = germany
    fig = results_panel(fc, obs, scores, split=None)
    rel_ax = next(ax for ax in fig.axes if ax.get_title() == "Relative WIS")
    assert len(rel_ax.patches) == 3 * scores["model"].nunique()
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
    labels = [t.get_text() for t in fig.legends[0].get_texts()]
    assert any(
        "% made" in label and "Random walk nowcast" in label for label in labels
    )
    assert not any(
        "% made" in label for label in labels if label.startswith("Hub")
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


def test_nowcast_panels_have_a_column_per_model_and_a_row_per_origin_or_place():
    from paperfigs.plots import nowcast_panels
    from paperfigs.truth import rolling_truth

    fc, obs = dummy_study("germany", seed=1, n_samples=5)
    truth = rolling_truth(obs, delay_days=40)
    origins = [pd.Timestamp(o) for o in sorted(fc["origin_date"].unique())[:2]]
    models = ["randomwalknowcast", "fm_chronos_daily_ragged_ft"]
    rows = [(o, "DE", "all", f"origin {o:%Y-%m-%d}") for o in origins]
    rows.append((origins[0], "DE", "80+", "DE, 80+"))
    fig = nowcast_panels(
        fc, obs, truth, models, rows, rel={"randomwalknowcast": 1.0}
    )
    assert len(fig.axes) == 6
    assert fig.axes[0].get_title() == "Random walk nowcast\nrel. 1.00"
    assert fig.axes[1].get_title() == "Fine-tuned (delays)"
    assert fig.axes[0].get_ylabel().startswith("origin ")
    assert fig.axes[4].get_ylabel() == "DE, 80+"
    plt.close(fig)


def test_place_dots_have_a_row_per_place() -> None:
    from paperfigs.plots import si_place_dots

    fc, _ = dummy_study("germany", seed=1, n_samples=5)
    scores = score_forecasts(fc)
    fig = si_place_dots(scores, ["randomwalknowcast", "hub_ensemble"])
    ax = fig.axes[0]
    places = {(loc, st) for loc, st in fc[["location", "stratum"]].to_numpy()}
    assert len(ax.get_yticks()) == len(places)
    plt.close(fig)


def test_rows_of_a_target_keep_that_target_only() -> None:
    from paperfigs.plots import rows_of_target

    df = pd.DataFrame(
        {"model": ["a", "a", "hub"], "target": ["daily", "7day", "7day"]}
    )
    assert list(rows_of_target(df, "daily")["model"]) == ["a"]
    assert len(rows_of_target(df, None)) == 3
    # Tables with no target column hold the hub's target only.
    plain = df.drop(columns="target")
    assert rows_of_target(plain, "daily").empty
    assert len(rows_of_target(plain, "7day")) == 3


def test_germany_has_a_daily_target_si_figure() -> None:
    from paperpipe.outputs import FIGURES

    assert "si-results-daily" in FIGURES["germany"]["si"]


def test_wis_bars_stack_their_parts_to_the_relative_wis(germany) -> None:
    fc, obs, scores, _ = germany
    fig = results_panel(fc, obs, scores, split=None)
    rel_ax = next(ax for ax in fig.axes if ax.get_title() == "Relative WIS")
    rel = relative_scores(scores).set_index("model")
    models = model_order(scores["model"].unique())
    ends = {}
    for bar in rel_ax.patches:
        y = round(bar.get_y() + bar.get_height() / 2, 6)
        ends[y] = max(ends.get(y, 0.0), bar.get_x() + bar.get_width())
    assert len(ends) == len(models)
    for y, m in zip(sorted(ends), models, strict=True):
        assert ends[y] == pytest.approx(rel.loc[m, "rel_score"])
    labels = [t.get_text() for t in fig.legends[0].get_texts()]
    assert {"Dispersion", "Overprediction", "Underprediction"} <= set(labels)
    plt.close(fig)


def test_crps_bars_are_drawn_from_one(study) -> None:
    fc, obs, _ = study
    if fc["sample_id"].isna().all():
        pytest.skip("quantile study")
    scores = score_forecasts(fc, samples="crps")
    fig = results_panel(fc, obs, scores)
    rel_ax = next(ax for ax in fig.axes if ax.get_title() == "Relative CRPS")
    assert len(rel_ax.patches) == scores["model"].nunique()
    assert rel_ax.get_xscale() == "log"
    plt.close(fig)


def test_results_panel_shows_only_the_given_models(germany) -> None:
    fc, obs, scores, _ = germany
    keep = ["randomwalknowcast", "fm_chronos_daily_ragged_ft"]
    fig = results_panel(fc, obs, scores, split=None, models=keep)
    rel_ax = next(ax for ax in fig.axes if ax.get_title() == "Relative WIS")
    # The two models and the study's baseline, three parts each.
    assert len(rel_ax.patches) == 3 * 3
    plt.close(fig)


def _bar_axes(fig):
    titles = {"Relative WIS", "Relative WIS, log scale", "Bias", "Coverage"}
    return [ax for ax in fig.axes if ax.get_title() in titles]


def test_split_bars_put_only_the_period_on_the_axis(germany) -> None:
    fc, obs, scores, split = germany
    fig = results_panel(fc, obs, scores, split=split)
    models = model_order(scores["model"].unique())
    colours = [model_style(m, [])[0] for m in models]
    for ax in _bar_axes(fig):
        # The axis names the periods only; models are told by colour.
        ticks = [t.get_text() for t in ax.get_yticklabels()]
        assert ticks == list(PERIODS)
        # Within each period, one bar per model side by side, in order.
        step = 0.8 / len(models)
        for c in ax.get_yticks():
            bars = {}
            for bar in ax.patches:
                y = bar.get_y() + bar.get_height() / 2
                if abs(y - c) < 0.5:
                    # Coverage has two bars per model; both are its slot.
                    slot = round((y - c) / step + (len(models) - 1) / 2)
                    bars.setdefault(slot, []).extend(
                        [bar.get_facecolor()[:3], bar.get_edgecolor()[:3]]
                    )
            assert len(bars) == len(models)
            for y, colour in zip(sorted(bars), colours, strict=True):
                assert any(mpl.colors.same_color(c, colour) for c in bars[y])
    legend = [t.get_text() for t in fig.legends[0].get_texts()]
    assert all(model_label(m) in legend for m in models)
    plt.close(fig)


def test_unsplit_bars_have_no_axis_names(gappy) -> None:
    fc, obs, scores, _ = gappy
    fig = results_panel(fc, obs, scores)
    for ax in _bar_axes(fig):
        assert all(not t.get_text() for t in ax.get_yticklabels())
    plt.close(fig)


def test_bars_are_neither_hatched_nor_faded(germany) -> None:
    fc, obs, scores, split = germany
    fig = results_panel(fc, obs, scores, split=split)
    bar_axes = [
        ax
        for ax in fig.axes
        if ax.get_title() in {"Relative WIS", "Relative WIS, log scale"}
        or ax.get_title() in {"Bias", "Coverage"}
    ]
    assert len(bar_axes) == 4
    for ax in bar_axes:
        for bar in ax.patches:
            assert not bar.get_hatch()
            assert bar.get_alpha() in (None, 1.0)
    plt.close(fig)


def test_no_model_id_reaches_a_figure(germany) -> None:
    fc, obs, scores, split = germany
    fig = results_panel(fc, obs, scores, split=split)
    texts = [t.get_text() for t in fig.legends[0].get_texts()]
    texts += [t.get_text() for ax in fig.axes for t in ax.get_yticklabels()]
    texts += [ax.get_xlabel() for ax in fig.axes]
    named = [m for m in scores["model"].unique() if model_label(m) != m]
    assert named
    assert not any(m in t for m in named for t in texts)
    plt.close(fig)


def test_main_results_open_with_nowcasts_over_time(germany) -> None:
    fc, obs, scores, split = germany
    fig = results_panel(fc, obs, scores, split=split, nowcast=True)
    titles = [ax.get_title() for ax in fig.axes]
    assert "Nowcasts over time" in titles
    plt.close(fig)
