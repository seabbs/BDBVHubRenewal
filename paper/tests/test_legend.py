import matplotlib.pyplot as plt
import pandas as pd
import pytest
from bdbvhubrenewal.evaluation.labels import model_label
from matplotlib.lines import Line2D

from paperfigs.dummy import dummy_study
from paperfigs.plots import (
    results_panel,
    si_scores_by_location_horizon,
    si_wis_over_time,
)
from paperfigs.scores import score_forecasts
from paperfigs.style import model_order, shared_legend


def assert_one_top_legend(fig) -> list[str]:
    """One figure legend above every panel, none per panel; its labels."""
    assert len(fig.legends) == 1
    assert all(ax.get_legend() is None for ax in fig.axes)
    fig.canvas.draw()
    legend = fig.legends[0].get_window_extent()
    tops = [ax.get_window_extent().y1 for ax in fig.axes if ax.get_visible()]
    assert legend.y0 >= max(tops) - 1
    labels = [t.get_text() for t in fig.legends[0].get_texts()]
    assert len(labels) == len(set(labels))
    return labels


def _fig_with(labels_by_panel):
    fig, axes = plt.subplots(
        1, len(labels_by_panel), layout="constrained", squeeze=False
    )
    axes = axes.ravel()
    for ax, labels in zip(axes, labels_by_panel, strict=True):
        for label in labels:
            ax.plot([0, 1], [0, 1], label=label)
    return fig, axes


def test_shared_legend_is_one_deduplicated_legend_on_top() -> None:
    rw, ft = (
        model_label("randomwalknowcast"),
        model_label("fm_chronos_daily_ragged_ft"),
    )
    fig, axes = _fig_with([[ft, "Truth"], [ft, rw]])
    shared_legend(fig, axes)
    labels = assert_one_top_legend(fig)
    # Models in the fixed model order first, then the rest as drawn.
    assert labels == [rw, ft, "Truth"]
    plt.close(fig)


def test_shared_legend_orders_checkpoints_and_made_notes_by_model() -> None:
    ft = f"{model_label('fm_chronos_daily_ragged_ft')}, 2,000 outbreaks"
    rw = f"{model_label('randomwalknowcast')} (90% made)"
    fig, axes = _fig_with([[ft, rw]])
    shared_legend(fig, axes)
    assert assert_one_top_legend(fig) == [rw, ft]
    plt.close(fig)


def test_shared_legend_takes_extra_handles() -> None:
    fig, axes = _fig_with([["a"]])
    extra = [Line2D([], [], label="Dispersion")]
    shared_legend(fig, axes, extra=extra)
    assert assert_one_top_legend(fig) == ["a", "Dispersion"]
    plt.close(fig)


@pytest.fixture(scope="module")
def germany():
    fc, obs = dummy_study("germany", seed=1, n_samples=5)
    return fc, obs, score_forecasts(fc)


def test_paper_figures_have_one_top_legend_in_model_order(germany) -> None:
    fc, obs, scores = germany
    split = pd.Timestamp("2022-02-10")
    figs = [
        results_panel(fc, obs, scores, split=split),
        si_wis_over_time(scores, split),
        si_scores_by_location_horizon(scores),
    ]
    order = [model_label(m) for m in model_order(scores["model"].unique())]
    for fig in figs:
        labels = assert_one_top_legend(fig)
        models = [label for label in labels if label in order]
        assert models == [m for m in order if m in models]
        plt.close(fig)
