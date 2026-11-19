import matplotlib.pyplot as plt
import pytest
from bdbvhubrenewal.evaluation.labels import model_label
from matplotlib.lines import Line2D
from paperfigs.style import shared_legend


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
    fig, axes = plt.subplots(1, len(labels_by_panel), layout="constrained")
    for ax, labels in zip(axes, labels_by_panel, strict=True):
        for label in labels:
            ax.plot([0, 1], [0, 1], label=label)
    return fig, axes


def test_shared_legend_is_one_deduplicated_legend_on_top() -> None:
    rw, ft = model_label("randomwalknowcast"), model_label(
        "fm_chronos_daily_ragged_ft"
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


@pytest.fixture
def one_top_legend():
    return assert_one_top_legend
