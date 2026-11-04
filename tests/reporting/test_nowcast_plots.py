from __future__ import annotations

import pandas as pd

from bdbvhubrenewal.reporting.nowcast_plots import plot_relative_overall


def test_plot_relative_overall_skips_models_missing_from_a_pool() -> None:
    table = pd.DataFrame(
        {
            "pool": ["national", "national", "zones"],
            "model": ["randomwalk", "renewal", "randomwalk"],
            "rel_wis": [1.0, 0.8, 1.0],
            "rel_log_wis": [1.0, 0.9, 1.0],
            "coverage_50": [0.5, 0.4, 0.45],
            "coverage_90": [0.9, 0.7, 0.85],
            "bias": [0.1, -0.2, 0.0],
        }
    )
    fig = plot_relative_overall(table, ["randomwalk", "renewal"])
    shown = [ax for ax in fig.axes if ax.get_visible()]
    assert [ax.get_ylabel() for ax in shown] == [
        "Relative WIS to randomwalk",
        "Relative log WIS to randomwalk",
        "50% coverage",
        "90% coverage",
        "Bias",
    ]
    wis, cover = shown[0], shown[3]
    # Three bars, as renewal has no zones.
    assert len(wis.patches) == 3
    assert [t.get_text() for t in wis.get_xticklabels()] == [
        "National",
        "Zones",
    ]
    assert cover.get_ylim() == (0, 1)
    assert shown[4].get_ylim() == (-1, 1)
