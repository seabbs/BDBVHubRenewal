from __future__ import annotations

import pandas as pd

from bdbvhubrenewal.reporting.nowcast_plots import plot_relative_overall


def test_plot_relative_overall_skips_models_missing_from_a_pool() -> None:
    table = pd.DataFrame(
        {
            "pool": ["national", "national", "zones"],
            "model": ["randomwalk", "renewal", "randomwalk"],
            "rel_wis": [1.0, 0.8, 1.0],
            "coverage_90": [0.9, 0.7, 0.85],
        }
    )
    fig = plot_relative_overall(table, ["randomwalk", "renewal"])
    wis, cover = fig.axes[:2]
    # Three bars, as renewal has no zones.
    assert len(wis.patches) == 3
    assert [t.get_text() for t in wis.get_xticklabels()] == [
        "National",
        "Zones",
    ]
    assert cover.get_ylim() == (0, 1)
