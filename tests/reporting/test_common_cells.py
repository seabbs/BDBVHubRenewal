"""Summary tables put a model that lacks some of the cells apart."""

from __future__ import annotations

import pandas as pd

from bdbvhubrenewal.evaluation.nowcast import relative_wis
from bdbvhubrenewal.reporting.scoresummary import mark_partial


def _rows() -> pd.DataFrame:
    rows = []
    for model in ("randomwalknowcast", "renewal", "teamA"):
        for origin in ("2026-10-17", "2026-10-24"):
            for loc in ("CD", "CD-IT"):
                # renewal has no 2026-10-17 origin.
                if model == "renewal" and origin == "2026-10-17":
                    continue
                rows.append(
                    {
                        "model": model,
                        "target": "complete",
                        "origin": origin,
                        "truth_vintage": "2026-11-08",
                        "location": loc,
                        "target_end_date": "2026-10-03",
                        "pool": "all",
                        "wis": 1.0 + len(model),
                        "log_wis": 0.1,
                        "coverage_50": 1.0,
                        "coverage_90": 1.0,
                        "bias": 0.0,
                    }
                )
    return pd.DataFrame(rows)


def test_full_rows_share_n_and_a_model_lacking_cells_is_marked() -> None:
    table = mark_partial(
        relative_wis(_rows(), by=("pool",), baseline="randomwalknowcast"),
        by=("pool",),
        baseline="randomwalknowcast",
    ).set_index("model")
    # The baseline's 4 cells: every full row has n 4.
    assert set(table["cells"]) == {4}
    full = table.loc[~table["partial"]]
    assert set(full.index) == {"randomwalknowcast", "teamA"}
    assert set(full["n"]) == {4}
    # renewal has 2 of them, so its row is apart and says so.
    assert bool(table.loc["renewal", "partial"])
    assert table.loc["renewal", "n"] == 2


def test_report_relative_crps_marks_partial_rows() -> None:
    from bdbvhubrenewal.reporting.report import relative_crps

    rows = _rows()
    rows["crps"] = rows["wis"]
    rows["log_crps"] = rows["log_wis"]
    table = relative_crps(
        rows, by=("pool",), baseline="randomwalknowcast"
    ).set_index("model")
    assert {"rel_crps", "rel_log_crps", "cells", "partial"} <= set(
        table.columns
    )
    assert table.loc["randomwalknowcast", "rel_crps"] == 1.0
    assert bool(table.loc["renewal", "partial"])
