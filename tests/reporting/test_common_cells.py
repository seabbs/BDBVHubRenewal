"""Summary tables score every model on the cells all of them forecast."""

from __future__ import annotations

import pandas as pd

from bdbvhubrenewal.evaluation.nowcast import relative_wis
from bdbvhubrenewal.reporting.scoresummary import common_cells


def _rows() -> pd.DataFrame:
    rows = []
    for model in ("randomwalknowcast", "renewal", "teamA"):
        for origin in ("2026-10-17", "2026-10-24"):
            for loc in ("CD", "CD-IT", "CD-IT-01"):
                # renewal has no zones and no 2026-10-17 origin.
                if model == "renewal" and (
                    loc == "CD-IT-01" or origin == "2026-10-17"
                ):
                    continue
                rows.append(
                    {
                        "model": model,
                        "target": "complete",
                        "origin": origin,
                        "truth_vintage": "2026-11-08",
                        "location": loc,
                        "target_end_date": "2026-10-03",
                        "pool": "zones" if loc.count("-") == 2 else "rest",
                        "wis": 1.0 + len(model) + len(origin) % 3,
                        "log_wis": 0.1,
                        "coverage_50": 1.0,
                        "coverage_90": 1.0,
                        "bias": 0.0,
                    }
                )
    return pd.DataFrame(rows)


def test_each_table_row_has_the_same_n() -> None:
    kept = common_cells(_rows(), by=("pool",))
    table = relative_wis(kept, by=("pool",), baseline="randomwalknowcast")
    # Outside the zones every model has the 2026-10-24 cells only.
    rest = table.loc[table["pool"] == "rest"]
    assert set(rest["n"]) == {2}
    assert len(rest) == 3
    # renewal has no zone cells, so the zone rows are the others', all
    # on the same cells.
    zones = table.loc[table["pool"] == "zones"]
    assert set(zones["model"]) == {"randomwalknowcast", "teamA"}
    assert set(zones["n"]) == {2}


def test_left_out_cells_are_counted_per_model() -> None:
    kept = common_cells(_rows(), by=("pool",))
    left = kept.attrs["left_out"].set_index(["pool", "model"])["cells"]
    assert left[("rest", "randomwalknowcast")] == 2
    assert left[("rest", "teamA")] == 2
    assert ("rest", "renewal") not in left.index
