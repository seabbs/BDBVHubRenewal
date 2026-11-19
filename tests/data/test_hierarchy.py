"""Location hierarchies read from ISO 3166-2 style codes."""

from __future__ import annotations

import pandas as pd

from bdbvhubrenewal.data.hierarchy import depth, parent
from bdbvhubrenewal.data.snapshots import snapshot_arrays


def test_parent_drops_the_last_part() -> None:
    assert parent("CD-IT-01") == "CD-IT"
    assert parent("CD-IT") == "CD"
    assert parent("DE-BY") == "DE"
    assert parent("CD") is None


def test_depth_counts_the_parts_below_the_root() -> None:
    assert [depth(c) for c in ("DE", "DE-BY", "CD-IT-01")] == [0, 1, 2]


def test_snapshots_take_the_root_from_the_codes() -> None:
    daily = pd.DataFrame(
        {
            "vintage": ["2026-10-24"] * 2,
            "location": ["DE-BY", "DE-BE"],
            "onset_date": ["2026-10-20", "2026-10-24"],
            "observed": [2, 3],
            "imputed": [0, 0],
            "cutoff": ["2026-10-24"] * 2,
        }
    )
    snap = snapshot_arrays(daily, "2026-10-24", ("DE-BY", "DE-BE"))
    assert snap.locations == ("DE", "DE-BE", "DE-BY")
    assert snap.parent.tolist() == [-1, 0, 0]
    assert snap.counts[1:, 0].sum() == 5
