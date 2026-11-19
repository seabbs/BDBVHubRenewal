"""Model inputs the task carries: the imputation delay and the places."""

from __future__ import annotations

import numpy as np
import pandas as pd

from bdbvhubrenewal.data.delays import hub_delay_pmf
from bdbvhubrenewal.data.zones import zone_places
from bdbvhubrenewal.model.inputs import imputation_pmf, province_distances


def test_imputation_pmf_truncates_and_renormalises() -> None:
    full = hub_delay_pmf("2026-10-24", 60)
    np.testing.assert_allclose(
        imputation_pmf(full, 30), hub_delay_pmf("2026-10-24", 30)
    )
    padded = imputation_pmf(np.array([0.5, 0.5]), 3)
    np.testing.assert_allclose(padded, [0.5, 0.5, 0.0, 0.0])


def test_distances_come_from_the_places_table() -> None:
    places = pd.DataFrame(
        {
            "location": ["A", "B"],
            "latitude": [0.0, 0.0],
            "longitude": [0.0, 1.0],
        }
    )
    d = province_distances(("A", "B"), places)
    assert d[0, 0] == 0
    np.testing.assert_allclose(d[0, 1], 111.19, rtol=1e-3)


def test_zone_places_fall_back_to_the_province() -> None:
    out = zone_places(("CD-IT-01", "CD-IT-99"))
    assert list(out.columns) == [
        "location",
        "population",
        "latitude",
        "longitude",
    ]
    assert out["location"].tolist() == ["CD-IT-01", "CD-IT-99"]
