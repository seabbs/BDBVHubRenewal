"""Model inputs the task carries: the imputation delay and the places."""

from __future__ import annotations

import numpy as np
import pandas as pd

from bdbvhub.data.delays import hub_delay_pmf
from bdbvhub.data.zones import zone_places
from nowcastrenewal.model.inputs import imputation_pmf, province_distances


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


def test_zone_places_are_the_mapped_zones() -> None:
    out = zone_places()
    assert list(out.columns) == [
        "location",
        "population",
        "latitude",
        "longitude",
    ]
    assert out["location"].str.match(r"^CD-[A-Z]{2}-[0-9]{2}$").all()


def test_unmapped_zones_take_their_province_place() -> None:
    from nowcastrenewal.zones import _zone_places
    from nowcastutils.base import ForecastTask

    places = pd.DataFrame(
        {
            "location": ["DE-BY", "DE-BY-01", "DE-BY-02"],
            "population": [100.0, 10.0, 30.0],
            "latitude": [48.0, 47.0, 49.0],
            "longitude": [11.0, 10.0, 12.0],
        }
    )
    task = ForecastTask(
        vintage="2026-10-24",
        reference_date=pd.Timestamp("2026-10-24").date(),
        kind="real",
        cutoff="2026-10-24",
        target=pd.DataFrame(),
        delays=None,
        locations=("DE-BY-02", "DE-BY-09"),
        target_end_dates=(),
        places=places,
    )
    out = _zone_places(task).set_index("location")
    assert out.loc["DE-BY-02", "population"] == 30.0
    assert out.loc["DE-BY-09", "population"] == 20.0
    assert out.loc["DE-BY-09", "latitude"] == 48.0


def test_an_imputation_delay_with_no_mass_is_an_error() -> None:
    import pytest

    with pytest.raises(ValueError, match="no mass"):
        imputation_pmf(np.array([0.0, 0.0, 1.0]), 1)
