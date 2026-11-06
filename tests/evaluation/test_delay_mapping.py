"""Each model is mapped to what later releases show by its own delay."""

from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd
import pytest

from bdbvhubrenewal.evaluation.nowcast import (
    _reported_target,
    delay_curves,
    week_shares,
)


def _summary(location: str, shares: list[float]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "model": "m",
            "vintage": "2026-10-24",
            "location": location,
            "kind": "cumulative",
            "x": np.arange(len(shares)),
            "quantity": "reported_share",
            "q50": shares,
        }
    )


def test_a_week_s_share_is_the_mean_over_its_days() -> None:
    summary = pd.concat(
        [
            _summary("CD-IT", [0.0, 0.5, 1.0]),
            _summary("CD-IT", [9.0]).assign(quantity="median_delay"),
        ]
    )
    curves = delay_curves(summary, ["CD-IT"])
    weeks = ["2026-10-24"]
    # The week ending Saturday 10-24 seen at that day: ages 6 to 0.
    got = week_shares(curves, dt.date(2026, 10, 24), weeks)
    assert got["share"].iloc[0] == pytest.approx((0 + 0.5 + 5 * 1.0) / 7)
    # Days after the cut-off are not reported.
    early = week_shares(curves, dt.date(2026, 10, 20), weeks)
    assert early["share"].iloc[0] == pytest.approx((1 + 0.5 + 0) / 7)


def test_a_place_takes_its_own_delay_and_a_zone_its_province_s() -> None:
    summary = pd.concat([_summary("CD-IT", [0.2]), _summary("CD-NK", [0.8])])
    curves = delay_curves(summary, ["CD", "CD-IT", "CD-NK", "CD-IT-01"])
    got = curves.groupby("location")["share"].first()
    assert got.to_dict() == {"CD-IT": 0.2, "CD-IT-01": 0.2, "CD-NK": 0.8}
    # A delay fitted once for every place holds at every place.
    pooled = delay_curves(_summary("pooled", [0.5]), ["CD", "CD-IT"])
    assert set(pooled["location"]) == {"CD", "CD-IT"}


def test_the_country_is_the_sum_of_mapped_provinces_without_its_own(
    daily: pd.DataFrame,
) -> None:
    weeks = ["2026-10-17"]
    n = 50
    rows = {"CD-IT": 100.0, "CD-NK": 50.0, "CD": 150.0}
    samples = pd.DataFrame(
        {
            "location": np.repeat(list(rows), n),
            "target_end_date": weeks[0],
            "sample": np.tile(np.arange(1, n + 1), len(rows)),
            "value": np.repeat(list(rows.values()), n),
        }
    )
    summary = pd.concat(
        [_summary("CD-IT", [0.5, 0.9]), _summary("CD-NK", [0.5, 0.9])]
    )
    reported = pd.DataFrame(columns=["location", "target_end_date", "reported"])
    (out,) = _reported_target(
        samples,
        summary,
        "m",
        "2026-10-17",
        ["2026-10-24"],
        daily,
        reported,
        np.random.default_rng(1),
    )
    wide = out.pivot_table(index="sample", columns="location", values="value")
    np.testing.assert_array_equal(wide["CD"], wide["CD-IT"] + wide["CD-NK"])


def test_randomwalk_is_returned_as_it_is(daily: pd.DataFrame) -> None:
    samples = pd.DataFrame(
        {
            "location": "CD-IT",
            "target_end_date": "2026-10-17",
            "sample": [1, 2],
            "value": [3.0, 7.0],
        }
    )
    (out,) = _reported_target(
        samples,
        None,
        "randomwalk",
        "2026-10-17",
        ["2026-10-24"],
        daily,
        samples.iloc[:0],
        np.random.default_rng(1),
    )
    assert out["value"].tolist() == [3.0, 7.0]
