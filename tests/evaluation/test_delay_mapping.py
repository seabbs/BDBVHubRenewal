"""Each model is mapped to what later releases show by its own delay."""

from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd
import pytest

from bdbvhubrenewal.evaluation.nowcast import delay_curves, week_shares


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


def _samples(rows: dict[str, float]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "location": list(rows),
            "target_end_date": "2026-10-24",
            "sample": 1,
            "value": list(rows.values()),
        }
    )


def test_delay_curves_keep_the_reported_shares_by_age() -> None:
    summary = pd.concat(
        [
            _summary("pooled", [0.2, 0.6, 1.0]),
            _summary("pooled", [9.0]).assign(quantity="median_delay"),
        ]
    )
    curves = delay_curves(summary)
    assert list(curves.columns) == ["location", "age", "share"]
    assert curves["share"].tolist() == [0.2, 0.6, 1.0]


def test_a_week_s_share_is_the_mean_over_its_days() -> None:
    curves = delay_curves(_summary("CD-IT", [0.0, 0.5, 1.0]))
    # The week ending Saturday 10-24 seen at that day: ages 6 to 0.
    got = week_shares(curves, dt.date(2026, 10, 24), _samples({"CD-IT": 10}))
    assert got["share"].iloc[0] == pytest.approx((0 + 0.5 + 5 * 1.0) / 7)
    # Days after the cut-off are not reported.
    early = week_shares(curves, dt.date(2026, 10, 20), _samples({"CD-IT": 10}))
    assert early["share"].iloc[0] == pytest.approx((1 + 0.5 + 0) / 7)


def test_places_fall_back_to_pooled_then_parent_then_a_weighted_mean() -> None:
    cut = dt.date(2026, 11, 7)
    pooled = delay_curves(_summary("pooled", [0.5]))
    got = week_shares(pooled, cut, _samples({"CD": 30, "CD-IT-01": 5}))
    assert (got["share"] == 0.5).all()
    by_province = delay_curves(
        pd.concat([_summary("CD-IT", [0.2]), _summary("CD-NK", [0.8])])
    )
    got = week_shares(
        by_province,
        cut,
        _samples({"CD": 40, "CD-IT": 10, "CD-NK": 30, "CD-IT-01": 4}),
    ).set_index("location")["share"]
    assert got["CD-IT-01"] == pytest.approx(0.2)
    # The country takes its provinces' shares weighted by the forecast.
    assert got["CD"] == pytest.approx((10 * 0.2 + 30 * 0.8) / 40)
