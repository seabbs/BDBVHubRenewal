"""Each model is mapped to what later releases show by its own delay."""

from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd
import pytest

from bdbvhubrenewal.evaluation.nowcast import (
    delay_curves,
    reported_samples,
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
    # A place with no delay of its own, such as a province never seeded,
    # takes the shared delay, as in the renewal model. The country does
    # not: it is the sum of its provinces.
    shared = pd.concat([summary, _summary("shared", [0.5])])
    got = delay_curves(shared, ["CD", "CD-IT", "CD-SU", "CD-SU-01"])
    got = got.groupby("location")["share"].first()
    assert got.to_dict() == {"CD-IT": 0.2, "CD-SU": 0.5, "CD-SU-01": 0.5}


def _samples(rows: dict[str, float], n: int = 50) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "location": np.repeat(list(rows), n),
            "target_end_date": "2026-10-17",
            "sample": np.tile(np.arange(1, n + 1), len(rows)),
            "value": np.repeat(list(rows.values()), n),
        }
    )


def _reported(samples, summary, model, daily):  # type: ignore[no-untyped-def]
    (out,) = reported_samples(
        samples,
        summary,
        model,
        "2026-10-17",
        ["2026-10-24"],
        daily,
        pd.DataFrame(columns=["location", "target_end_date", "reported"]),
        np.random.default_rng(1),
    ).values()
    return out.pivot_table(index="sample", columns="location", values="value")


def test_a_place_with_no_delay_is_kept_and_the_country_sums_them(
    daily: pd.DataFrame,
) -> None:
    samples = _samples({"CD-IT": 100.0, "CD-NK": 50.0, "CD": 999.0})
    # The summary has no delay for CD-NK or the country.
    wide = _reported(samples, _summary("CD-IT", [0.5, 0.9]), "m", daily)
    assert (wide["CD-IT"] < 100).any()
    assert (wide["CD-NK"] == 50).all()
    np.testing.assert_array_equal(wide["CD"], wide["CD-IT"] + wide["CD-NK"])


def test_the_country_sums_its_provinces_beside_a_shared_delay(
    daily: pd.DataFrame,
) -> None:
    samples = _samples({"CD-IT": 100.0, "CD-NK": 50.0, "CD": 999.0})
    summary = pd.concat(
        [
            _summary("CD-IT", [0.5, 0.9]),
            _summary("CD-NK", [0.3, 0.6]),
            _summary("shared", [0.01, 0.02]),
        ]
    )
    wide = _reported(samples, summary, "m", daily)
    assert (wide["CD-NK"] < 50).any()
    np.testing.assert_array_equal(wide["CD"], wide["CD-IT"] + wide["CD-NK"])


@pytest.mark.parametrize("model", ["randomwalk", "fm_chronos_weekly_complete"])
def test_walk_models_are_returned_as_they_are_and_a_hub_team_not_at_all(
    model: str, daily: pd.DataFrame
) -> None:
    samples = _samples({"CD-IT": 3.0})
    wide = _reported(samples, None, model, daily)
    assert (wide["CD-IT"] == 3).all()
    none = reported_samples(
        samples, None, "team", "2026-10-17", ["2026-10-24"], daily,
        samples.iloc[:0], np.random.default_rng(1),
    )  # fmt: skip
    assert none == {}


def test_samples_are_kept_when_the_delay_maps_no_place(
    daily: pd.DataFrame,
) -> None:
    samples = _samples({"CD-IT": 100.0, "CD-NK": 50.0, "CD": 999.0})
    # Only a delay for a place not forecast, which maps no place.
    wide = _reported(samples, _summary("CD-SU", [0.5, 0.9]), "m", daily)
    assert (wide["CD-IT"] == 100).all()
    assert (wide["CD-NK"] == 50).all()
    assert (wide["CD"] == 150).all()


def test_a_pooled_delay_maps_every_place_but_the_country() -> None:
    # Fits from before province delays store one delay for every place,
    # labelled "pooled". It maps each place without its own, not the
    # country, which is the sum of its provinces.
    summary = pd.concat([_summary("pooled", [0.4]), _summary("CD-NK", [0.8])])
    curves = delay_curves(summary, ["CD", "CD-IT", "CD-NK", "CD-IT-01"])
    got = curves.groupby("location")["share"].first()
    assert got.to_dict() == {"CD-IT": 0.4, "CD-IT-01": 0.4, "CD-NK": 0.8}
