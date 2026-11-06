from __future__ import annotations

import datetime as dt

import pandas as pd
import pytest

from bdbvhubrenewal.data.releases import (
    ORIGINAL_AS_OF,
    last_report_days,
    release_cutoff,
    release_id,
)

REWRITTEN = {
    "2026-08-19": "2026-08-18",
    "2026-10-10": "2026-10-09",
    "2026-10-18": "2026-10-17",
    "2026-10-25": "2026-10-24",
}
"""Each published label as the hub would rewrite it, a day later."""


def test_a_rewritten_time_series_maps_to_the_same_releases(target):
    labels = sorted(target.loc[target["kind"] == "real", "vintage"].unique())
    assert set(labels) >= ORIGINAL_AS_OF
    rewritten = [
        {v: k for k, v in REWRITTEN.items()}.get(label, label)
        for label in labels
    ]
    assert rewritten != labels
    assert [release_id(r) for r in rewritten] == labels


def test_the_last_report_day_is_the_latest_report_the_data_prove():
    daily = pd.DataFrame(
        {
            "vintage": ["2026-10-09", "2026-10-09", "2026-10-17"],
            "onset_date": ["2026-10-05", "2026-10-07", "2026-10-16"],
        }
    )
    # Onset week ending 10-10 starts 10-04, so a delay of 4 days proves a
    # report on 10-08. At 10-17 the latest onset is later than any report
    # the delay rows prove.
    delays = pd.DataFrame(
        {
            "vintage": ["2026-10-09", "2026-10-09", "2026-10-17"],
            "onset_week": ["2026-10-10", "2026-10-03", "2026-10-10"],
            "delay": [4, 6, 5],
        }
    )
    cut = last_report_days(daily, delays)
    assert cut.to_dict() == {
        "2026-10-09": "2026-10-08",
        "2026-10-17": "2026-10-16",
    }


def test_a_release_without_delay_rows_ends_at_its_latest_onset():
    daily = pd.DataFrame(
        {"vintage": ["2026-08-18"], "onset_date": ["2026-08-17"]}
    )
    delays = pd.DataFrame(columns=["vintage", "onset_week", "delay"])
    assert last_report_days(daily, delays).to_dict() == {
        "2026-08-18": "2026-08-17"
    }


@pytest.mark.parametrize(
    ("label", "cut"),
    [
        ("2026-08-18", "2026-08-18"),
        ("2026-10-09", "2026-10-08"),
        ("2026-10-17", "2026-10-16"),
        ("2026-10-24", "2026-10-24"),
        ("2026-11-01", "2026-10-31"),
    ],
)
def test_every_table_carries_the_same_cut_off(
    target, daily, delays, daily_zones, label, cut
):
    day = dt.date.fromisoformat(cut)
    for table in (target, daily, daily_zones):
        assert release_cutoff(table, label) == day
    if label in set(delays["vintage"]):
        assert release_cutoff(delays, label) == day


def test_the_cut_off_is_on_or_after_every_onset_in_the_release(daily):
    latest = daily.groupby("vintage")["onset_date"].max()
    for label, onset in latest.items():
        assert release_cutoff(daily, label) >= pd.Timestamp(onset).date()
