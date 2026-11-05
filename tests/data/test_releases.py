from __future__ import annotations

import datetime as dt

import pandas as pd
import pytest

from bdbvhubrenewal.data.releases import (
    ORIGINAL_AS_OF,
    extraction_date,
    last_report,
    release_id,
)

REWRITTEN = {
    "2026-08-19": "2026-08-18",
    "2026-10-10": "2026-10-09",
    "2026-10-18": "2026-10-17",
    "2026-10-25": "2026-10-24",
}
"""Each original label as the hub would rewrite it (hub issue #90)."""


@pytest.mark.parametrize(
    ("as_of", "extracted", "last"),
    [
        ("2026-10-24", "2026-10-25", "2026-10-24"),
        ("2026-10-25", "2026-10-25", "2026-10-24"),
        ("2026-08-18", "2026-08-19", "2026-08-18"),
        ("2026-11-01", "2026-11-01", "2026-10-31"),
    ],
)
def test_original_rewritten_and_new_labels_give_the_same_dates(
    as_of, extracted, last
):
    assert extraction_date(as_of) == dt.date.fromisoformat(extracted)
    assert last_report(as_of) == dt.date.fromisoformat(last)


def test_simulated_labels_are_their_last_report_day():
    assert last_report("2026-11-01", "simulated") == dt.date(2026, 11, 1)


def test_a_rewritten_time_series_maps_to_the_same_releases(target):
    labels = sorted(target.loc[target["kind"] == "real", "vintage"].unique())
    assert set(labels) >= ORIGINAL_AS_OF
    rewritten = [
        {v: k for k, v in REWRITTEN.items()}.get(label, label)
        for label in labels
    ]
    assert rewritten != labels
    assert [release_id(r) for r in rewritten] == labels
    assert [last_report(r) for r in rewritten] == [
        last_report(label) for label in labels
    ]


def test_last_report_is_on_or_after_every_onset_in_the_release(daily):
    latest = daily.groupby("vintage")["onset_date"].max()
    for label, onset in latest.items():
        assert last_report(label) >= pd.Timestamp(onset).date()
