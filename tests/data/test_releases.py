from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd
import pytest

from bdbvhubrenewal.data.releases import (
    ORIGINAL_AS_OF,
    extraction_date,
    last_report,
    release_id,
    release_pairs,
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


def _rows(vintage, cells):
    """Delay rows of one release from (onset week, delay, count) cells."""
    return pd.DataFrame(
        [
            {
                "vintage": vintage,
                "location": "CD-IT",
                "onset_week": week,
                "delay": delay,
            }
            for week, delay, n in cells
            for _ in range(n)
        ]
    )


def test_release_pairs_count_rows_kept_from_the_earlier_release():
    delays = pd.concat(
        [
            _rows("2026-11-01", [("2026-10-24", 2, 3), ("2026-10-17", 9, 4)]),
            _rows(
                "2026-11-08",
                [
                    ("2026-10-24", 2, 5),
                    ("2026-10-17", 9, 4),
                    ("2026-10-31", 1, 2),
                ],
            ),
        ]
    )
    pairs = release_pairs(delays, "2026-11-08")
    assert pairs is not None
    order = np.argsort(pairs["total"])
    # Some onset days of the new week could be reported by the earlier
    # cut-off, so its rows stay, with none kept.
    np.testing.assert_array_equal(pairs["kept"][order], [0, 4, 3])
    np.testing.assert_array_equal(pairs["total"][order], [2, 4, 5])
    # Report age at each release for each onset day of the week: the
    # 10-24 week with delay 2 at the 10-31 cut-off is 5 to 11 days old.
    row = order[2]
    np.testing.assert_array_equal(pairs["before"][row], np.arange(5, 12))
    np.testing.assert_array_equal(pairs["after"][row] - pairs["before"][row], 7)
    assert pairs["pair"].shape == (3, 1)


def test_release_pairs_use_only_releases_up_to_the_forecast_date():
    delays = pd.concat(
        [
            _rows("2026-10-24", [("2026-10-17", 2, 3)]),
            _rows("2026-11-01", [("2026-10-17", 2, 4)]),
        ]
    )
    assert release_pairs(delays, "2026-10-24") is None
    assert release_pairs(delays, "2026-11-01")["pair"].shape[1] == 1
