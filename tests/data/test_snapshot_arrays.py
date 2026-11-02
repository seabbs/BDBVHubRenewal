"""Release snapshots on the onset clock."""

from __future__ import annotations

import datetime as dt
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from bdbvhubrenewal.data.snapshots import (
    parent_code,
    release_cutoff,
    snapshot_arrays,
    weekly_sums,
)

VINTAGES = ("2026-08-18", "2026-10-09", "2026-10-17", "2026-10-24")


DATA = Path(__file__).resolve().parents[2] / "data"


@pytest.fixture(scope="module")
def daily() -> pd.DataFrame:
    return pd.read_csv(DATA / "daily.csv")


@pytest.fixture(scope="module")
def snap(daily: pd.DataFrame):
    return snapshot_arrays(daily, "2026-10-24", ("CD-IT", "CD-NK"))


def test_cutoff_is_the_latest_onset_day_over_all_locations(
    daily: pd.DataFrame,
) -> None:
    cuts = [release_cutoff(daily, v).isoformat() for v in VINTAGES]
    assert cuts == ["2026-08-18", "2026-10-07", "2026-10-16", "2026-10-24"]


def test_stated_cutoff_comes_first() -> None:
    rows = pd.DataFrame(
        {
            "vintage": ["2026-10-24", "2026-10-24"],
            "onset_date": ["2026-10-20", "2026-10-22"],
            "cutoff": ["2026-10-23", "2026-10-23"],
        }
    )
    assert release_cutoff(rows, "2026-10-24") == dt.date(2026, 10, 23)


def test_ages_count_from_the_cutoff_not_the_release_date(snap) -> None:
    days = snap.days
    second = snap.releases.index("2026-10-09")
    assert days[snap.cut[second]].date() == dt.date(2026, 10, 7)
    assert days[snap.asof[second]].date() == dt.date(2026, 10, 9)
    # Age 0 at the second release is 10-07, not 10-09.
    assert snap.cut[second] - np.flatnonzero(days == "2026-10-07")[0] == 0


def test_cells_after_the_cutoff_are_unobserved(snap) -> None:
    for r in range(len(snap.releases)):
        assert snap.observed[r].sum() == snap.cut[r] + 1
        assert (snap.counts[:, r][:, ~snap.observed[r]] == 0).all()


def test_one_cutoff_for_every_location(daily: pd.DataFrame) -> None:
    quiet = daily.loc[
        ~((daily["location"] == "CD-HU") & (daily["vintage"] == "2026-10-24"))
    ]
    a = snapshot_arrays(quiet, "2026-10-24", ("CD-HU",))
    b = snapshot_arrays(daily, "2026-10-24", ("CD-HU",))
    assert (a.cut == b.cut).all()
    assert a.observed.shape == (4, len(a.days))


def test_national_is_the_sum_of_provinces(daily: pd.DataFrame) -> None:
    provinces = tuple(sorted(set(daily["location"]) - {"CD"}))
    snap = snapshot_arrays(daily, "2026-10-24", provinces)
    kids = snap.parent == 0
    assert np.allclose(snap.counts[0], snap.counts[kids].sum(axis=0))


def test_hierarchy_and_node_order(daily: pd.DataFrame) -> None:
    snap = snapshot_arrays(daily, "2026-10-24", ("CD-IT", "CD-IT-01"))
    assert snap.locations == ("CD", "CD-IT", "CD-IT-01")
    assert snap.parent.tolist() == [-1, 0, 1]
    assert parent_code("CD-IT-01") == "CD-IT"
    assert parent_code("CD") is None


def test_releases_after_the_vintage_are_not_used(daily: pd.DataFrame) -> None:
    snap = snapshot_arrays(daily, "2026-10-17", ("CD-IT",))
    assert snap.releases == VINTAGES[:3]
    with pytest.raises(ValueError, match="No daily releases"):
        snapshot_arrays(daily, "2026-01-01", ("CD-IT",))


def test_weeks_run_sunday_to_saturday(snap) -> None:
    ends = pd.to_datetime(list(snap.week_ends))
    assert (ends.day_of_week == 5).all()
    assert snap.week_ends[-1] == "2026-10-24"
    last = snap.releases.index("2026-10-24")
    weekly = weekly_sums(snap, last)
    assert weekly.shape == (len(snap.locations), snap.n_weeks)
    release = pd.read_csv(DATA / "target.csv")
    rows = release.loc[
        (release["vintage"] == "2026-10-24")
        & (release["location"] == "CD")
        & (release["target_end_date"] == "2026-10-17")
    ]
    assert (
        weekly[0, snap.week_ends.index("2026-10-17")] == rows["total"].iloc[0]
    )
