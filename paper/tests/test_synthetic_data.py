"""Tests of the held-out simulation loader."""

import pandas as pd
import pytest

from paperdata import synthetic
from paperpipe.config import seed_of


def _counts(rows) -> pd.DataFrame:
    return pd.DataFrame(
        rows,
        columns=[
            "outbreak",
            "location",
            "reference_date",
            "report_date",
            "count",
        ],
    ).assign(timestep="day")


@pytest.fixture
def sims(tmp_path):
    # Outbreak 9001-0001-1 in one place: onsets in the week ending
    # Saturday 2026-01-10, reported over the next fortnight.
    counts = _counts(
        [
            ("9001-0001-1", "P01", "2026-01-05", "2026-01-06", 4),
            ("9001-0001-1", "P01", "2026-01-05", "2026-01-13", 2),
            ("9001-0001-1", "P01", "2026-01-08", "2026-01-20", 4),
            ("9001-0001-1", "P02", "2026-01-12", "2026-01-12", 5),
        ]
    )
    counts.to_csv(tmp_path / "counts-9001.csv.gz", index=False)
    pd.DataFrame(
        {"outbreak": ["9001-0001"], "process": ["growth"], "strata": ["places"]}
    ).to_csv(tmp_path / "settings-9001.csv", index=False)
    return tmp_path


def test_observations_are_weekly_onsets_as_of_each_saturday(sims) -> None:
    obs, _ = synthetic.load(sims, [9001])
    assert set(obs["study"]) == {"synthetic"}
    assert (pd.to_datetime(obs["target_date"]).dt.day_of_week == 5).all()
    assert {seed_of(loc) for loc in obs["location"]} == {9001}
    p01 = obs.loc[obs["location"].str.endswith("P01")].set_index("as_of_date")
    week = p01.loc[p01["target_date"] == pd.Timestamp("2026-01-10"), "value"]
    # As of 2026-01-10 only the first report; by 2026-01-24 all ten.
    assert week.loc[pd.Timestamp("2026-01-10")] == 4
    assert week.loc[pd.Timestamp("2026-01-24")] == 10


def test_each_place_and_the_total_is_a_location(sims) -> None:
    obs, locations = synthetic.load(sims, [9001])
    assert set(locations["location"]) == set(obs["location"])
    assert any(loc.endswith("-total") for loc in locations["location"])


def test_locations_carry_the_recorded_cell_and_the_98pct_delay(sims) -> None:
    _, locations = synthetic.load(sims, [9001])
    assert set(locations["process"]) == {"growth"}
    assert set(locations["strata"]) == {"places"}
    # Delays 1, 8, 12 and 0 days weighted by count: the 98% point is 12.
    assert set(locations["delay_days"]) == {12}


def test_missing_names_the_seeds_without_simulations(sims) -> None:
    assert synthetic.missing(sims, [9001]) is None
    assert "9002" in synthetic.missing(sims, [9001, 9002])
