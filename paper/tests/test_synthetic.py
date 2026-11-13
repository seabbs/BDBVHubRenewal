"""Tests of the held-out synthetic evaluation (`paper.synthetic`)."""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass

import numpy as np
import pandas as pd
import pytest

from bdbvhubrenewal.forecasters.base import (
    NATIONAL,
    ModelUnavailableError,
    get_model,
)
from bdbvhubrenewal.reporting.hubtables import HUB_QUANTILE_LEVELS
from paper import synthetic

COLUMNS = [
    "study",
    "model",
    "location",
    "origin_date",
    "target_date",
    "horizon",
    "quantile_level",
    "value",
    "truth",
]


def _counts(places: tuple[str, ...], weeks: int = 30) -> pd.DataFrame:
    """One outbreak's triangle: each place has 6 onsets a day, reported
    3 on the day, 2 a week later and 1 after 20 days."""
    first = dt.date(2026, 1, 5)
    rows = []
    for place in places:
        for i in range(7 * weeks):
            onset = first + dt.timedelta(days=i)
            for delay, n in ((0, 3), (7, 2), (20, 1)):
                rows.append(
                    {
                        "outbreak": "9-0001",
                        "timestep": "day",
                        "location": place,
                        "reference_date": onset.isoformat(),
                        "report_date": (
                            onset + dt.timedelta(days=delay)
                        ).isoformat(),
                        "count": n,
                    }
                )
    # The simulations run twelve weeks past the onsets used.
    out = pd.DataFrame(rows)
    keep = pd.to_datetime(out["report_date"]) <= pd.Timestamp(
        first + dt.timedelta(weeks=weeks)
    )
    return out.loc[keep].reset_index(drop=True)


def test_origins_are_spread_over_complete_cutoffs_on_one_weekday() -> None:
    counts = _counts(("P01", "P02"))
    cutoffs, origins = synthetic.origin_cutoffs(counts, "9-0001", 4)
    days = {dt.date.fromisoformat(c).weekday() for c in cutoffs}
    assert len(days) == 1
    assert days.pop() in {2, 3, 4, 5}
    assert len(origins) == 4
    assert origins == sorted(origins)
    assert set(origins) <= set(cutoffs)
    # At least six releases before the first origin, for D plus context.
    assert cutoffs.index(origins[0]) >= synthetic.MIN_RELEASES - 1
    # The last origin's horizon is complete by the end of the reports.
    end = pd.Timestamp(counts["report_date"].max()).date()
    last = dt.date.fromisoformat(origins[-1])
    assert end - last >= dt.timedelta(weeks=synthetic.COMPLETE_WEEKS)
    # Fixed by the outbreak.
    assert synthetic.origin_cutoffs(counts, "9-0001", 4) == (cutoffs, origins)


def test_tasks_hold_the_releases_up_to_their_origin() -> None:
    counts = _counts(("P01", "P02"))
    cutoffs, origins = synthetic.origin_cutoffs(counts, "9-0001", 2)
    tasks = synthetic.outbreak_tasks(counts, "9-0001", 2, n_samples=10)
    assert [t.vintage for t in tasks] == origins
    for task in tasks:
        ref = dt.date.fromisoformat(task.cutoff)
        ref += dt.timedelta(days=(5 - ref.weekday()) % 7)
        assert task.reference_date == ref
        assert task.locations == ("P01", "P02")
        assert task.target_end_dates == tuple(
            (ref + dt.timedelta(weeks=h)).isoformat()
            for h in synthetic.HORIZONS
        )
        assert task.daily_history is not None
        assert task.target_history is not None
        assert task.daily_history["vintage"].max() == task.vintage
        assert task.target_history["vintage"].max() == task.vintage
        assert set(task.target["vintage"]) == {task.vintage}
        assert task.n_samples == 10
        # Weekly releases on the outbreak's weekday only.
        assert set(task.daily_history["vintage"]) <= set(cutoffs)


def test_a_single_place_is_the_country() -> None:
    counts = _counts(("P01",))
    task = synthetic.outbreak_tasks(counts, "9-0001", 1, n_samples=10)[0]
    assert task.locations == ()
    assert set(task.target["location"]) == {NATIONAL}


def test_truth_is_the_final_weekly_count() -> None:
    counts = _counts(("P01", "P02"), weeks=20)
    truth = synthetic.truth_table(counts)
    assert set(truth["location"]) == {NATIONAL, "P01", "P02"}
    full = truth.set_index(["location", "target_end_date"])["truth"]
    # A full week in the middle has 7 days of 6 onsets.
    assert full[("P01", "2026-03-07")] == 42
    assert full[(NATIONAL, "2026-03-07")] == 84


@dataclass
class _Constant:
    name: str
    value: float = 3.0

    def forecast(self, task):
        rows = [
            (loc, week, s, self.value + s)
            for loc in task.locations
            for week in task.target_end_dates
            for s in range(1, task.n_samples + 1)
        ]
        return pd.DataFrame(
            rows, columns=["location", "target_end_date", "sample", "value"]
        )


@dataclass
class _Missing:
    name: str = "missing"

    def forecast(self, task):  # noqa: ARG002
        raise FileNotFoundError("no checkpoint")


@dataclass
class _Unavailable:
    name: str = "unavailable"

    def forecast(self, task):  # noqa: ARG002
        raise ModelUnavailableError("no data")


def test_evaluate_gives_one_long_quantile_table_with_truth() -> None:
    counts = _counts(("P01", "P02"))
    out = synthetic.evaluate(
        counts,
        [_Constant("flat"), _Missing(), _Unavailable()],
        n_origins=2,
        n_samples=20,
    )
    assert list(out.columns) == COLUMNS
    assert set(out["study"]) == {"synthetic"}
    assert set(out["model"]) == {"flat"}
    assert set(out["location"]) == {
        synthetic.series_label("9-0001", p) for p in (NATIONAL, "P01", "P02")
    }
    assert out["origin_date"].nunique() == 2
    # Horizons are in days from the origin, as `paper/schema.md` has it.
    days = (
        pd.to_datetime(out["target_date"]) - pd.to_datetime(out["origin_date"])
    ).dt.days
    assert (out["horizon"] == days).all()
    span = out["horizon"].max() - out["horizon"].min()
    assert span == 7 * (len(synthetic.HORIZONS) - 1)
    np.testing.assert_allclose(
        sorted(out["quantile_level"].unique()), HUB_QUANTILE_LEVELS
    )
    rows = out.groupby(["location", "origin_date", "target_date"]).size()
    assert (rows == len(HUB_QUANTILE_LEVELS)).all()
    assert out["truth"].notna().all()
    # National is the sum of the provinces' samples.
    q = out.loc[out["quantile_level"] == 0.5].set_index(
        ["origin_date", "target_date", "location"]
    )["value"]
    nat = q.xs(synthetic.series_label("9-0001", NATIONAL), level="location")
    assert (nat > q.xs(synthetic.series_label("9-0001", "P01"), level=2)).all()


def test_models_by_name_skip_unregistered_ones() -> None:
    models, skipped = synthetic.models_by_name(
        ["randomwalknowcast", "no_such_model"]
    )
    assert [m.name for m in models] == ["randomwalknowcast"]
    assert skipped == ["no_such_model"]


@pytest.mark.parametrize("name", ["fm_chronos_weekly_ragged"])
def test_default_models_include_the_zero_shot_ragged_ones(name: str) -> None:
    assert name in synthetic.MODELS
    assert "randomwalknowcast" in synthetic.MODELS
    assert f"{name}_ft" in synthetic.MODELS


def test_the_random_walk_forecasts_a_single_place_as_the_country() -> None:
    task = synthetic.outbreak_tasks(_counts(("P01",)), "9-0001", 1, 10)[0]
    walk = synthetic.model_task(get_model("randomwalknowcast"), task)
    chronos = synthetic.model_task(_Constant("flat"), task)
    assert walk.locations == (NATIONAL,)
    assert chronos.locations == ()


def test_observations_are_each_release_to_the_last_origin_then_final() -> None:
    counts = _counts(("P01", "P02"))
    cutoffs, origins = synthetic.origin_cutoffs(counts, "9-0001", 2)
    obs = synthetic.observation_table(counts, "9-0001", 2)
    assert list(obs.columns) == [
        "study",
        "location",
        "target_date",
        "as_of_date",
        "value",
    ]
    assert set(obs["study"]) == {"synthetic"}
    end = str(counts["report_date"].max())
    releases = cutoffs[: cutoffs.index(origins[-1]) + 1]
    assert set(obs["as_of_date"]) == {*releases, end}
    # The latest as-of date is the final data, the truth.
    final = obs.loc[obs["as_of_date"] == end].set_index(
        ["location", "target_date"]
    )["value"]
    truth = synthetic.truth_table(counts)
    truth["location"] = [
        synthetic.series_label("9-0001", p) for p in truth["location"]
    ]
    truth = truth.set_index(["location", "target_end_date"])["truth"]
    np.testing.assert_array_equal(final.sort_index(), truth.sort_index())
