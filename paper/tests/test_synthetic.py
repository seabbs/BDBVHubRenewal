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
from paper import synthetic

COLUMNS = [
    "study",
    "model",
    "location",
    "origin_date",
    "target_date",
    "horizon",
    "quantile_level",
    "sample_id",
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
    weeks = synthetic.longest_delay_weeks(counts) + synthetic.HORIZON_WEEKS
    assert end - last >= dt.timedelta(weeks=weeks)
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


def test_evaluate_gives_one_long_sample_table_with_truth() -> None:
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
    # Sample forecasts, as `paper/schema.md` has the synthetic study.
    assert out["quantile_level"].isna().all()
    rows = out.groupby(["location", "origin_date", "target_date"])
    assert (rows["sample_id"].nunique() == 20).all()
    assert (rows.size() == 20).all()
    assert out["truth"].notna().all()
    # National is the sum of the provinces' samples.
    v = out.set_index(["origin_date", "target_date", "sample_id", "location"])[
        "value"
    ]
    nat = v.xs(synthetic.series_label("9-0001", NATIONAL), level="location")
    p1 = v.xs(synthetic.series_label("9-0001", "P01"), level="location")
    p2 = v.xs(synthetic.series_label("9-0001", "P02"), level="location")
    np.testing.assert_allclose(nat, p1 + p2)


def test_models_by_name_skip_unregistered_ones() -> None:
    models, skipped = synthetic.models_by_name(
        ["randomwalknowcast", "no_such_model"]
    )
    assert [m.name for m in models] == ["randomwalknowcast"]
    assert skipped == ["no_such_model"]


@pytest.mark.parametrize(
    "name", ["fm_chronos_weekly_ragged", "fm_chronos_daily_ragged"]
)
def test_default_models_include_the_zero_shot_ragged_ones(name: str) -> None:
    assert name in synthetic.MODELS
    assert "randomwalknowcast" in synthetic.MODELS
    assert "randomwalk" in synthetic.MODELS
    assert name.replace("ragged", "complete") in synthetic.MODELS
    assert f"{name}_ft" in synthetic.MODELS


def test_the_random_walk_forecasts_a_single_place_as_the_country() -> None:
    task = synthetic.outbreak_tasks(_counts(("P01",)), "9-0001", 1, 10)[0]
    walk = synthetic.model_task(get_model("randomwalknowcast"), task)
    chronos = synthetic.model_task(_Constant("flat"), task)
    assert walk.locations == (NATIONAL,)
    assert chronos.locations == ()


def test_observations_are_every_weekly_release_then_final() -> None:
    counts = _counts(("P01", "P02"))
    cutoffs, _ = synthetic.origin_cutoffs(counts, "9-0001", 2)
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
    # Weekly releases on the origins' weekday run to the last report, so
    # every target has a release D days after it for the rolling truth.
    releases = synthetic.release_cutoffs(counts, "9-0001")
    assert releases[: len(cutoffs)] == cutoffs
    assert dt.date.fromisoformat(end) - dt.date.fromisoformat(
        releases[-1]
    ) < dt.timedelta(weeks=1)
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


def test_factors_give_each_series_its_outbreak_settings() -> None:
    counts = _counts(("P01", "P02"))
    settings = pd.DataFrame(
        {
            "outbreak": ["9-0001", "9-0002"],
            "delay_mean": [5.0, 9.0],
            "n_places": [2, 1],
            "gt_mean": [10.0, 12.0],
            "unused": [0, 0],
        }
    )
    out = synthetic.factor_table(counts, settings)
    assert list(out.columns) == [
        "location",
        *[f for f in synthetic.FACTORS if f in settings],
        "delay_q98",
    ]
    assert set(out["location"]) == {
        synthetic.series_label("9-0001", p) for p in (NATIONAL, "P01", "P02")
    }
    assert (out["delay_mean"] == 5.0).all()
    assert (out["n_places"] == 2).all()


def test_delay_q98_is_the_outbreaks_98th_delay_percentile_in_days() -> None:
    # Delays of 0, 7 and 20 days hold 3, 2 and 1 of each 6 cases, so the
    # 98th percentile is 20 days.
    counts = _counts(("P01", "P02"))
    assert synthetic.delay_quantile(counts, 0.98) == 20
    assert synthetic.delay_quantile(counts, 0.5) == 0
    assert synthetic.delay_quantile(counts, 0.6) == 7
    settings = pd.DataFrame({"outbreak": ["9-0001"], "gt_mean": [10.0]})
    out = synthetic.factor_table(counts, settings)
    assert (out["delay_q98"] == 20).all()


def test_observations_have_zero_weeks_so_every_target_has_truth() -> None:
    # P02 has no cases in its first ten weeks.
    counts = _counts(("P01", "P02"))
    late = pd.to_datetime(counts["reference_date"]) < pd.Timestamp("2026-03-16")
    counts = counts.loc[~(late & (counts["location"] == "P02"))]
    obs = synthetic.observation_table(counts, "9-0001", 2)
    p02 = obs.loc[obs["location"] == synthetic.series_label("9-0001", "P02")]
    first = obs["target_date"].min()
    for _, rows in p02.groupby("as_of_date"):
        weeks = pd.date_range(first, rows["target_date"].max(), freq="7D")
        assert list(rows["target_date"]) == list(weeks.strftime("%Y-%m-%d"))
    early = p02.loc[p02["target_date"] < "2026-03-14", "value"]
    assert len(early) > 0
    assert (early == 0).all()


def test_a_longer_delay_moves_the_last_origin_back() -> None:
    # The r2-real grid has 12-week delays in some outbreaks, so the last
    # origin must leave its forecast weeks that long to complete.
    counts = _counts(("P01",), weeks=40)
    assert synthetic.longest_delay_weeks(counts) == 8
    late = counts.iloc[:1].assign(
        report_date=(
            pd.Timestamp(counts["reference_date"].iloc[0])
            + pd.Timedelta(days=80)
        ).strftime("%Y-%m-%d")
    )
    longer = pd.concat([counts, late], ignore_index=True)
    assert synthetic.longest_delay_weeks(longer) == 12
    _, short = synthetic.origin_cutoffs(counts, "9-0001", 4)
    _, long = synthetic.origin_cutoffs(longer, "9-0001", 4)
    end = dt.date.fromisoformat(str(longer["report_date"].max()))
    assert end - dt.date.fromisoformat(long[-1]) >= dt.timedelta(
        weeks=12 + synthetic.HORIZON_WEEKS
    )
    assert long[-1] < short[-1]
