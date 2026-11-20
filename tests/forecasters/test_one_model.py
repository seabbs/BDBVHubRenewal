"""Fine-tuned Chronos-2 on daily releases: daily rows plus weekly totals.

The weekly and daily fine-tuned models share one adapter. The weekly one
is a weekly-steps block of the weekly releases, as the zero-shot weekly
model. The daily one adds, per place, a row of its weekly totals summed
from its own daily counts, and reads the weekly forecast from that row;
the two sources are never mixed.
"""

import dataclasses
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from bdbvhubrenewal.forecasters import ForecastTask, get_model
from nowcasteval.backtest import BacktestConfig, make_tasks
from nowcastfoundation import forecaster as foundation
from nowcastfoundation import series_bias
from nowcastfoundation.backends import chronos2
from nowcastfoundation.forecaster import (
    FoundationModel,
    HubTriangle,
    layout_block,
    week_rows,
)

# Two Saturdays, 2026-10-03 and 2026-10-10, in a fortnight of days.
DAYS = [
    str(pd.Timestamp("2026-09-27") + pd.Timedelta(days=i))[:10]
    for i in range(14)
]
PLACES = ["CD", "CD-NK"]


def daily_tri(lags: int = 3) -> HubTriangle:
    final = np.arange(2 * 14, dtype=float).reshape(2, 14)
    partial = np.full((2, 14, lags), 0.5)
    return HubTriangle(DAYS, final, partial, last=13)


def test_week_rows_sum_each_whole_week_of_daily_counts_on_its_saturday():
    tri = daily_tri()
    rows = week_rows(tri)
    # 2026-09-27 is a Sunday: two whole weeks, ending on these Saturdays.
    sat = [DAYS.index("2026-10-03"), DAYS.index("2026-10-10")]
    assert list(np.flatnonzero(rows.ends)) == sat
    assert rows.totals.shape == (2, 14)
    assert rows.totals[0, sat[0]] == tri.final[0, 0:7].sum()
    assert rows.totals[1, sat[1]] == tri.final[1, 7:14].sum()
    off = np.ones(14, dtype=bool)
    off[sat] = False
    assert np.isnan(rows.totals[:, off]).all()


def test_a_week_cut_by_the_grid_has_no_total():
    tri = daily_tri()
    cut = HubTriangle(tri.times[2:], tri.final[:, 2:], tri.partial[:, 2:], 11)
    rows = week_rows(cut)
    # The first week lost two days, so only the second has a total.
    assert list(np.flatnonzero(rows.ends)) == [cut.times.index("2026-10-10")]


def test_a_period_block_adds_one_total_row_per_place_after_the_daily_rows():
    tri, start = daily_tri(), 10
    rows = week_rows(tri)
    group = layout_block(tri, "ragged", PLACES, start, period=rows)
    target = group["target"]
    assert target.shape == (4, start)
    np.testing.assert_array_equal(target[:2], tri.final[:, :start])
    np.testing.assert_array_equal(target[2:], rows.totals[:, :start])
    # The total rows have no lags; each place's daily lags are as before.
    assert len(group["past_covariates"]) == 2 * 3
    assert group["places"] == PLACES
    assert list(group["target_places"]) == [0, 1, 0, 1]


def test_a_block_without_periods_names_one_row_per_place():
    group = layout_block(daily_tri(), "ragged", PLACES, 10)
    assert group["target"].shape == (2, 10)
    assert list(group["target_places"]) == [0, 1]


def test_a_total_row_joins_its_places_bias_group():
    # Targets: CD, CD-NK, then their totals; then each place's 3 lags.
    rows = series_bias.block_rows([0, 1, 0, 1], 2 * 3, [-1, 0])
    place, parent = rows
    assert list(place) == [0, 1, 0, 1, 0, 0, 0, 1, 1, 1]
    assert list(parent) == [-1, 0, -1, 0, -1, -1, -1, 0, 0, 0]
    torch = pytest.importorskip("torch")

    same, up, _ = series_bias.pair_masks(
        torch.as_tensor(place), torch.as_tensor(parent)
    ).values()
    # CD's total is the same place as CD's counts and lags, CD-NK's total
    # points up to CD.
    assert bool(same[2, 0])
    assert bool(same[2, 4])
    assert bool(up[3, 0])
    assert bool(up[3, 2])


def test_inference_rows_carry_the_total_rows_places():
    tri = daily_tri()
    group = layout_block(tri, "ragged", PLACES, 10, period=week_rows(tri))
    place, parent = chronos2._group_rows(group)
    assert list(place[:4]) == [0, 1, 0, 1]
    assert list(parent[:4]) == [-1, 0, -1, 0]
    # The lags, in Chronos-2's sorted row order, keep their place.
    names = sorted(group["past_covariates"])
    for k, name in enumerate(names):
        assert place[4 + k] == PLACES.index(name.rsplit("_lag_", 1)[0])


def test_the_weekly_and_daily_finetuned_models_share_one_adapter(ft):
    weekly = get_model("fm_chronos_weekly_ragged_ft")
    assert weekly.checkpoint == ft.checkpoint
    assert (weekly.source, weekly.period) == ("weekly", None)
    assert (ft.source, ft.period) == ("daily", "rolling7")
    # The zero-shot daily model has the same rolling rows; complete data
    # alone has none.
    assert get_model("fm_chronos_daily_ragged").period == "rolling7"
    assert get_model("fm_chronos_daily_complete").period is None


def _aggregate_sampler(store: list):
    """Draws 1 on daily rows and 1000 + place on total rows."""

    def sample(groups, horizon, n, _seed):
        out = []
        for g in groups:
            store.append(g)
            v = np.atleast_2d(g["target"]).shape[0]
            draws = np.ones((n, v, horizon))
            if "target_places" in g and v > len(g["places"]):
                for r in range(len(g["places"]), v):
                    draws[:, r, :] = 1000 + g["target_places"][r]
            out.append(draws)
        return out

    return sample


@pytest.fixture(scope="module")
def task(
    target: pd.DataFrame, delays: pd.DataFrame, daily: pd.DataFrame
) -> ForecastTask:
    rows = target.loc[target["vintage"] == "2026-10-24"]
    real = target.loc[target["kind"] == "real"]
    return make_tasks(
        rows, delays, BacktestConfig(seed=3), history=real, daily=daily
    )[0]


@pytest.fixture
def ft(fake_checkpoint):  # noqa: ARG001 (the fake release)
    return get_model("fm_chronos_daily_ragged_ft")


def test_weekly_forecasts_are_read_from_the_total_rows(
    ft: FoundationModel, task: ForecastTask
) -> None:
    store: list = []
    model = dataclasses.replace(ft, sample=_aggregate_sampler(store))
    out = model.forecast(task)
    group = store[-1]
    n_places = len(group["places"])
    assert np.atleast_2d(group["target"]).shape[0] == 2 * n_places
    # Weeks whose Saturday is past the context come from the totals,
    # which are summed from the same daily releases.
    tri = foundation.hub_triangle(
        task.daily_history, "daily", group["places"], task.target_end_dates[-1]
    )
    future = [
        w for w in task.target_end_dates if tri.times.index(w) >= tri.n_context
    ]
    assert future
    for i, place in enumerate(group["places"]):
        rows = out[
            (out["location"] == place) & out["target_end_date"].isin(future)
        ]
        assert (rows["value"] >= 1000 + i).all()


def test_the_weekly_finetuned_model_uses_the_weekly_block_alone(
    task: ForecastTask, fake_checkpoint: str
) -> None:
    store: list = []
    weekly = dataclasses.replace(
        get_model("fm_chronos_weekly_ragged_ft"),
        sample=_aggregate_sampler(store),
    )
    assert weekly.checkpoint.hash == fake_checkpoint
    weekly.forecast(task)
    group = store[-1]
    assert np.atleast_2d(group["target"]).shape[0] == len(group["places"])
    n_lags = len(group["places"]) * weekly.lag_count(task)
    assert len(group["past_covariates"]) == n_lags


def test_one_forecast_gives_daily_draws_from_the_count_rows_too(
    ft: FoundationModel, task: ForecastTask
) -> None:
    store: list = []
    model = dataclasses.replace(ft, sample=_aggregate_sampler(store))
    weekly, daily = model.forecast_both(task)
    assert len(store) == 1  # one draw gives both
    pd.testing.assert_frame_equal(weekly, model.forecast(task))
    assert list(daily.columns) == ["location", "date", "sample", "value"]
    group = store[0]
    tri = foundation.hub_triangle(
        task.daily_history, "daily", group["places"], task.target_end_dates[-1]
    )
    future = set(tri.times[tri.n_context :])
    late = daily.loc[daily["date"].isin(future)]
    assert not late.empty
    # The count rows drew 1, the total rows 1000 and up: daily values come
    # from the count rows as drawn, not floored at the counts reported.
    assert (late["value"] == 1.0).all()
    # Every day of the target weeks is there, for every place and sample.
    first = pd.Timestamp(task.target_end_dates[0]) - pd.Timedelta(days=6)
    days = pd.date_range(first, task.target_end_dates[-1]).strftime("%Y-%m-%d")
    assert set(daily["date"]) == set(days)
    assert len(daily) == len(days) * len(group["places"]) * task.n_samples


def test_german_7_day_totals_end_on_the_nowcast_date():
    # German tasks move every date to the next Saturday, so the weekly
    # total rows end on the nowcast date and every 7 days back.
    from bdbvhubrenewal.data.germany import HORIZONS, _shift

    nowcast = "2021-11-17"  # a Wednesday
    shifted = pd.Timestamp(nowcast) + _shift(nowcast)
    assert shifted.day_name() == "Saturday"
    assert all(h % 7 == 0 for h in HORIZONS)


@pytest.mark.parametrize("source", ["daily", "weekly"])
def test_zero_shot_and_finetuned_get_the_same_block(
    ft: FoundationModel,  # noqa: ARG001 (the fake release)
    task: ForecastTask,
    source: str,
) -> None:
    # Only the weights differ: the same rows, rolling rows and lag rows,
    # the zero-case places' lags included.
    blocks = []
    for name in (
        f"fm_chronos_{source}_ragged",
        f"fm_chronos_{source}_ragged_ft",
    ):
        store: list = []
        model = dataclasses.replace(
            get_model(name), sample=_aggregate_sampler(store)
        )
        model.forecast(task)
        blocks.append(store[-1])
    zero, tuned = blocks
    assert zero["places"] == tuned["places"]
    assert zero["target_places"] == tuned["target_places"]
    assert sorted(zero["past_covariates"]) == sorted(tuned["past_covariates"])
    np.testing.assert_array_equal(zero["target"], tuned["target"])


def test_the_daily_rolling_rows_hold_every_days_7_day_sum(
    ft: FoundationModel, task: ForecastTask
) -> None:
    store: list = []
    model = dataclasses.replace(ft, sample=_aggregate_sampler(store))
    model.forecast(task)
    group = store[-1]
    n = len(group["places"])
    counts, sums = group["target"][:n], group["target"][n:]
    want = foundation.rolling_totals(counts, 7)
    np.testing.assert_array_equal(sums[:, 6:], want[:, 6:])


class _LevelPipeline:
    """Quantiles of each series and step: the step index plus the level."""

    quantiles = (0.1, 0.5, 0.9)

    def predict_quantiles(
        self, groups, prediction_length, quantile_levels, **_kw
    ):
        import torch

        out = []
        for g in groups:
            n_var = np.atleast_2d(g["target"]).shape[0]
            step = torch.arange(prediction_length, dtype=torch.float32)
            level = torch.tensor(quantile_levels, dtype=torch.float32)
            out.append((step[:, None] + level[None, :]).expand(n_var, -1, -1))
        return out, None


def _spy_triangle(monkeypatch) -> list:
    """Keep each triangle the model builds."""
    seen: list = []
    real = foundation.hub_triangle

    def spy(*args, **kwargs):
        seen.append(real(*args, **kwargs))
        return seen[-1]

    monkeypatch.setattr(foundation, "hub_triangle", spy)
    return seen


def test_daily_7_day_sums_are_the_aggregate_rows_native_quantiles(
    ft: FoundationModel, task: ForecastTask, monkeypatch
) -> None:
    # No draws, no rounding, no floor at the reported count: the 7-day
    # sum of a week past the context is its aggregate row's own quantile.
    pytest.importorskip("torch")
    monkeypatch.setattr(foundation, "_chronos", lambda *_: _LevelPipeline())
    levels = [0.1, 0.5, 0.9]
    seen = _spy_triangle(monkeypatch)
    out = ft.quantiles(task, levels)
    assert out is not None
    tri = seen[-1]
    start = tri.last - ft.lag_count(task) + 1
    future = [w for w in task.target_end_dates if tri.times.index(w) >= start]
    rows = out[out["target_end_date"].isin(future)]
    step = rows["target_end_date"].map(lambda w: tri.times.index(w) - start)
    np.testing.assert_allclose(rows["value"], step + rows["quantile"])


def test_daily_counts_are_the_count_rows_native_quantiles(
    ft: FoundationModel, task: ForecastTask, monkeypatch
) -> None:
    pytest.importorskip("torch")
    monkeypatch.setattr(foundation, "_chronos", lambda *_: _LevelPipeline())
    seen = _spy_triangle(monkeypatch)
    out = ft.daily_quantiles(task, [0.1, 0.5, 0.9])
    assert set(out.columns) == {"location", "date", "quantile", "value"}
    tri = seen[-1]
    start = tri.last - ft.lag_count(task) + 1
    later = out[out["date"].map(lambda d: tri.times.index(d) >= start)]
    assert len(later)
    step = later["date"].map(lambda d: tri.times.index(d) - start)
    np.testing.assert_allclose(later["value"], step + later["quantile"])


def test_the_pipeline_interpolates_to_the_hub_levels_itself(monkeypatch):
    # One path: the hub levels go to Chronos-2's predict_quantiles, which
    # interpolates between its own levels; we do not interpolate again.
    pytest.importorskip("torch")
    asked = []

    class Spy(_LevelPipeline):
        def predict_quantiles(
            self, groups, prediction_length, quantile_levels, **kw
        ):
            asked.append(list(quantile_levels))
            return super().predict_quantiles(
                groups, prediction_length, quantile_levels, **kw
            )

    monkeypatch.setattr(foundation, "_chronos", lambda *_: Spy())
    hub = [0.025, 0.1, 0.25, 0.5, 0.75, 0.9, 0.975]
    group = {"target": np.zeros((2, 6))}
    out = foundation.native_sampler(np.array(hub))([group], 3, 1, 0)[0]
    assert asked == [hub]
    assert out.shape == (7, 2, 3)
    np.testing.assert_allclose(
        out[:, 0, :], np.arange(3)[None, :] + np.array(hub)[:, None]
    )
    assert "np.interp" not in Path(foundation.__file__).read_text()
