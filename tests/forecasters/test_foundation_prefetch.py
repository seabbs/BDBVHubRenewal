"""Chronos-2 predicted in few batched passes: `_predict` and `prefetch`."""

from __future__ import annotations

import contextlib
import dataclasses
import importlib.util

import numpy as np
import pandas as pd
import pytest

from bdbvhubrenewal.data.zones import ZoneData, hub_zones
from bdbvhubrenewal.evaluation.backtest import BacktestConfig, make_tasks
from bdbvhubrenewal.forecasters import foundation, get_model, series_bias
from bdbvhubrenewal.forecasters.base import ForecastTask

pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("torch") is None, reason="needs torch"
)


class _Pipeline:
    """Quantiles set by each group's own values, the week and the level.

    A group's quantiles differ from any other's, so a batch that mixes up
    its groups gives the wrong quantiles. Records each call.
    """

    quantiles = (0.1, 0.5, 0.9)

    def __init__(self, model: object | None = None) -> None:
        self.calls: list[tuple[int, int]] = []
        if model is not None:
            self.model = model

    def predict_quantiles(
        self, inputs, prediction_length, quantile_levels, batch_size=256
    ):
        import torch

        self.calls.append((len(inputs), batch_size))
        out = []
        for g in inputs:
            own = np.atleast_2d(g["target"]).sum(axis=1)
            week = np.arange(prediction_length)
            level = np.asarray(quantile_levels)
            q = own[:, None, None] + week[None, :, None] + level[None, None]
            out.append(torch.tensor(q, dtype=torch.float32))
        return out, None


def _group(value: float, n: int = 2) -> dict:
    return {
        "target": np.full((n, 5), value),
        "past_covariates": {f"P{i}_lag_1": np.zeros(5) for i in range(n)},
        "places": [f"P{i}" for i in range(n)],
    }


def _bias_rows(monkeypatch: pytest.MonkeyPatch) -> list:
    """Make every model biased; return the rows each pass is given."""
    seen: list = []

    @contextlib.contextmanager
    def assigned(_model, rows):
        seen.append(None if rows is None else np.asarray(rows))
        yield

    monkeypatch.setattr(series_bias, "has_bias", lambda _m: True)
    monkeypatch.setattr(series_bias, "assigned", assigned)
    return seen


def test_predict_batches_groups_of_one_schema(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Each group has 4 rows (2 targets, 2 lags); 3 fit a batch of 8 as 2
    # passes: two groups, then one.
    monkeypatch.setenv("CHRONOS_BATCH", "8")
    pipe = _Pipeline()
    groups = [_group(v) for v in (1.0, 2.0, 3.0)]
    levels = np.asarray(pipe.quantiles)
    out = foundation._predict(pipe, groups, 3, levels)
    assert pipe.calls == [(2, 8), (1, 4)]
    for g, q in zip(groups, out, strict=True):
        alone = foundation._predict(_Pipeline(), [g], 3, levels)[0]
        np.testing.assert_array_equal(q.numpy(), alone.numpy())


def test_predict_gives_each_pass_its_groups_bias_rows(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen = _bias_rows(monkeypatch)
    pipe = _Pipeline(model=object())
    groups = [_group(1.0), _group(2.0, n=3)]
    groups[1]["past_covariates"] = {
        f"P{i}_lag_1": np.zeros(5) for i in range(2)
    }
    groups[1]["places"] = ["P0", "P1", "P2"]
    groups.append(_group(3.0))
    foundation._predict(pipe, groups, 3, np.asarray(pipe.quantiles))
    # Two schemas: groups 0 and 2 in one pass, group 1 in its own.
    assert sorted(pipe.calls) == [(1, 5), (2, 8)]
    want = series_bias.batch_rows(
        [foundation._group_rows(groups[0]), foundation._group_rows(groups[2])]
    )
    assert any(
        r is not None and r.shape == want.shape and (r == want).all()
        for r in seen
    )


def test_native_quantiles_carry_the_series_bias(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # The weekly model's own quantiles were predicted without the rows'
    # places, so a checkpoint's series bias was left out.
    seen = _bias_rows(monkeypatch)
    pipe = _Pipeline(model=object())
    monkeypatch.setattr(foundation, "_chronos", lambda *_: pipe)
    group = _group(1.0)
    foundation.native_sampler(np.array([0.5]))([group], 3, 1, 0)
    want = foundation._group_rows(group)
    assert len(seen) == 1 and seen[0] is not None
    np.testing.assert_array_equal(seen[0], want)


@pytest.fixture(scope="module")
def tasks(
    target: pd.DataFrame,
    delays: pd.DataFrame,
    daily: pd.DataFrame,
    daily_zones: pd.DataFrame,
    health_zones: pd.DataFrame,
    tasks_json,
) -> list[ForecastTask]:
    out = []
    for vintage in ("2026-10-17", "2026-10-24"):
        rows = target.loc[target["vintage"] == vintage]
        real = target.loc[target["kind"] == "real"]
        out += make_tasks(
            rows,
            delays,
            BacktestConfig(seed=3),
            history=real,
            daily=daily,
            zones=ZoneData(health_zones, daily_zones, hub_zones(tasks_json)),
        )
    return out


@pytest.mark.parametrize(
    "name", ["fm_chronos_daily_ragged", "fm_chronos_weekly_ragged"]
)
def test_prefetch_draws_as_one_task_at_a_time(
    monkeypatch: pytest.MonkeyPatch, tasks: list[ForecastTask], name: str
) -> None:
    pipe = _Pipeline()
    monkeypatch.setattr(foundation, "_chronos", lambda *_: pipe)
    model = get_model(name)
    alone = [model.forecast(t) for t in tasks]
    pipe.calls.clear()
    ready = foundation.prefetch(model, tasks)
    passes = len(pipe.calls)
    assert passes <= len(tasks)
    for t, want in zip(tasks, alone, strict=True):
        pd.testing.assert_frame_equal(ready.forecast(t), want)
    # Every task's quantiles came from the prefetch.
    assert len(pipe.calls) == passes


def test_prefetch_serves_native_quantiles(
    monkeypatch: pytest.MonkeyPatch, tasks: list[ForecastTask]
) -> None:
    pipe = _Pipeline()
    monkeypatch.setattr(foundation, "_chronos", lambda *_: pipe)
    model = get_model("fm_chronos_weekly_ragged")
    levels = [0.25, 0.5, 0.75]
    alone = [model.quantiles(t, levels) for t in tasks]
    pipe.calls.clear()
    ready = foundation.prefetch(
        model, [dataclasses.replace(t, n_samples=len(levels)) for t in tasks]
    )
    passes = len(pipe.calls)
    for t, want in zip(tasks, alone, strict=True):
        pd.testing.assert_frame_equal(ready.quantiles(t, levels), want)
    assert len(pipe.calls) == passes
