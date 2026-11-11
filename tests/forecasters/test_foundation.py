"""The Chronos-2 forecasters, with a stand-in for Chronos-2."""

from __future__ import annotations

import dataclasses
import importlib.util

import numpy as np
import pandas as pd
import pytest

from bdbvhubrenewal.data.zones import province_of
from bdbvhubrenewal.evaluation.backtest import BacktestConfig, make_tasks
from bdbvhubrenewal.forecasters import (
    NATIONAL,
    ForecastTask,
    FoundationModel,
    ModelUnavailableError,
    check_samples,
    foundation,
    get_model,
    list_models,
)
from bdbvhubrenewal.forecasters.foundation import (
    LAYOUTS,
    MAX_LAG,
    SOURCES,
    chronos_sampler,
    hub_triangle,
    quantile_draws,
)

MODELS = tuple(f"fm_chronos_{s}_{lay}" for s in SOURCES for lay in LAYOUTS)


def _constant(value: float):
    """A sampler whose every draw is `value`."""

    def sample(groups, horizon, n, seed):
        assert seed >= 0
        return [
            np.full((n, np.atleast_2d(g["target"]).shape[0], horizon), value)
            for g in groups
        ]

    return sample


def _recording(store: list):
    """A sampler that keeps its inputs and draws zeros."""

    def sample(groups, horizon, n, seed):
        store.append((groups, horizon))
        return _constant(0.0)(groups, horizon, n, seed)

    return sample


@pytest.fixture(scope="module")
def task(
    target: pd.DataFrame,
    delays: pd.DataFrame,
    daily: pd.DataFrame,
    daily_zones: pd.DataFrame,
) -> ForecastTask:
    from bdbvhubrenewal.data import load_health_zones
    from tests.conftest import DATA

    rows = target.loc[target["vintage"] == "2026-10-24"]
    real = target.loc[target["kind"] == "real"]
    return make_tasks(
        rows,
        delays,
        BacktestConfig(seed=3),
        history=real,
        daily=daily,
        zones=load_health_zones(DATA),
        daily_zones=daily_zones,
    )[0]


def _zones(task: ForecastTask, source: str) -> set[str]:
    rows = task.zone_history if source == "weekly" else task.zone_daily_history
    assert rows is not None
    return {z for z in rows["location"] if province_of(z) in task.locations}


def _reported(task: ForecastTask, out: pd.DataFrame) -> np.ndarray:
    keys = pd.MultiIndex.from_frame(out[["location", "target_end_date"]])
    return (
        task.target.set_index(["location", "target_end_date"])["total"]
        .reindex(keys)
        .fillna(0.0)
        .to_numpy()
    )


def test_registered_as_fitted_models() -> None:
    specs = {s.name: s for s in list_models()}
    for name in MODELS:
        assert specs[name].fitted
        assert not specs[name].baseline
        model = get_model(name)
        assert isinstance(model, FoundationModel)
        # The fit store keys a fit on the repr, so it must not hold an
        # address.
        assert "0x" not in repr(model)


def _history() -> pd.DataFrame:
    """Two weekly releases of one place, the second a week later."""
    return pd.DataFrame(
        {
            "vintage": ["2026-10-17"] * 2 + ["2026-10-24"] * 3,
            "location": ["A"] * 5,
            "target_end_date": [
                "2026-10-10",
                "2026-10-17",
                "2026-10-10",
                "2026-10-17",
                "2026-10-24",
            ],
            "total": [5, 1, 7, 4, 2],
            "cutoff": ["2026-10-17"] * 2 + ["2026-10-24"] * 3,
        }
    )


def test_weekly_triangle_puts_each_release_at_its_lag() -> None:
    tri = hub_triangle(_history(), "weekly", "2026-10-24", ("A",), "2026-10-31")
    assert tri.times == ["2026-10-10", "2026-10-17", "2026-10-24", "2026-10-31"]
    assert tri.last == 2
    assert tri.final[0].tolist() == [7.0, 4.0, 2.0, 0.0]
    partial = tri.partial[0]
    assert partial.shape == (4, MAX_LAG["weekly"])
    # Week 10-10 was 5 at lag 1 and 7 at lag 2; 10-17 1 then 4.
    assert partial[0, 1] == 5.0
    assert partial[0, 2] == 7.0
    assert partial[1, 0] == 1.0
    assert partial[1, 1] == 4.0
    assert partial[2, 0] == 2.0
    # No release saw week 10-10 at lag 0, or any week after the data.
    assert np.isnan(partial[0, 0])
    assert np.isnan(partial[3]).all()
    assert tri.n_context == tri.last - MAX_LAG["weekly"] + 1


def test_daily_triangle_lags_are_days() -> None:
    daily = pd.DataFrame(
        {
            "vintage": ["2026-10-17", "2026-10-24", "2026-10-24"],
            "location": ["A"] * 3,
            "onset_date": ["2026-10-15", "2026-10-15", "2026-10-20"],
            "observed": [1, 2, 1],
            "imputed": [0, 1, 0],
            "cutoff": ["2026-10-16", "2026-10-23", "2026-10-23"],
        }
    )
    tri = hub_triangle(daily, "daily", "2026-10-24", ("A",), "2026-10-24")
    assert tri.times[0] == "2026-10-15"
    assert tri.times[-1] == "2026-10-24"
    assert tri.last == tri.times.index("2026-10-23")
    # Observed plus imputed, at lag 1 day then 8 days.
    assert tri.partial[0, 0, 1] == 1.0
    assert tri.partial[0, 0, 8] == 3.0
    assert tri.final[0, 0] == 3.0


def test_quantile_draws_invert_the_quantile_function() -> None:
    levels = np.array([0.1, 0.5, 0.9])
    values = np.array([[[0.0, 1.0, 3.0]], [[2.0, 2.0, 2.0]]])
    u = np.array([[0.5], [0.3]])
    out = quantile_draws(levels, values, u)
    assert out.shape == (2, 1)
    assert out[0, 0] == pytest.approx(1.0)
    assert out[1, 0] == pytest.approx(2.0)
    # Outside the levels the draw is the outermost quantile.
    low = quantile_draws(levels, values[:1], np.array([[0.01]]))
    assert low[0, 0] == pytest.approx(0.0)


@pytest.mark.parametrize("name", MODELS)
def test_draws_are_counts_at_least_the_reported(
    name: str, task: ForecastTask
) -> None:
    for value in (-50.0, 0.0, 3.0):
        model = dataclasses.replace(get_model(name), sample=_constant(value))
        out = model.forecast(task)
        check_samples(out, task)
        assert set(out["location"]) == {NATIONAL, *task.locations}
        values = out["value"].to_numpy()
        assert (values >= 0).all()
        assert np.array_equal(values, np.round(values))
        reported = _reported(task, out)
        assert (values >= reported).all()
        if value < 0:
            assert np.array_equal(values, reported)
    # Draws above the reported counts raise the forecast past them.
    high = dataclasses.replace(get_model(name), sample=_constant(1e4))
    out = high.forecast(task)
    assert (out["value"].to_numpy() > _reported(task, out)).any()


@pytest.mark.parametrize("name", MODELS)
def test_maps_final_counts_through_a_delay(
    name: str, task: ForecastTask
) -> None:
    model = dataclasses.replace(get_model(name), sample=_constant(0.0))
    model.forecast(task)
    summary = model.delay_summary
    assert summary is not None
    assert set(summary["model"]) == {name}
    assert set(summary["quantity"]) == {"reported_share"}


@pytest.mark.parametrize("source", SOURCES)
def test_one_group_holds_every_place(source: str, task: ForecastTask) -> None:
    lags = MAX_LAG[source]
    # The country, the provinces and their zones are each a series.
    n_places = 1 + len(task.locations) + len(_zones(task, source))
    assert n_places > 1 + len(task.locations)
    sizes = {}
    for layout in LAYOUTS:
        seen: list = []
        model = dataclasses.replace(
            get_model(f"fm_chronos_{source}_{layout}"),
            sample=_recording(seen),
        )
        model.forecast(task)
        groups, horizon = seen[0]
        assert len(groups) == 1
        group = groups[0]
        future = group.get("future_covariates", {})
        sizes[layout] = (group["target"].shape[0], len(future))
        for values in future.values():
            assert len(values) == horizon
    assert sizes["complete"] == (n_places, 0)
    assert sizes["cov"] == (n_places, n_places * lags)
    assert sizes["ragged"] == (n_places * (1 + lags), n_places * lags)


@pytest.mark.parametrize("source", SOURCES)
def test_covariates_hold_only_cells_already_reported(
    source: str, task: ForecastTask
) -> None:
    seen: list = []
    model = dataclasses.replace(
        get_model(f"fm_chronos_{source}_cov"), sample=_recording(seen)
    )
    model.forecast(task)
    groups, horizon = seen[0]
    lags = MAX_LAG[source]
    # Lag d of a time t after the context is known only if t + d is no
    # later than the vintage's cutoff, the last `lags` steps of context.
    for name, values in groups[0]["future_covariates"].items():
        d = int(name.rsplit("_", 1)[1])
        assert np.isnan(values[lags - d :]).all()
    assert any(
        not np.isnan(v).all() for v in groups[0]["future_covariates"].values()
    )
    assert horizon > lags


def test_no_daily_releases_is_unavailable(task: ForecastTask) -> None:
    bare = dataclasses.replace(task, daily_history=None)
    model = dataclasses.replace(
        get_model("fm_chronos_daily_complete"), sample=_constant(0.0)
    )
    with pytest.raises(ModelUnavailableError):
        model.forecast(bare)


class _FakePipeline:
    """Quantiles that grow with the level and the week."""

    quantiles = (0.1, 0.5, 0.9)

    def predict_quantiles(self, groups, prediction_length, quantile_levels):
        import torch

        out = []
        for g in groups:
            n_var = np.atleast_2d(g["target"]).shape[0]
            week = torch.arange(prediction_length, dtype=torch.float32)
            level = torch.tensor(quantile_levels, dtype=torch.float32)
            q = week[:, None] + level[None, :]
            out.append(q.expand(n_var, -1, -1))
        return out, None


@pytest.mark.skipif(
    importlib.util.find_spec("torch") is None, reason="needs torch"
)
def test_draws_are_independent_across_series_and_weeks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(foundation, "_chronos", lambda: _FakePipeline())
    groups = [{"target": np.zeros((2, 5))}, {"target": np.zeros((1, 5))}]
    draws = chronos_sampler(groups, 4, 200, 1)
    assert [d.shape for d in draws] == [(200, 2, 4), (200, 1, 4)]
    # Each draw is the week index plus a level between 0.1 and 0.9.
    level = draws[0] - np.arange(4)
    assert ((level > 0.1 - 1e-6) & (level < 0.9 + 1e-6)).all()
    assert abs(np.corrcoef(level[:, 0, 0], level[:, 0, 1])[0, 1]) < 0.2
    assert abs(np.corrcoef(level[:, 0, 0], level[:, 1, 0])[0, 1]) < 0.2


@pytest.mark.skipif(
    importlib.util.find_spec("chronos") is not None,
    reason="Chronos-2 is installed",
)
def test_chronos_missing_is_unavailable() -> None:
    with pytest.raises(ModelUnavailableError, match="extra chronos"):
        chronos_sampler([{"target": np.zeros((1, 5))}], 3, 10, 0)
