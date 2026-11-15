"""The TimesFM 3 forecasters, with a stand-in for TimesFM 3."""

from __future__ import annotations

import dataclasses
import importlib.util
from statistics import NormalDist

import numpy as np
import pandas as pd
import pytest
from bdbvhubrenewal.forecasters.timesfm import (
    EXTENDED_LEVELS,
    TIMESFM_LEVELS,
    extend_quantiles,
    timesfm_arrays,
    timesfm_sampler,
)

from bdbvhubrenewal.evaluation.backtest import BacktestConfig, make_tasks
from bdbvhubrenewal.forecasters import (
    ForecastTask,
    FoundationModel,
    check_samples,
    get_model,
    list_models,
    timesfm,
)
from bdbvhubrenewal.forecasters.foundation import LAYOUTS, SOURCES

MODELS = tuple(f"fm_timesfm_{s}_{lay}" for s in SOURCES for lay in LAYOUTS)
needs_torch = pytest.mark.skipif(
    importlib.util.find_spec("torch") is None, reason="needs torch"
)


@pytest.fixture(scope="module")
def task(
    target: pd.DataFrame, delays: pd.DataFrame, daily: pd.DataFrame
) -> ForecastTask:
    rows = target.loc[target["vintage"] == "2026-10-24"]
    real = target.loc[target["kind"] == "real"]
    return make_tasks(
        rows, delays, BacktestConfig(seed=3), history=real, daily=daily
    )[0]


def test_registered_as_fitted_models_on_timesfm() -> None:
    specs = {s.name: s for s in list_models()}
    for name in MODELS:
        assert specs[name].fitted
        model = get_model(name)
        assert isinstance(model, FoundationModel)
        assert model.sample is timesfm_sampler
        assert "0x" not in repr(model)


def test_complete_block_has_no_covariates() -> None:
    target = np.arange(6.0).reshape(2, 3)
    arrays = timesfm_arrays({"target": target})
    np.testing.assert_array_equal(arrays.target, target)
    assert arrays.covariates is None
    assert arrays.mask is None


def test_ragged_block_masks_missing_cells_and_fills_none() -> None:
    group = {
        "target": np.ones((1, 3)),
        "past_covariates": {
            "a": np.array([1.0, np.nan, 3.0]),
            "b": np.array([np.nan, 5.0, 6.0]),
        },
        "future_covariates": {
            "a": np.array([np.nan, np.nan]),
            "b": np.array([7.0, np.nan]),
        },
    }
    arrays = timesfm_arrays(group)
    assert arrays.covariates.shape == (2, 5)
    expected = np.array(
        [[False, True, False, True, True], [True, False, False, False, True]]
    )
    np.testing.assert_array_equal(arrays.mask, expected)
    # Missing cells are zero and masked, not interpolated or carried.
    np.testing.assert_array_equal(arrays.covariates[arrays.mask], 0.0)
    np.testing.assert_array_equal(
        arrays.covariates[~arrays.mask], [1.0, 3.0, 5.0, 6.0, 7.0]
    )


def test_extension_keeps_the_native_quantiles() -> None:
    levels = np.asarray(TIMESFM_LEVELS)
    q = np.cumsum(np.arange(1.0, 10.0))
    np.testing.assert_allclose(extend_quantiles(levels, q, levels), q)
    # Between levels it is linear in the level.
    mid = extend_quantiles(levels, q, np.array([0.25]))
    np.testing.assert_allclose(mid, (q[1] + q[2]) / 2)


def test_extension_beyond_the_deciles_is_normal_in_the_tails() -> None:
    norm = NormalDist(10.0, 3.0)
    levels = np.asarray(TIMESFM_LEVELS)
    q = np.array([norm.inv_cdf(p) for p in levels])
    out = np.array([0.01, 0.025, 0.05, 0.95, 0.975, 0.99])
    expected = [norm.inv_cdf(p) for p in out]
    np.testing.assert_allclose(extend_quantiles(levels, q, out), expected)


def test_extension_is_monotone_on_leading_axes() -> None:
    levels = np.asarray(TIMESFM_LEVELS)
    rng = np.random.default_rng(1)
    q = rng.gamma(2.0, size=(3, 4, len(levels)))  # crossed on purpose
    out = extend_quantiles(levels, q, np.asarray(EXTENDED_LEVELS))
    assert out.shape == (3, 4, len(EXTENDED_LEVELS))
    assert (np.diff(out, axis=-1) >= 0).all()


class _FakeModel:
    """Quantile `k` of variate `v` at step `h` is `h + level_k`."""

    quantiles = TIMESFM_LEVELS

    def __init__(self) -> None:
        self.calls: list[dict] = []

    def decode(self, target, past_future_covariates=None, **kwargs):
        import torch

        self.calls.append(
            {"target": target, "covariates": past_future_covariates, **kwargs}
        )
        n_cov = (
            0
            if past_future_covariates is None
            else past_future_covariates.shape[1]
        )
        horizon = kwargs.get("horizon") or (
            past_future_covariates.shape[-1] - target.shape[-1]
        )
        n_var = target.shape[1] + n_cov
        step = torch.arange(horizon, dtype=torch.float32)
        level = torch.tensor(self.quantiles, dtype=torch.float32)
        q = step[:, None] + level[None, :]
        return q.expand(target.shape[0], n_var, -1, -1)


@needs_torch
def test_sampler_passes_masks_and_draws_targets_only(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake = _FakeModel()
    monkeypatch.setattr(timesfm, "_timesfm", lambda: fake)
    groups = [
        {
            "target": np.zeros((2, 5)),
            "past_covariates": {"a": np.array([1.0, np.nan, 1, 1, 1])},
            "future_covariates": {"a": np.array([np.nan, 2, 3, 4])},
        },
        {"target": np.zeros((1, 5))},
    ]
    draws = timesfm_sampler(groups, 4, 300, 1)
    assert [d.shape for d in draws] == [(300, 2, 4), (300, 1, 4)]
    mask = fake.calls[0]["past_future_mask"].numpy()[0]
    np.testing.assert_array_equal(np.flatnonzero(mask[0]), [1, 5])
    assert fake.calls[1]["covariates"] is None
    # Levels are extended to Chronos-2's outer levels, 0.01 to 0.99.
    level = draws[0] - np.arange(4)
    assert level.min() < 0.1 - 1e-3
    assert level.max() > 0.9 + 1e-3
    assert ((level > 0.01 - 0.2) & (level < 0.99 + 0.2)).all()
    assert abs(np.corrcoef(level[:, 0, 0], level[:, 1, 0])[0, 1]) < 0.2


@pytest.mark.parametrize("name", MODELS)
def test_forecasts_are_valid_samples(name: str, task: ForecastTask) -> None:
    def sample(groups, horizon, n, seed):  # noqa: ARG001
        return [
            np.full((n, np.atleast_2d(g["target"]).shape[0], horizon), 3.0)
            for g in groups
        ]

    model = dataclasses.replace(get_model(name), sample=sample)
    out = model.forecast(task)
    check_samples(out, task)


@pytest.mark.skipif(
    importlib.util.find_spec("timesfm3") is not None,
    reason="TimesFM 3 is installed",
)
def test_timesfm_missing_fails() -> None:
    with pytest.raises(ImportError):
        timesfm_sampler([{"target": np.zeros((1, 5))}], 3, 10, 0)
