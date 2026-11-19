"""The walk models pair their province draws inside their own forecast."""

from __future__ import annotations

import dataclasses
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from bdbvhubrenewal.data.snapshots import weekly_counts
from bdbvhubrenewal.data.zones import hub_zones
from bdbvhubrenewal.evaluation.backtest import BacktestConfig, make_tasks
from bdbvhubrenewal.forecasters import ForecastTask, get_model
from bdbvhubrenewal.forecasters.zones import zone_task

MODELS = ("randomwalk", "randomwalknowcast")
DATA = Path(__file__).resolve().parents[1] / "fixtures" / "data"


@pytest.fixture(scope="module")
def task(
    target: pd.DataFrame, delays: pd.DataFrame, daily: pd.DataFrame
) -> ForecastTask:
    real = target.loc[target["kind"] == "real"]
    rows = real.loc[real["vintage"] == real["vintage"].max()]
    config = BacktestConfig(seed=3, n_samples=400)
    return make_tasks(rows, delays, config, history=real, daily=daily)[0]


def _run(name: str, task: ForecastTask):
    """Return the model, its paired forecast and its unpaired draws."""
    model = dataclasses.replace(get_model(name), keep_variants=True)
    paired = model.forecast(task)
    return model, paired, model.unpaired()


def _wide(samples: pd.DataFrame) -> pd.DataFrame:
    return samples.pivot(
        index="sample", columns=["location", "target_end_date"], values="value"
    )


def _cells(samples: pd.DataFrame) -> pd.Series:
    return samples.groupby(["location", "target_end_date"])["value"].apply(
        lambda v: tuple(np.sort(v.to_numpy()))
    )


@pytest.mark.parametrize("name", MODELS)
def test_pairing_keeps_every_cells_draws(name: str, task) -> None:
    _, paired, off = _run(name, task)
    pd.testing.assert_series_equal(_cells(paired), _cells(off))
    assert not paired["value"].equals(off["value"])


@pytest.mark.parametrize("name", MODELS)
def test_a_paired_forecast_is_a_function_of_its_task(name: str, task) -> None:
    pd.testing.assert_frame_equal(_run(name, task)[1], _run(name, task)[1])


@pytest.mark.parametrize("name", MODELS)
def test_provinces_are_paired_within_a_week(name: str, task) -> None:
    model, paired, off = _run(name, task)
    weeks = sorted(off["target_end_date"].unique())

    def mean_rank_corr(samples: pd.DataFrame, last: str) -> float:
        week = samples.loc[samples["target_end_date"] == last]
        x = week.pivot(index="sample", columns="location", values="value")
        c = x.loc[:, x.std() > 0].rank().corr().to_numpy()
        k = c.shape[0]
        return float((c.sum() - k) / (k * (k - 1)))

    def national_var(samples: pd.DataFrame, last: str) -> float:
        week = samples.loc[samples["target_end_date"] == last]
        return float(week.groupby("sample")["value"].sum().var())

    assert model.copula_fit["rho_provinces"] > 0.1
    # In every week, nowcast and forecast alike.
    for week in weeks:
        assert abs(mean_rank_corr(off, week)) < 0.02
        assert mean_rank_corr(paired, week) > 0.02
        # So the country, their sum, widens.
        assert national_var(paired, week) > national_var(off, week)


def test_the_copula_record_is_rho_p_with_its_pairs(task) -> None:
    model, _, _ = _run("randomwalknowcast", task)
    fit = model.copula_fit
    assert set(fit) == {
        "model", "origin", "n_scores", "rho_provinces", "n_pairs", "fallback"
    }  # fmt: skip
    assert fit["model"] == "randomwalknowcast"
    assert fit["origin"] == task.vintage
    assert fit["n_scores"] > 0
    # Places of one forecast give pairs of provinces, so rho_p is estimated.
    assert fit["n_pairs"] > 0
    assert not fit["fallback"]
    assert 0.0 <= fit["rho_provinces"] <= 0.95


@pytest.mark.parametrize("name", MODELS)
def test_zones_are_their_own_forecasts_and_not_paired(
    name: str, task, daily: pd.DataFrame, daily_zones: pd.DataFrame
) -> None:
    counts = weekly_counts(daily_zones, task.vintage)
    codes = hub_zones(DATA / "hub-config" / "tasks.json")
    active = tuple(z for z in codes if z in set(counts["location"]))
    zt = zone_task(task, counts, active, pd.concat([daily, daily_zones]))
    model, paired, off = _run(name, zt)
    pd.testing.assert_frame_equal(paired, off)
    assert model.copula_fit is None


@pytest.mark.parametrize("name", MODELS)
def test_a_walk_model_is_not_mistaken_for_a_renewal_variant(name: str) -> None:
    # The fit store reads a renewal variant from a model's `variant`.
    assert not hasattr(get_model(name), "variant")
