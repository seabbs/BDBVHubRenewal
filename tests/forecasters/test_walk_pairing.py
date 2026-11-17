"""The walk models pair their province draws inside their own forecast."""

from __future__ import annotations

import dataclasses
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy.stats import spearmanr

from bdbvhubrenewal.data.zones import hub_zones, weekly_zone_counts
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
def test_each_places_walk_moves_as_a_whole(name: str, task) -> None:
    # The walk weeks of a place are joint trajectories, which pairing
    # keeps: the set of each place's last-two-week pairs is unchanged.
    _, paired, off = _run(name, task)
    weeks = sorted(off["target_end_date"].unique())[-2:]
    for loc in task.locations:
        a = _wide(off)[[(loc, w) for w in weeks]].to_numpy()
        b = _wide(paired)[[(loc, w) for w in weeks]].to_numpy()
        np.testing.assert_array_equal(
            a[np.lexsort(a.T[::-1])], b[np.lexsort(b.T[::-1])]
        )


@pytest.mark.parametrize("name", MODELS)
def test_provinces_are_paired_within_a_week(name: str, task) -> None:
    model, paired, off = _run(name, task)
    last = sorted(off["target_end_date"].unique())[-1]
    big = (
        off.loc[off["target_end_date"] == last]
        .groupby("location")["value"]
        .std()
        .sort_values()
        .index[-2:]
    )
    a, b = ((loc, last) for loc in big)

    def rho(w: pd.DataFrame) -> float:
        return float(spearmanr(w[a], w[b]).statistic)

    assert model.copula_fit["rho_provinces"] > 0.1
    assert abs(rho(_wide(off))) < 0.15
    assert rho(_wide(paired)) > rho(_wide(off)) + 0.05


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
    counts = weekly_zone_counts(daily_zones, task.vintage)
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
