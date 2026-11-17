"""The walk models pair their draws inside their own forecast."""

from __future__ import annotations

import dataclasses

import numpy as np
import pandas as pd
import pytest
from scipy.stats import spearmanr

from bdbvhubrenewal.evaluation.backtest import BacktestConfig, make_tasks
from bdbvhubrenewal.forecasters import ForecastTask, get_model

MODELS = ("randomwalk", "randomwalknowcast")


@pytest.fixture(scope="module")
def task(
    target: pd.DataFrame, delays: pd.DataFrame, daily: pd.DataFrame
) -> ForecastTask:
    real = target.loc[target["kind"] == "real"]
    rows = real.loc[real["vintage"] == real["vintage"].max()]
    config = BacktestConfig(seed=3, n_samples=400)
    return make_tasks(rows, delays, config, history=real, daily=daily)[0]


def _run(name: str, task: ForecastTask, pairing: str) -> pd.DataFrame:
    model = dataclasses.replace(get_model(name), pairing=pairing)
    return model.forecast(task)


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
    off, paired = _run(name, task, "off"), _run(name, task, "rolling")
    pd.testing.assert_series_equal(_cells(paired), _cells(off))


@pytest.mark.parametrize("name", MODELS)
def test_a_paired_forecast_is_a_function_of_its_task(name: str, task) -> None:
    pd.testing.assert_frame_equal(
        _run(name, task, "rolling"), _run(name, task, "rolling")
    )


@pytest.mark.parametrize("name", MODELS)
def test_each_places_walk_moves_as_a_whole(name: str, task) -> None:
    # The walk weeks of a place are joint trajectories, which pairing
    # keeps: the set of each place's last-two-week pairs is unchanged.
    off, paired = _run(name, task, "off"), _run(name, task, "prior")
    weeks = sorted(off["target_end_date"].unique())[-2:]
    for loc in task.locations:
        a = _wide(off)[[(loc, w) for w in weeks]].to_numpy()
        b = _wide(paired)[[(loc, w) for w in weeks]].to_numpy()
        np.testing.assert_array_equal(
            a[np.lexsort(a.T[::-1])], b[np.lexsort(b.T[::-1])]
        )


@pytest.mark.parametrize("name", MODELS)
def test_places_are_paired_by_the_copula(name: str, task) -> None:
    off, paired = _run(name, task, "off"), _run(name, task, "prior")
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

    assert abs(rho(_wide(off))) < 0.15
    assert rho(_wide(paired)) > 0.3


def test_randomwalknowcast_records_its_copula_from_its_own_errors(
    task,
) -> None:
    model = dataclasses.replace(get_model("randomwalknowcast"))
    model.forecast(task)
    fit = model.copula_fit
    assert fit["model"] == "randomwalknowcast"
    assert fit["origin"] == task.vintage
    assert fit["n_scores"] > 0
    assert 0.0 <= fit["rho_provinces"] <= 0.95
    off = dataclasses.replace(get_model("randomwalknowcast"), pairing="off")
    off.forecast(task)
    assert off.copula_fit is None


def test_the_copula_record_names_the_pairs_behind_each_correlation(
    task,
) -> None:
    model = get_model("randomwalknowcast")
    model.forecast(task)
    fit = model.copula_fit
    for name in ("time", "provinces", "zones"):
        assert isinstance(fit[f"n_{name}"], int)
        assert isinstance(fit[f"prior_{name}"], bool)
    # Places of one forecast give pairs of provinces, so rho_p is estimated.
    assert fit["n_provinces"] > 0
    assert not fit["prior_provinces"]
    # A province task has no zones, so rho_z falls back to the prior.
    assert fit["n_zones"] == 0
    assert fit["prior_zones"]


def test_shrinkage_strength_is_a_setting(task) -> None:
    weak = dataclasses.replace(get_model("randomwalknowcast"), prior_weight=0)
    strong = dataclasses.replace(
        get_model("randomwalknowcast"), prior_weight=1e9
    )
    weak.forecast(task)
    strong.forecast(task)
    assert strong.copula_fit["rho_provinces"] == pytest.approx(0.7, abs=1e-4)
    assert weak.copula_fit["rho_provinces"] != pytest.approx(0.7)


@pytest.mark.parametrize("name", MODELS)
def test_variants_come_from_one_forecast_as_the_model_gives_them(
    name: str, task
) -> None:
    # One forecast gives every pairing and shrinkage strength from the same
    # draws, through the step the model itself calls.
    model = dataclasses.replace(get_model(name), keep_variants=True)
    paired = model.forecast(task)
    pd.testing.assert_frame_equal(model.variant("rolling"), paired)
    pd.testing.assert_frame_equal(model.variant("off"), _run(name, task, "off"))
    pd.testing.assert_frame_equal(
        model.variant("prior"), _run(name, task, "prior")
    )
    weak = dataclasses.replace(get_model(name), prior_weight=0.0)
    pd.testing.assert_frame_equal(
        model.variant("rolling", weight=0.0), weak.forecast(task)
    )
