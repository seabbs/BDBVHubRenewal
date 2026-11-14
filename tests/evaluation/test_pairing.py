"""Rolling copula pairing of real models' forecasts across origins."""

from __future__ import annotations

import dataclasses

import numpy as np
import pandas as pd
import pytest

from bdbvhubrenewal.evaluation.backtest import (
    BacktestConfig,
    make_tasks,
    run_forecasts,
)
from bdbvhubrenewal.evaluation.nowcast import release_backtest, rolling_truth
from bdbvhubrenewal.forecasters import NATIONAL, get_model, list_models
from bdbvhubrenewal.forecasters.copula import (
    PRIOR,
    RollingCopula,
    pair_registered,
)


def _independent(groups, horizon, n, seed):
    rng = np.random.default_rng(seed)
    return [
        rng.gamma(
            2.0, 1e3, size=(n, np.atleast_2d(g["target"]).shape[0], horizon)
        )
        for g in groups
    ]


@pytest.fixture(scope="module")
def forecasts(
    target: pd.DataFrame, delays: pd.DataFrame, daily: pd.DataFrame
) -> pd.DataFrame:
    real = target.loc[target["kind"] == "real"]
    rows = real.loc[real["vintage"].isin(["2026-10-17", "2026-10-24"])]
    tasks = make_tasks(
        rows, delays, BacktestConfig(seed=3), history=real, daily=daily
    )
    models = [
        get_model("randomwalknowcast"),
        dataclasses.replace(
            get_model("fm_chronos_weekly_ragged"), sample=_independent
        ),
    ]
    out = run_forecasts(models, tasks)
    return out.loc[out["as_of"].isna()].rename(columns={"vintage": "origin"})


@pytest.fixture(scope="module")
def truth(target: pd.DataFrame) -> pd.DataFrame:
    return rolling_truth(target.loc[target["kind"] == "real"], 7)


def _cells(rows: pd.DataFrame) -> pd.Series:
    keys = ["model", "origin", "location", "target_end_date"]
    return rows.groupby(keys)["value"].apply(
        lambda v: tuple(np.sort(v.to_numpy()))
    )


@pytest.mark.parametrize(
    "model", ["randomwalknowcast", "fm_chronos_weekly_ragged"]
)
def test_rolling_pairing_keeps_each_weeks_forecast(
    model: str, forecasts: pd.DataFrame, truth: pd.DataFrame
) -> None:
    mine = forecasts.loc[forecasts["model"] == model]
    paired, fits = RollingCopula(PRIOR).pair(mine, truth, seed=1)
    assert len(paired) == len(mine)
    summed = model == "randomwalknowcast"
    kept = paired["location"] != NATIONAL if summed else slice(None)
    before = mine["location"] != NATIONAL if summed else slice(None)
    pd.testing.assert_series_equal(
        _cells(paired.loc[kept]), _cells(mine.loc[before])
    )
    assert list(fits["origin"]) == ["2026-10-17", "2026-10-24"]
    assert fits["n_scores"].iloc[0] == 0
    assert fits["n_scores"].iloc[1] > 0


def test_randomwalknowcast_country_stays_the_sum_of_its_provinces(
    forecasts: pd.DataFrame, truth: pd.DataFrame
) -> None:
    mine = forecasts.loc[forecasts["model"] == "randomwalknowcast"]
    paired, _ = RollingCopula(PRIOR).pair(mine, truth, seed=1)
    keys = ["origin", "target_end_date", "sample"]
    prov = (
        paired.loc[paired["location"] != NATIONAL].groupby(keys)["value"].sum()
    )
    nat = paired.loc[paired["location"] == NATIONAL].set_index(keys)["value"]
    pd.testing.assert_series_equal(
        nat.sort_index(),
        prov.reindex(nat.index).sort_index(),
        check_names=False,
    )


def test_models_with_draws_per_cell_are_registered_as_paired() -> None:
    specs = {s.name: s for s in list_models()}
    assert specs["randomwalknowcast"].paired
    for source in ("weekly", "daily"):
        for layout in ("complete", "ragged", "ragged_ft"):
            assert specs[f"fm_chronos_{source}_{layout}"].paired
    assert not specs["randomwalk"].paired


def test_only_paired_models_are_paired(
    forecasts: pd.DataFrame, truth: pd.DataFrame
) -> None:
    other = forecasts.loc[forecasts["model"] == "randomwalknowcast"].assign(
        model="randomwalk"
    )
    both = pd.concat([forecasts, other], ignore_index=True)
    paired, fits = pair_registered(both, truth, seed=1)
    keys = ["model", "origin", "location", "target_end_date", "sample"]
    walk = paired.loc[paired["model"] == "randomwalk"].sort_values(keys)
    pd.testing.assert_frame_equal(
        walk.reset_index(drop=True),
        other.sort_values(keys).reset_index(drop=True),
    )
    assert set(fits["model"]) == {
        "randomwalknowcast",
        "fm_chronos_weekly_ragged",
    }


def test_release_backtest_pairs_with_the_rolling_truth(
    target: pd.DataFrame, delays: pd.DataFrame, daily: pd.DataFrame
) -> None:
    real = target.loc[target["kind"] == "real"]
    tasks = make_tasks(
        real,
        delays.assign(kind="real"),
        BacktestConfig(n_samples=20, seed=2),
        daily=daily,
    )
    result = release_backtest(tasks, daily, target, truth_delay=14)
    copulas = result.copulas
    assert set(copulas["model"]) == {"randomwalknowcast"}
    assert {"origin", "n_scores", "rho_time"} <= set(copulas)
    # With no rolling truth nothing is paired.
    plain = release_backtest(tasks, daily, target)
    assert plain.copulas.empty
