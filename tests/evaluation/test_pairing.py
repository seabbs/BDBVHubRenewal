"""Rolling copula pairing of real models' forecasts across origins."""

from __future__ import annotations

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


@pytest.fixture(scope="module")
def forecasts(
    target: pd.DataFrame, delays: pd.DataFrame, daily: pd.DataFrame
) -> pd.DataFrame:
    real = target.loc[target["kind"] == "real"]
    rows = real.loc[real["vintage"].isin(["2026-10-17", "2026-10-24"])]
    tasks = make_tasks(
        rows, delays, BacktestConfig(seed=3), history=real, daily=daily
    )
    out = run_forecasts([get_model("randomwalknowcast")], tasks)
    return out.loc[out["as_of"].isna()].rename(columns={"vintage": "origin"})


@pytest.fixture(scope="module")
def truth(target: pd.DataFrame) -> pd.DataFrame:
    return rolling_truth(target.loc[target["kind"] == "real"], 7)


def _cells(rows: pd.DataFrame) -> pd.Series:
    keys = ["model", "origin", "location", "target_end_date"]
    return rows.groupby(keys)["value"].apply(
        lambda v: tuple(np.sort(v.to_numpy()))
    )


def test_rolling_pairing_keeps_each_weeks_forecast(
    forecasts: pd.DataFrame, truth: pd.DataFrame
) -> None:
    mine = forecasts.loc[forecasts["model"] == "randomwalknowcast"]
    paired, fits = RollingCopula(PRIOR).pair(mine, truth, seed=1)
    assert len(paired) == len(mine)
    # The country is the provinces' sum, so only the provinces keep theirs.
    kept = paired["location"] != NATIONAL
    pd.testing.assert_series_equal(
        _cells(paired.loc[kept]), _cells(mine.loc[mine["location"] != NATIONAL])
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
    # Chronos-2 forecasts each week's total directly, so it is not paired.
    for source in ("weekly", "daily"):
        for layout in ("complete", "ragged", "ragged_ft"):
            assert not specs[f"fm_chronos_{source}_{layout}"].paired
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
    assert set(fits["model"]) == {"randomwalknowcast"}


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
    assert copulas["n_scores"].max() > 0


def test_with_no_truth_delay_the_release_backtest_pairs_by_the_prior(
    target: pd.DataFrame, delays: pd.DataFrame, daily: pd.DataFrame
) -> None:
    real = target.loc[target["kind"] == "real"]
    tasks = make_tasks(
        real,
        delays.assign(kind="real"),
        BacktestConfig(n_samples=20, seed=2),
        daily=daily,
    )
    copulas = release_backtest(tasks, daily, target).copulas
    assert set(copulas["model"]) == {"randomwalknowcast"}
    assert (copulas["n_scores"] == 0).all()
    assert (copulas["rho_time"] == PRIOR.rho_time).all()
