"""The partially pooled random walk on a nowcast."""

from __future__ import annotations

import jax
import numpy as np
import pandas as pd
import pytest

from bdbvhubrenewal.evaluation.backtest import BacktestConfig, make_tasks
from bdbvhubrenewal.forecasters import (
    ForecastTask,
    PPRandomWalkNowcast,
    RandomWalkNowcast,
    check_samples,
    get_model,
    list_models,
)
from bdbvhubrenewal.forecasters.chain import Origin, walk_changes
from bdbvhubrenewal.forecasters.pprandomwalk import (
    Sampler,
    Tree,
    _median_of,
    _run,
    spread_model,
    walk_data,
)
from bdbvhubrenewal.forecasters.walk import walk_noise

SMALL = Sampler(num_warmup=40, num_samples=40, num_chains=1)


def _task(
    target: pd.DataFrame,
    delays: pd.DataFrame,
    daily: pd.DataFrame,
    vintage: str,
) -> ForecastTask:
    rows = target.loc[target["vintage"] == vintage]
    return make_tasks(rows, delays, BacktestConfig(seed=3), daily=daily)[0]


@pytest.fixture(scope="module")
def task(
    target: pd.DataFrame, delays: pd.DataFrame, daily: pd.DataFrame
) -> ForecastTask:
    return _task(target, delays, daily, "2026-11-01")


@pytest.fixture(scope="module")
def fitted(task: ForecastTask) -> tuple[PPRandomWalkNowcast, pd.DataFrame]:
    model = PPRandomWalkNowcast(sampler=SMALL)
    return model, model.forecast(task)


@pytest.fixture(scope="module")
def origins(task: ForecastTask) -> tuple[Origin, Origin]:
    """The pooled and the own-or-pooled fits at the task's vintage."""
    pooled = PPRandomWalkNowcast(sampler=SMALL).origin_at(task, errors=True)
    return pooled, RandomWalkNowcast().origin_at(task, errors=True)


def test_registered_as_a_fitted_randomwalknowcast() -> None:
    spec = {s.name: s for s in list_models()}["pprandomwalknowcast"]
    assert spec.fitted
    assert not spec.baseline
    model = get_model("pprandomwalknowcast")
    assert isinstance(model, PPRandomWalkNowcast)
    assert isinstance(model, RandomWalkNowcast)


def test_tree_levels_paths_and_leaves() -> None:
    # Country, two provinces, and two zones in the first province.
    tree = Tree.of(np.array([-1, 0, 0, 1, 1]))
    assert tree.level.tolist() == [0, 1, 1, 2, 2]
    assert tree.leaf.tolist() == [False, False, True, True, True]
    # A zone's value adds its province's offset and its own.
    assert tree.path[3].tolist() == [0, 1, 0, 1, 0]
    assert tree.path[0].sum() == 0


def test_samples_are_valid_and_the_fit_is_reproducible(
    task: ForecastTask,
    fitted: tuple[PPRandomWalkNowcast, pd.DataFrame],
    origins: tuple[Origin, Origin],
) -> None:
    model, out = fitted
    check_samples(out, task)
    assert (out["value"] >= 0).all()
    # A second fit of the same task gives the same estimates, so the
    # samples, drawn from them with the task's key, repeat too.
    pooled, _ = origins
    rows = model.rows(pooled, task)
    sigma = [r["sigma"] for r in model.fit_diagnostics if "sigma" in r]
    assert sigma == [float(pooled.sigma[i]) for i in rows]


def test_pooled_delay_is_close_to_the_chain_ladder_for_a_large_place(
    origins: tuple[Origin, Origin],
) -> None:
    pooled, own = origins
    it = pooled.snap.locations.index("CD-IT")
    week = own.snap.n_weeks - 2
    ratio = pooled.point[it, week] / own.point[it, week]
    assert 0.8 < ratio < 1.25


def test_pooled_delay_fits_own_data_where_randomwalknowcast_does(
    origins: tuple[Origin, Origin],
) -> None:
    pooled, own = origins
    np.testing.assert_array_equal(pooled.fit.own, own.fit.own)
    # The country fits every age, so its newest full week, which leans
    # on the young ages, is nowcast as the chain ladder does.
    week = own.start_nowcast
    ratio = pooled.point[0, week] / own.point[0, week]
    assert 0.85 < ratio < 1.15


def test_pooling_shrinks_a_removal_rate_from_one_revision(
    origins: tuple[Origin, Origin],
) -> None:
    pooled, own = origins
    to = own.snap.locations.index("CD-TO")
    assert own.removal is not None
    assert pooled.removal is not None
    k = int(np.argmax(own.removal[to]))
    assert own.removal[to, k] > 0.1
    assert pooled.removal[to, k] < own.removal[to, k]


def test_diagnostics_hold_each_fit(
    fitted: tuple[PPRandomWalkNowcast, pd.DataFrame],
) -> None:
    model, _ = fitted
    fits = [r for r in model.fit_diagnostics if "fit" in r]
    assert {r["fit"] for r in fits} >= {"spread and errors"}
    assert any(str(r["fit"]).startswith("delay at") for r in fits)
    assert all("worst_rhat" in r for r in fits)
    assert all("median_steps" in r and "step_size" in r for r in fits)
    tau = {str(r["fit"])[:5]: r for r in fits}
    assert "delay_tau_1" in tau["delay"]
    assert "sigma_tau_1" in tau["sprea"]


def _spread_fit(
    changes: np.ndarray, noise: np.ndarray, seen: np.ndarray
) -> float:
    """Fit the pooled walk spread alone for one place and return sigma."""
    tree = Tree.of(np.array([-1, 0]))
    empty = np.zeros((0, 2, 1))
    draws, _ = _run(
        spread_model,
        jax.random.key(0),
        Sampler(num_warmup=300, num_samples=300, num_chains=1),
        tree=tree,
        changes=np.vstack([np.zeros_like(changes), changes]),
        noise=np.vstack([np.zeros_like(noise), noise]),
        seen=np.vstack([np.zeros_like(seen), seen]),
        added=empty,
        remaining=empty,
        reported=empty,
        valid=empty > 0,
    )
    return float(np.exp(_median_of(tree, draws, "sigma"))[1])


def test_pooled_walk_spread_is_net_of_poisson_noise() -> None:
    # Weekly counts that are Poisson noise around a flat level of 5: the
    # changes of log(1 + y) are all noise, so the walk spread is small.
    y = np.random.default_rng(1).poisson(5.0, size=200).astype(float)
    changes = np.diff(np.log1p(y))[None, :]
    noise = walk_noise(y)[None, :]
    seen = np.ones_like(changes, dtype=bool)
    raw = float(np.std(changes))
    sigma = _spread_fit(changes, noise, seen)
    assert raw > 0.4
    assert sigma < 0.5 * raw


def test_pooled_walk_spread_keeps_changes_between_weeks_with_no_cases(
    task: ForecastTask,
) -> None:
    model = RandomWalkNowcast()
    origin = model.origin_at(task, errors=False)
    changes, noise, seen = walk_data(origin)
    diffs = walk_changes(origin.reported, origin.start, origin.settings)
    assert seen.sum(axis=1).tolist() == [d.size for d in diffs]
    # 0 to 0 changes are 0 with no Poisson noise, and they enter.
    zero = seen & (changes == 0) & (noise == 0)
    assert zero.any()
