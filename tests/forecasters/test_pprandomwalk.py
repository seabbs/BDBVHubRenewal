"""The partially pooled random walk on a nowcast."""

from __future__ import annotations

import jax
import jax.numpy as jnp
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
from bdbvhubrenewal.forecasters.chain import Origin
from bdbvhubrenewal.forecasters.pprandomwalk import Sampler, Tree

SMALL = Sampler(
    num_warmup=40,
    num_samples=40,
    num_chains=1,
    max_rhat=np.inf,
    max_divergent=1.0,
)


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


def test_a_fit_that_does_not_converge_raises() -> None:
    # Two chains of 5 draws from a wide funnel cannot mix.
    import numpyro
    import numpyro.distributions as dist

    from bdbvhubrenewal.forecasters import ModelUnavailableError
    from bdbvhubrenewal.forecasters.pprandomwalk import _run

    def funnel() -> None:
        v = numpyro.sample("v", dist.Normal(0.0, 3.0))
        numpyro.sample("x", dist.Normal(0.0, jnp.exp(v / 2)).expand((10,)))

    strict = Sampler(num_warmup=5, num_samples=20, num_chains=2)
    with pytest.raises(ModelUnavailableError, match="converge"):
        _run(funnel, jax.random.key(0), strict)


def test_the_sampler_checks_r_hat_and_divergences() -> None:
    sampler = Sampler()
    assert sampler.max_rhat == 1.05
    assert sampler.max_divergent == 0.01
