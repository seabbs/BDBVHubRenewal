"""float32 log density against float64 at real-data values.

The model runs in float32 for speed. A loss of precision (such as
`1 - exp(-y)` at per-capita rates near 1e-7) shows up as noise in the
log density and its second differences, which stalls NUTS. This test
catches such a regression.
"""

import jax
import jax.numpy as jnp
import numpy as np
import numpyro
import pytest
from jax import enable_x64

from bdbv_hub_renewal.model import FALLBACK, MAIN, build_model, model_args
from bdbv_hub_renewal.model.inputs import vintage_from_tables

from .conftest import MAIN_REPO_DATA


def real_vintage(table, config):
    from bdbv_hub_renewal.data import load_delays, load_target

    target, delays = load_target(MAIN_REPO_DATA), load_delays(MAIN_REPO_DATA)
    vintage = target["vintage"].max()
    locations = tuple(sorted(set(target["location"]) - {"CD"}))
    return vintage_from_tables(
        target,
        delays.loc[delays["vintage"] == vintage],
        vintage,
        locations,
        table,
        config,
    )


def log_density_curve(model, data, params, site, steps, dtype):
    def one(t):
        p = {k: jnp.asarray(v, dtype) for k, v in params.items()}
        p[site] = p[site].at[-1].add(t)
        return numpyro.infer.util.log_density(
            model.sample, (), model_args(model, data), p
        )[0]

    return np.array([float(one(t)) for t in steps])


@pytest.mark.skipif(not MAIN_REPO_DATA.exists(), reason="main repo data absent")
@pytest.mark.parametrize("config", [MAIN, FALLBACK], ids=["main", "fallback"])
def test_float32_log_density_matches_float64(table, config):
    data = real_vintage(table, config)
    model = build_model(data, config)
    with numpyro.handlers.seed(rng_seed=3), numpyro.handlers.trace() as tr:
        model.sample(**model_args(model, data))
    params = {
        k: np.asarray(v["value"])
        for k, v in tr.items()
        if v["type"] == "sample" and not v["is_observed"]
    }
    site = "log_rt_baseline_knots"
    steps = np.linspace(-0.02, 0.02, 21)
    lp32 = log_density_curve(model, data, params, site, steps, jnp.float32)
    with enable_x64():
        lp64 = log_density_curve(model, data, params, site, steps, jnp.float64)
    rel = np.max(np.abs(lp32 - lp64) / np.abs(lp64))
    noise = np.std(np.diff(lp32, 2) - np.diff(lp64, 2))
    print(f"\nrelative error {rel:.2e}, second-difference noise {noise:.2e}")
    assert jax.numpy.isfinite(lp32).all()
    assert rel < 1e-4
    assert noise < 0.05


@pytest.mark.skipif(not MAIN_REPO_DATA.exists(), reason="main repo data absent")
@pytest.mark.parametrize("x", [3e-2, 3e-4])
def test_float32_density_stays_smooth_near_poisson(table, x):
    # As 1/sqrt(phi) goes to zero the negative binomial approaches Poisson;
    # the concentration cap keeps its float32 density smooth there.
    data = real_vintage(table, FALLBACK)
    model = build_model(data, FALLBACK)
    with numpyro.handlers.seed(rng_seed=3), numpyro.handlers.trace() as tr:
        model.sample(**model_args(model, data))
    params = {
        k: np.asarray(v["value"])
        for k, v in tr.items()
        if v["type"] == "sample" and not v["is_observed"]
    }
    params["inv_sqrt_phi_known"] = np.float32(x)
    site = "log_rt_baseline_knots"
    steps = np.linspace(-0.02, 0.02, 21)
    lp32 = log_density_curve(model, data, params, site, steps, jnp.float32)
    with enable_x64():
        lp64 = log_density_curve(model, data, params, site, steps, jnp.float64)
    noise = np.std(np.diff(lp32, 2) - np.diff(lp64, 2))
    assert noise < 0.05
