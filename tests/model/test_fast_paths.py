"""The fast weekly-delay and release-pair paths give the same density.

Each fast path is checked against the code it replaces, in float64, for
the values and for the gradient.
"""

import dataclasses

import jax
import jax.numpy as jnp
import numpy as np
import numpyro
import pytest
from jax.flatten_util import ravel_pytree
from numpyro.infer.util import log_density

from bdbvhubrenewal.model import MAIN, build_model, model_args
from bdbvhubrenewal.model import publication_lag as pl
from bdbvhubrenewal.model.observation import (
    onset_expectations_by_place,
    weekly_onset_expectations,
)

from .reference_kernels import kept_share_reference

STREAMS = ("known", "imputed", "known_final", "imputed_final", "triangle")


def _inputs(rng, n_w=12, dmax=20, n_prov=3):
    n = 7 * n_w
    return (
        rng.gamma(5, 4, (n, n_prov)),
        rng.dirichlet(np.ones(dmax + 1), (n_prov, n_w)),
        rng.uniform(0.05, 0.3, (n, n_prov)),
        rng.dirichlet(np.ones(dmax + 1)),
        rng.dirichlet(np.ones(4)),
    )


def _weekly_sum(x):
    return x.reshape(x.shape[0] // 7, 7, *x.shape[1:]).sum(1)


@pytest.mark.parametrize("final", [True, False])
@pytest.mark.parametrize("cutoff_back", [1, 10, 30])
def test_weekly_expectations_match_the_daily_blocks(final, cutoff_back):
    rng = np.random.default_rng(11)
    onsets, delay_w, missing, hub, lag = _inputs(rng)
    n = onsets.shape[0]
    cutoff = n - cutoff_back
    weights = {
        k: rng.normal(size=s)
        for k, s in (
            ("known", (n // 7, 3)),
            ("imputed", (n // 7, 3)),
            ("known_final", (n // 7, 3)),
            ("imputed_final", (n // 7, 3)),
            ("triangle", (n // 7, 3, delay_w.shape[-1])),
        )
    }

    def daily(on, f, pi, lg):
        out = onset_expectations_by_place(
            on, jnp.repeat(f, 7, axis=1), pi, hub, cutoff, final, lg, True
        )
        return {k: _weekly_sum(v) for k, v in out.items()}

    def weekly(on, f, pi, lg):
        return weekly_onset_expectations(on, f, pi, hub, cutoff, final, lg)

    def loss(fn):
        def total(*args):
            out = fn(*args)
            return sum(jnp.sum(jnp.log(out[k]) * weights[k]) for k in STREAMS)

        return jax.value_and_grad(total, argnums=(0, 1, 2, 3))

    with jax.enable_x64(True):
        args = tuple(jnp.asarray(a) for a in (onsets, delay_w, missing, lag))
        ref, new = daily(*args), weekly(*args)
        for key in STREAMS:
            assert new[key].shape == ref[key].shape
            np.testing.assert_allclose(new[key], ref[key], rtol=1e-10)
        v0, g0 = loss(daily)(*args)
        v1, g1 = loss(weekly)(*args)
        np.testing.assert_allclose(v1, v0, rtol=1e-10)
        for a, b in zip(g1, g0, strict=True):
            np.testing.assert_allclose(a, b, rtol=1e-9, atol=1e-12)


def _pairs(rng, n=40, k=3):
    before = rng.integers(-3, 12, (n, 1)) + np.arange(7)
    return {
        "before": before,
        "after": before + rng.integers(1, 9, (n, 1)),
        "kept": rng.integers(0, 10, n).astype(float),
        "total": np.full(n, 10.0),
        "pair": np.eye(k)[rng.integers(0, k, n)],
    }


def test_kept_share_matches_the_direct_lookup():
    rng = np.random.default_rng(3)
    pairs = _pairs(rng)
    with jax.enable_x64(True):
        pmf = jnp.asarray(rng.dirichlet(np.ones(8)))
        background = jnp.asarray([0.1, 0.05, 0.2])
        np.testing.assert_allclose(
            pl.kept_share(pmf, background, pairs),
            kept_share_reference(pmf, background, pairs),
            rtol=1e-12,
        )

        def grads(fn):
            return jax.grad(
                lambda p, b: jnp.sum(jnp.log(fn(p, b, pairs))), (0, 1)
            )(pmf, background)

        for a, b in zip(
            grads(pl.kept_share), grads(kept_share_reference), strict=True
        ):
            np.testing.assert_allclose(a, b, rtol=1e-10)


def test_full_model_density_and_gradient_are_unchanged(
    sim_vintage, monkeypatch
):
    """Fast paths against the daily blocks and direct lookup, 4 draws."""
    rng = np.random.default_rng(8)
    data = dataclasses.replace(sim_vintage, release_pairs=_pairs(rng, 30))
    with jax.enable_x64(True):
        model = build_model(data, MAIN)
        args = model_args(model, data)
        obs = model.observations["onsets"]
        draws = []
        for seed in range(4):
            with numpyro.handlers.trace() as tr:
                numpyro.handlers.seed(model.sample, rng_seed=seed)(**args)
            draws.append(
                {
                    k: v["value"]
                    for k, v in tr.items()
                    if v["type"] == "sample" and not v["is_observed"]
                }
            )

        def density_and_grad(values):
            flat, unravel = ravel_pytree(values)

            def ld(x):
                return log_density(model.sample, (), args, unravel(x))[0]

            return jax.value_and_grad(ld)(flat)

        fast = [density_and_grad(v) for v in draws]
        assert obs.weekly_delay
        obs.weekly_delay = False
        monkeypatch.setattr(pl, "kept_share", kept_share_reference)
        slow = [density_and_grad(v) for v in draws]
    for (v1, g1), (v0, g0) in zip(fast, slow, strict=True):
        np.testing.assert_allclose(v1, v0, rtol=1e-10)
        np.testing.assert_allclose(
            g1, g0, rtol=1e-8, atol=1e-10 * float(jnp.max(jnp.abs(g0)))
        )
