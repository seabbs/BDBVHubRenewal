"""Samplers against R baselinenowcast 0.2.2 in distribution."""

from __future__ import annotations

from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from bnc_helpers import Case, read, read_ints

from bdbvhubrenewal.baselinenowcast import (
    apply_delay,
    baselinenowcast,
    combine_obs_with_pred,
    estimate_delay,
    preprocess_negative_values,
    rolling_sum,
    sample_nb,
    sample_nowcast,
    sample_nowcasts,
    sample_prediction,
    sample_skellam,
)

N_DRAW = 20000


def _moments(x: jax.Array) -> tuple[float, float]:
    a = np.asarray(x)
    return float(a.mean()), float(a.var(ddof=1))


def _agree(
    mean_a: np.ndarray,
    var_a: np.ndarray,
    n_a: int,
    mean_b: np.ndarray,
    var_b: np.ndarray,
    n_b: int,
) -> None:
    """Means within 4.5 and variances within 6 Monte Carlo SE."""
    se_mean = np.sqrt(var_a / n_a + var_b / n_b)
    both_zero = (var_a == 0) & (var_b == 0)
    z = np.where(
        both_zero,
        0.0,
        np.abs(mean_a - mean_b) / np.where(se_mean > 0, se_mean, 1.0),
    )
    assert np.all(np.where(both_zero, np.abs(mean_a - mean_b) < 1e-9, z < 4.5))
    se_var = np.sqrt(2 / (n_a - 1) + 2 / (n_b - 1)) * (var_a + var_b) / 2
    close = np.abs(var_a - var_b) <= 6 * se_var + 1e-9
    assert np.all(close), np.max(
        np.abs(var_a - var_b) / np.where(se_var > 0, se_var, 1)
    )


def test_sample_nb_matches_r_moments() -> None:
    par = read("nb", "sample_par")
    mom = read("nb", "sample_mom")
    for i, (mu, size) in enumerate(par):
        key = jax.random.key(i)
        x = sample_nb(key, jnp.full((N_DRAW,), mu), size)
        m, v = _moments(x)
        _agree(
            np.array(m),
            np.array(v),
            N_DRAW,
            np.array(mom[i, 0]),
            np.array(mom[i, 1]),
            20000,
        )


def test_sample_nb_is_deterministic_in_the_key() -> None:
    pred = jnp.array([5.0, 50.0, 0.0])
    a = sample_nb(jax.random.key(1), pred, 3.0)
    b = sample_nb(jax.random.key(1), pred, 3.0)
    c = sample_nb(jax.random.key(2), pred, 3.0)
    np.testing.assert_array_equal(a, b)
    assert not np.array_equal(a, c)
    assert float(a[2]) == 0.0


def test_sample_nb_vector_size_and_errors() -> None:
    out = sample_nb(
        jax.random.key(0), jnp.ones(4) * 10, jnp.array([1.0, 2, 3, 4])
    )
    assert out.shape == (4,)
    with pytest.raises(ValueError):
        sample_nb(jax.random.key(0), jnp.ones(4), jnp.array([1.0, 2.0]))


def test_sample_skellam_mean_and_variance() -> None:
    for mu, phi in ((10.0, 0.0), (50.0, 0.2), (200.0, 1.0)):
        x = sample_skellam(jax.random.key(5), jnp.full((N_DRAW,), mu), phi)
        m, v = _moments(x)
        se = np.sqrt((mu + phi * mu**2) / N_DRAW)
        assert abs(m - mu) < 4.5 * se
        expected = mu + phi * mu**2
        assert (
            abs(v - expected) < 6 * np.sqrt(2 / (N_DRAW - 1)) * expected * 1.5
        )


def test_sample_skellam_can_be_below_the_prediction_by_more_than_poisson() -> (
    None
):
    x = sample_skellam(jax.random.key(0), jnp.full((5000,), 20.0), 2.0)
    assert float(x.min()) < 0 or float(x.min()) < 5
    assert np.all(np.asarray(x) == np.round(np.asarray(x)))


def _prepared(case: Case) -> tuple[jax.Array, jax.Array]:
    pre = preprocess_negative_values(case.values, case.mask)
    pmf = estimate_delay(pre, case.mask, n=case.n_hist)
    return pre, apply_delay(pre, case.mask, pmf)


@pytest.mark.parametrize("mode", ["id", "agg"])
def test_sample_nowcasts_match_r_in_distribution(case: Case, mode: str) -> None:
    n_r = read_ints(case.name, "draws")[0]
    r = read(case.name, f"samples_{mode}")
    kwargs = {}
    if mode == "agg":
        kwargs["ref_time_aggregator"] = partial(rolling_sum, k=case.k)
    out = baselinenowcast(
        case.values, case.mask, jax.random.key(11), draws=n_r, **kwargs
    )
    n_agg = out.shape[1]
    r = r[-n_agg:]
    a = np.asarray(out)
    _agree(a.mean(0), a.var(0, ddof=1), n_r, r[:, 0], r[:, 1], n_r)


def test_sample_nowcasts_same_key_same_draws(small: Case) -> None:
    a = baselinenowcast(small.values, small.mask, jax.random.key(0), draws=50)
    b = baselinenowcast(small.values, small.mask, jax.random.key(0), draws=50)
    c = baselinenowcast(small.values, small.mask, jax.random.key(1), draws=50)
    np.testing.assert_array_equal(a, b)
    assert not np.array_equal(a, c)


def test_sample_nowcasts_equal_observed_when_fully_observed(
    small: Case,
) -> None:
    out = baselinenowcast(small.values, small.mask, jax.random.key(0), draws=20)
    done = small.mask.all(axis=1)
    pre = np.asarray(preprocess_negative_values(small.values, small.mask))
    np.testing.assert_allclose(
        np.asarray(out)[:, done], np.tile(pre.sum(1)[done], (20, 1))
    )


def test_sample_nowcasts_use_skellam_sampler(small: Case) -> None:
    out = baselinenowcast(
        small.values,
        small.mask,
        jax.random.key(0),
        draws=30,
        uncertainty_model=lambda o, _p: jnp.zeros(o.shape[1]),
        uncertainty_sampler=sample_skellam,
    )
    assert out.shape == (30, small.values.shape[0])


def test_sample_prediction_pieces(small: Case) -> None:
    pre, nc = _prepared(small)
    n_h = int(np.sum(~small.mask.all(axis=1)))
    params = jnp.full((n_h,), 5.0)
    key = jax.random.key(0)
    pred = sample_prediction(key, nc, pre, small.mask, params)
    assert pred.shape == (small.values.shape[0],)
    assert float(pred[:-n_h].sum() if n_h < pred.shape[0] else 0.0) == 0.0
    total = sample_nowcast(key, nc, pre, small.mask, params)
    np.testing.assert_allclose(
        total, combine_obs_with_pred(pred, pre, small.mask)
    )
    many = sample_nowcasts(key, nc, pre, small.mask, params, draws=7)
    assert many.shape == (7, small.values.shape[0])


def test_sample_prediction_uses_tail_of_params(small: Case) -> None:
    pre, nc = _prepared(small)
    n_h = int(np.sum(~small.mask.all(axis=1)))
    long = jnp.concatenate([jnp.full((2,), 1e-3), jnp.full((n_h,), 1e6)])
    out = sample_nowcasts(
        jax.random.key(0), nc, pre, small.mask, long, draws=200
    )
    expected = np.asarray(nc).sum(1)
    tail_rows = slice(-n_h, None)
    assert np.all(
        np.abs(np.asarray(out).mean(0)[tail_rows] - expected[tail_rows])
        < 0.2 * expected[tail_rows] + 5
    )


def test_sample_prediction_aborts(small: Case) -> None:
    pre, nc = _prepared(small)
    n_h = int(np.sum(~small.mask.all(axis=1)))
    key = jax.random.key(0)
    with pytest.raises(ValueError):
        sample_prediction(key, nc, pre, small.mask, jnp.ones(n_h - 1))
    with pytest.raises(ValueError):
        sample_prediction(
            key, nc, pre, small.mask, jnp.ones(small.values.shape[0] + 1)
        )
    other = jnp.asarray(nc).at[0, 0].add(1.0)
    with pytest.raises(ValueError):
        sample_prediction(key, other, pre, small.mask, jnp.ones(n_h))
    with pytest.raises(ValueError):
        sample_nowcasts(key, nc, pre, small.mask, jnp.ones(n_h), draws=0)


def test_no_horizons_returns_observed_counts() -> None:
    v = np.arange(12.0).reshape(4, 3) + 1
    m = np.ones((4, 3), bool)
    out = sample_nowcasts(jax.random.key(0), v, v, m, jnp.ones(1), draws=3)
    np.testing.assert_allclose(out, np.tile(v.sum(1), (3, 1)))
