"""Uncertainty estimation against R baselinenowcast 0.2.2."""

from __future__ import annotations

import warnings
from functools import partial
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from bnc_helpers import Case, from_na, read, read_vec

from bdbvhubrenewal.baselinenowcast import (
    estimate_delay,
    estimate_uncertainty,
    estimate_uncertainty_retro,
    fit_by_horizon,
    fit_nb,
    fit_skellam,
    preprocess_negative_values,
    rolling_sum,
    sample_skellam,
)

NB_CASES = 7


def _capture() -> tuple[dict[str, Any], Any]:
    store: dict[str, Any] = {}

    def model(obs: Any, pred: Any) -> Any:
        store["obs"] = np.asarray(obs)
        store["pred"] = np.asarray(pred)
        return fit_by_horizon(obs, pred)

    return store, model


def _retro(case: Case, **kwargs: Any) -> Any:
    pre = preprocess_negative_values(case.values, case.mask)
    return estimate_uncertainty_retro(
        pre, case.mask, case.n_hist, case.n_retro, **kwargs
    )


def test_retro_obs_and_pred_match_r_identity(case: Case) -> None:
    store, model = _capture()
    _retro(case, uncertainty_model=model)
    np.testing.assert_allclose(
        store["obs"], read(case.name, "retro_obs_id"), rtol=1e-10, atol=1e-8
    )
    np.testing.assert_allclose(
        store["pred"], read(case.name, "retro_pred_id"), rtol=1e-9, atol=1e-8
    )


def test_retro_obs_and_pred_match_r_rolling_sum(case: Case) -> None:
    store, model = _capture()
    agg = partial(rolling_sum, k=case.k)
    _retro(case, uncertainty_model=model, ref_time_aggregator=agg)
    np.testing.assert_allclose(
        store["obs"], read(case.name, "retro_obs_agg"), rtol=1e-10, atol=1e-8
    )
    np.testing.assert_allclose(
        store["pred"], read(case.name, "retro_pred_agg"), rtol=1e-9, atol=1e-8
    )


def test_sizes_match_r(case: Case) -> None:
    got = _retro(case)
    np.testing.assert_allclose(got, read_vec(case.name, "size_id"), rtol=1e-3)
    agg = partial(rolling_sum, k=case.k)
    got_agg = _retro(case, ref_time_aggregator=agg)
    np.testing.assert_allclose(
        got_agg, read_vec(case.name, "size_agg"), rtol=1e-3
    )


def test_fit_by_horizon_matches_per_column_fit() -> None:
    obs = np.round(np.abs(np.random.default_rng(0).normal(30, 6, (8, 3))))
    pred = np.full((8, 3), 30.0)
    got = fit_by_horizon(obs, pred)
    expected = [float(fit_nb(obs[:, i], pred[:, i])) for i in range(3)]
    np.testing.assert_allclose(got, expected)


@pytest.mark.parametrize("i", range(1, NB_CASES + 1))
def test_fit_nb_matches_r_optimize(i: int) -> None:
    xm = read("nb", f"fit_in_{i}")
    expected = read("nb", f"fit_out_{i}").item()
    got = float(fit_nb(xm[:, 0], xm[:, 1]))
    np.testing.assert_allclose(got, expected, rtol=1e-3)


def test_fit_nb_empty_is_nan() -> None:
    assert np.isnan(float(fit_nb(np.array([]), np.array([]))))


def test_fit_nb_aborts() -> None:
    with pytest.raises(ValueError):
        fit_nb(np.array([-1.0, 2.0]), np.array([1.0, 1.0]))
    with pytest.raises(ValueError):
        fit_nb(np.array([1.0, 2.0]), np.array([-1.0, 1.0]))
    with pytest.raises(ValueError):
        fit_nb(np.array([1.5, 2.0]), np.array([1.0, 1.0]))


def test_fit_skellam_recovers_phi() -> None:
    key = jax.random.key(3)
    for phi in (0.0, 0.05, 0.4):
        mu = jnp.full((20000,), 60.0)
        draws = sample_skellam(key, mu, phi)
        got = float(fit_skellam(draws, mu))
        assert abs(got - phi) < 0.02 + 0.05 * phi
        key = jax.random.fold_in(key, 1)


def test_fit_skellam_allows_negative_updates() -> None:
    x = np.array([-3.0, 2.0, -1.0, 5.0])
    mu = np.array([4.0, 4.0, 4.0, 4.0])
    assert float(fit_skellam(x, mu)) > 0


def test_fit_skellam_zero_prediction_is_zero() -> None:
    assert float(fit_skellam(np.array([1.0, 2.0]), np.zeros(2))) == 0.0


def test_fit_skellam_floors_at_zero() -> None:
    mu = np.array([10.0, 10.0])
    assert float(fit_skellam(mu, mu)) == 0.0


def test_fit_skellam_in_fit_by_horizon() -> None:
    obs = np.array([[5.0, 20.0], [7.0, 1.0], [3.0, 40.0]])
    pred = np.array([[6.0, 10.0], [6.0, 10.0], [6.0, 10.0]])
    got = fit_by_horizon(obs, pred, fit_model=fit_skellam)
    assert got.shape == (2,)
    assert float(got[1]) > float(got[0])


def test_retro_aborts(small: Case) -> None:
    pre = preprocess_negative_values(small.values, small.mask)
    with pytest.raises(ValueError):
        estimate_uncertainty_retro(pre, small.mask, small.n_hist, 1)
    with pytest.raises(ValueError):
        estimate_uncertainty_retro(
            pre, small.mask, small.mask.shape[0], small.mask.shape[0]
        )
    with pytest.raises(ValueError):
        estimate_uncertainty_retro(pre, small.mask, 1, 3)


def test_estimate_uncertainty_skips_failed_nowcasts() -> None:
    v, m = from_na(read("toy", "tri"))
    pre = preprocess_negative_values(v, m)
    from bdbvhubrenewal.baselinenowcast import (
        apply_reporting_structure,
        estimate_and_apply_delay,
        get_reporting_structure,
        truncate_to_rows,
    )

    trunc = truncate_to_rows(pre, m, n=3)
    s = get_reporting_structure(m)
    retro = []
    for tv, tm in trunc:
        rm = apply_reporting_structure(tv, tm, s)
        retro.append((jnp.where(jnp.asarray(rm), tv, 0.0), rm))
    ncs = [
        estimate_and_apply_delay(rv, rm, n=5, validate=False)
        for rv, rm in retro
    ]
    full = estimate_uncertainty(ncs, trunc, retro, n=3)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        skipped = estimate_uncertainty(
            [ncs[0], None, ncs[2]], trunc, retro, n=3
        )
    assert full.shape == skipped.shape
    with pytest.raises(ValueError):
        estimate_uncertainty([None, None, None], trunc, retro, n=3)
    with pytest.raises(ValueError):
        estimate_uncertainty(ncs[:2], trunc, retro, n=3)


def test_delay_pmf_argument_is_used(small: Case) -> None:
    pre = preprocess_negative_values(small.values, small.mask)
    pmf = estimate_delay(pre, small.mask, n=small.n_hist)
    a = estimate_uncertainty_retro(
        pre, small.mask, small.n_hist, small.n_retro, delay_pmf=pmf
    )
    assert a.shape[0] == int(np.sum(~small.mask.all(axis=1)))


def _hub_week_inputs(
    cutoff: str, step: int, n_ref: int = 262, d: int = 84
) -> Any:
    import pandas as pd

    from bdbvhubrenewal.baselinenowcast import (
        estimate_and_apply_delay,
        hub_weeks,
    )

    rng = np.random.default_rng(1)
    pmf = np.exp(-np.arange(d + 1) / 10)
    pmf /= pmf.sum()
    full = rng.poisson(30 * pmf[None, :] * np.ones((n_ref, 1))).astype(float)

    def tri(c: int) -> tuple[np.ndarray, np.ndarray]:
        t = np.arange(c + 1)[:, None]
        dl = np.arange(d + 1)[None, :]
        m = (t + dl) <= c
        return np.where(m, full[: c + 1], 0.0), m

    first = pd.Timestamp(cutoff) - pd.Timedelta(days=n_ref - 1)
    lv, lm = tri(n_ref - 1)
    qs = [n_ref - 1 - step, n_ref - 1 - 2 * step]
    ncs, trs, rts = [], [], []
    for q in qs:
        v, m = tri(q)
        ncs.append(estimate_and_apply_delay(v, m, n=150))
        trs.append((lv[: q + 1], lm[: q + 1]))
        rts.append((v, m))
    return ncs, trs, rts, hub_weeks(first.date())


@pytest.mark.parametrize(("step", "horizons"), [(7, 12), (8, 13)])
def test_estimate_uncertainty_runs_with_hub_weeks(
    step: int, horizons: int
) -> None:
    # The cut-off is a Saturday, so the first retrospective origin is a
    # Saturday only when it is a whole number of weeks earlier.
    ncs, trs, rts, agg = _hub_week_inputs("2026-10-24", step)
    store, model = _capture()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        out = estimate_uncertainty(
            ncs, trs, rts, n=2, uncertainty_model=model, ref_time_aggregator=agg
        )
    assert store["obs"].shape == (2, horizons)
    assert out.shape[0] == horizons
