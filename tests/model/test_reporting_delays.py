"""The reporting delay's discretised pmfs, against the R packages, live.

epinowcast's own Stan functions (`enw_stan_to_r`), the simulator's R
engine (`packages/nowcastfoundation/r/R/delays.R`) and primarycensored are
called directly; the tests skip only where R or a package is missing.
"""

from __future__ import annotations

import itertools
from typing import Any

import jax
import numpy as np
import pytest
from pyrenew.deterministic import DeterministicVariable

from nowcastrenewal.reporting.delays import (
    FAMILIES,
    DelayParameters,
    ParametricDelay,
    double_censored_log_pmf,
    double_censored_logit_hazard,
    interval_pmf,
)

from .r_live import run_r

# Parameters across the grid's ranges (about its 5%, 50% and 95% points):
# lognormal median and sdlog, gamma mean and shape, exponential mean.
DMAX = (21, 42, 84)
CASES: list[dict[str, Any]] = (
    [
        {"family": "lognormal", "values": {"delay_median": m, "delay_sdlog": s}}
        for m, s in itertools.product((1.4, 3.2, 7.3), (0.55, 0.9, 1.5))
    ]
    + [
        {"family": "gamma", "values": {"delay_mean": m, "delay_cv": 1 / k**0.5}}
        for m, k in itertools.product((1.9, 5.0, 13.0), (0.6, 1.33, 1.9))
    ]
    + [
        {"family": "exponential", "values": {"delay_mean": m}}
        for m in (1.9, 5.0, 13.0)
    ]
)
DIST = {"exponential": 1, "lognormal": 2, "gamma": 3}


def _rows() -> list[dict[str, Any]]:
    rows = []
    for case, dmax in itertools.product(CASES, DMAX):
        p = DelayParameters.from_values(case["family"], case["values"])
        rows.append({**case, "dmax": dmax, "mu": p.mu, "sigma": p.sigma})
    return rows


ROWS = _rows()


@pytest.fixture(scope="module")
def epinowcast_hazards() -> list[dict[str, Any]]:
    """epinowcast's discretised logit hazards and log pmfs, one R run."""
    code = """
    suppressMessages(epinowcast::enw_stan_to_r(
      "discretised_logit_hazard.stan", verbose = FALSE
    ))
    lapply(seq_len(nrow(input)), function(i) {
      r <- input[i, ]
      list(
        logit_hazard = discretised_logit_hazard(
          r$mu, r$sigma, r$dmax, r$dist, 2L, 0L
        ),
        log_pmf = discretised_logit_hazard(
          r$mu, r$sigma, r$dmax, r$dist, 2L, 1L
        )
      )
    })
    """
    inputs = [
        {
            "mu": r["mu"],
            "sigma": r["sigma"],
            "dmax": r["dmax"],
            "dist": DIST[r["family"]],
        }
        for r in ROWS
    ]
    return run_r(code, inputs, packages=["epinowcast", "cmdstanr"])


@pytest.mark.parametrize("i", range(len(ROWS)))
def test_double_censored_delays_match_epinowcast(
    i: int, epinowcast_hazards: list[dict[str, Any]]
) -> None:
    r, want = ROWS[i], epinowcast_hazards[i]
    with jax.enable_x64(True):
        lh = np.asarray(
            double_censored_logit_hazard(
                r["family"], r["mu"], r["sigma"], r["dmax"]
            )
        )
        lp = np.asarray(
            double_censored_log_pmf(r["family"], r["mu"], r["sigma"], r["dmax"])
        )
    want_lh = np.array(
        [np.inf if x in (None, "Inf") else x for x in want["logit_hazard"]],
        dtype=float,
    )
    assert np.isposinf(lh[-1])
    assert np.isposinf(want_lh[-1])
    np.testing.assert_allclose(lh[:-1], want_lh[:-1], rtol=1e-9, atol=1e-9)
    np.testing.assert_allclose(lp, want["log_pmf"], rtol=1e-9, atol=1e-9)


def test_the_double_censored_pmf_sums_to_one() -> None:
    for r in ROWS:
        with jax.enable_x64(True):
            p = np.exp(
                double_censored_log_pmf(
                    r["family"], r["mu"], r["sigma"], r["dmax"]
                )
            )
        assert p.sum() == pytest.approx(1.0, abs=1e-12)


@pytest.fixture(scope="module")
def engine_pmfs() -> list[list[float]]:
    """The simulator R engine's delay pmfs (`delay_parameters`)."""
    code = """
    source("packages/nowcastfoundation/r/R/delays.R")
    lapply(seq_len(length(input$family)), function(i) {
      delay_parameters(input$family[i], input$values[[i]], input$dmax[i])$pmf
    })
    """
    cases = [*CASES, {"family": "none", "values": {"delay_mean": 4.0}}]
    inputs = {
        "family": [c["family"] for c in cases for _ in DMAX],
        "values": [c["values"] for c in cases for _ in DMAX],
        "dmax": [d for _ in cases for d in DMAX],
    }
    return run_r(code, inputs)


def test_the_interval_pmf_matches_the_simulator_engine(
    engine_pmfs: list[list[float]],
) -> None:
    cases = [*CASES, {"family": "none", "values": {"delay_mean": 4.0}}]
    pairs = [(c, d) for c in cases for d in DMAX]
    for (case, dmax), want in zip(pairs, engine_pmfs, strict=True):
        got = interval_pmf(case["family"], case["values"], dmax)
        np.testing.assert_allclose(got, want, rtol=1e-12, atol=1e-15)


def test_a_parametric_delay_component_returns_its_pmf() -> None:
    p = DelayParameters.from_values("lognormal", CASES[4]["values"])
    delay = ParametricDelay(
        "delay",
        family="lognormal",
        mu_rv=DeterministicVariable("mu", np.array([p.mu, p.mu + 0.3])),
        sigma_rv=DeterministicVariable("sigma", np.array([p.sigma, p.sigma])),
        dmax=21,
    )
    with jax.enable_x64(True):
        out = np.asarray(delay())
    assert out.shape == (2, 21)
    np.testing.assert_allclose(out.sum(axis=1), 1.0, atol=1e-12)
    # A larger location delays reports.
    assert (out[1] * np.arange(21)).sum() > (out[0] * np.arange(21)).sum()


def test_the_families_are_the_grids() -> None:
    assert set(FAMILIES) == {"exponential", "lognormal", "gamma"}
