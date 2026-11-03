"""Fit the main configuration to a simulated vintage.

The outbreak comes from `bdbvhubrenewal.data.simulate.simulate_outbreak`, a
case-level simulator independent of this model: Rt turns from about 1.6
to 0.85 around day 120, the mean delay falls from 12 to 6 days and the
missing-onset share rises then falls.
"""

import datetime as dt
import json
import time
from pathlib import Path

import jax
import numpy as np
import pytest

from bdbvhubrenewal.evaluation.ppc import summarise_draws
from bdbvhubrenewal.forecasters.renewal import ppc_rows
from bdbvhubrenewal.model import MAIN, build_model, fit
from bdbvhubrenewal.model.model import forecast_trajectories

from .conftest import VINTAGE_DAYS, vintage_date

pytestmark = pytest.mark.slow
RESULTS = Path(__file__).parents[1] / "results"


@pytest.fixture(scope="module", params=["main"])
def fitted(request, sim_vintage):
    cfg = {"main": MAIN}[request.param]
    model = build_model(sim_vintage, cfg)
    start = time.perf_counter()
    fit(model, sim_vintage, num_warmup=300, num_samples=300, num_chains=2)
    jax.block_until_ready(model.mcmc.get_samples())
    seconds = time.perf_counter() - start
    return request.param, model, seconds


def truth(outbreak, column, location="CD-IT"):
    par = outbreak.parameters
    return par.loc[par["location"] == location, column].to_numpy()


def on_sim_days(model, outbreak, data, site, subpop=True):
    """Posterior median of a daily site, indexed by simulation day.

    The model axis starts at `data.start_date` (the first-case week), not
    at the simulation start, so days before it are NaN.
    """
    n_init = model.latent.n_initialization_points
    draws = np.asarray(model.mcmc.get_samples()[site])[:, n_init:]
    if subpop:
        draws = draws[..., 0]
    est = np.median(draws, axis=0)[: data.cutoff + 1]
    offset = (data.start_date - outbreak.settings.start).days
    out = np.full(offset + data.cutoff + 1, np.nan)
    out[offset:] = est
    return out


def first_below_one(rt, after=60):
    below = np.nonzero(rt[after:] < 1)[0]
    return int(below[0]) + after if below.size else None


def test_recovery(fitted, outbreak, sim_vintage):
    name, model, seconds = fitted
    smp = model.mcmc.get_samples()
    extra = model.mcmc.get_extra_fields()
    divergent = float(np.mean(extra["diverging"]))
    # Everything below is on simulation days, up to the vintage.
    rt = on_sim_days(model, outbreak, sim_vintage, "latent::rt_subpop")
    true_rt = truth(outbreak, "rt")[: rt.size]
    # Every four weeks from day 56 to the last fitted day.
    checks = list(range(56, rt.size, 28))
    assert checks[-1] >= 196, "the fit should reach the turning point"
    rt_err = np.abs(rt[checks] - true_rt[checks])
    turn, true_turn = first_below_one(rt), first_below_one(true_rt)
    summary = {
        "config": name,
        "fit_seconds": round(seconds, 1),
        "divergent_share": divergent,
        "rt_days": checks,
        "rt_true": np.round(true_rt[checks], 2).tolist(),
        "rt_median": np.round(rt[checks], 2).tolist(),
        "turn_day": turn,
        "true_turn_day": true_turn,
    }
    if "delay_mean" in smp:
        dm = on_sim_days(
            model, outbreak, sim_vintage, "delay_mean", subpop=False
        )
        summary["delay_true"] = np.round(
            truth(outbreak, "delay_mean")[[42, 168]], 1
        ).tolist()
        summary["delay_median"] = np.round(dm[[42, 168]], 1).tolist()
    if "missing_fraction" in smp:
        pi = on_sim_days(model, outbreak, sim_vintage, "missing_fraction")
        window = slice(56, 169)
        summary["pi_true_mean"] = round(
            float(truth(outbreak, "missing")[window].mean()), 3
        )
        summary["pi_median_mean"] = round(float(pi[window].mean()), 3)
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / f"recovery_{name}.json").write_text(
        json.dumps(summary, indent=2)
    )
    print(summary)

    assert divergent < 0.02
    assert np.all(rt_err < 0.3), summary
    assert rt[84] > 1.3 and rt[182] < 1.0
    assert turn is not None and abs(turn - true_turn) <= 21
    if "delay_mean" in summary:
        true_d, est_d = summary["delay_true"], summary["delay_median"]
        assert est_d[1] < est_d[0]
        assert all(
            abs(e - t) / t < 0.2 for e, t in zip(est_d, true_d, strict=True)
        )
    if "pi_median_mean" in summary:
        assert abs(summary["pi_median_mean"] - summary["pi_true_mean"]) < 0.06


def ppc_coverage(data, insample, as_of):
    observed, draws = ppc_rows(data, insample)
    rows = summarise_draws(observed, draws, as_of=as_of)["rows"]
    out = {}
    for name, keep in (
        ("province", lambda r: r["location"] != "CD"),
        ("national", lambda r: r["location"] == "CD"),
    ):
        sel = [r for r in rows if keep(r)]
        inside = [r["lower_90"] <= r["observed"] <= r["upper_90"] for r in sel]
        inside50 = [
            r["lower_50"] <= r["observed"] <= r["upper_50"] for r in sel
        ]
        out[f"ppc90_{name}"] = round(float(np.mean(inside)), 3)
        out[f"ppc50_{name}"] = round(float(np.mean(inside50)), 3)
    return out


def test_forecast_trajectories(fitted, outbreak, sim_vintage):
    name, model, _ = fitted
    start = time.perf_counter()
    draws = forecast_trajectories(model, sim_vintage, n_trajectories=100)
    seconds = time.perf_counter() - start
    n_weeks = sim_vintage.n_weeks + sim_vintage.horizon_weeks
    assert draws["province"].shape == (100, n_weeks, 4)
    assert draws["inactive"].shape == (100, n_weeks, 2)
    assert np.all(draws["province"] >= 0) and np.all(draws["inactive"] >= 0)
    # Coherence: national sample k is the sum of the provinces' sample k.
    np.testing.assert_array_equal(
        draws["national"],
        draws["province"].sum(axis=2) + draws["inactive"].sum(axis=2),
    )
    # Final truth for the last data week and the 3 forecast weeks, for the
    # nation and each modelled province: 20 intervals, so the check does
    # not hang on one or two weeks.
    true = outbreak.truth().set_index(["location", "target_end_date"])["truth"]
    weeks = [
        (sim_vintage.start_date + dt.timedelta(days=7 * i + 6)).isoformat()
        for i in range(n_weeks - 4, n_weeks)
    ]
    series = {"CD": draws["national"][:, -4:]}
    for j, loc in enumerate(sim_vintage.names):
        series[loc] = draws["province"][:, -4:, j]
    inside = []
    for loc, d in series.items():
        lo, hi = np.quantile(d, [0.05, 0.95], axis=0)
        obs = np.array([true.get((loc, w), 0.0) for w in weeks])
        inside.extend((obs >= lo) & (obs <= hi))
        if loc == "CD":
            nat_obs, nat_lo, nat_hi = obs, lo, hi
    forecast90 = float(np.mean(inside))
    coverage = ppc_coverage(
        sim_vintage, draws["insample"], vintage_date(outbreak, VINTAGE_DAYS[-1])
    )
    RESULTS.mkdir(exist_ok=True)
    out = RESULTS / f"recovery_{name}.json"
    summary = json.loads(out.read_text()) if out.exists() else {}
    summary |= {
        "forecast_seconds": round(seconds, 1),
        "national_truth_last4": nat_obs.tolist(),
        "national_90pc_low": nat_lo.tolist(),
        "national_90pc_high": nat_hi.tolist(),
        "forecast90_cells": len(inside),
        "forecast90": round(forecast90, 3),
        **coverage,
    }
    out.write_text(json.dumps(summary, indent=2))
    print(summary)
    # 90% intervals over 20 cells: below 0.7 has under 2% chance if they
    # are calibrated.
    assert forecast90 >= 0.7, summary
    assert 0.8 <= coverage["ppc90_province"] <= 1.0
    assert coverage["ppc90_national"] >= 0.75
