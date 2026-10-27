"""Fit the main and fallback configurations to a simulated vintage.

The outbreak comes from `bdbv_hub_renewal.data.simulate.simulate_outbreak`, a
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

from bdbv_hub_renewal.model import FALLBACK, MAIN, build_model, fit
from bdbv_hub_renewal.model.model import forecast_trajectories
from bdbv_hub_renewal.models.renewal import ppc_rows
from bdbv_hub_renewal.ppc import summarise_draws

from .conftest import VINTAGE_DAYS, vintage_date

pytestmark = pytest.mark.slow
RESULTS = Path(__file__).parents[1] / "results"
N_INIT = 42


@pytest.fixture(scope="module", params=["main", "fallback"])
def fitted(request, sim_vintage):
    cfg = {"main": MAIN, "fallback": FALLBACK}[request.param]
    model = build_model(sim_vintage, cfg)
    start = time.perf_counter()
    fit(model, sim_vintage, num_warmup=300, num_samples=300, num_chains=2)
    jax.block_until_ready(model.mcmc.get_samples())
    seconds = time.perf_counter() - start
    return request.param, model, seconds


def truth(outbreak, column, location="CD-IT"):
    par = outbreak.parameters
    return par.loc[par["location"] == location, column].to_numpy()


def first_below_one(rt, after=60):
    below = np.nonzero(rt[after:] < 1)[0]
    return int(below[0]) + after if below.size else None


def test_recovery(fitted, outbreak, sim_vintage):
    name, model, seconds = fitted
    smp = {k: np.asarray(v) for k, v in model.mcmc.get_samples().items()}
    extra = model.mcmc.get_extra_fields()
    divergent = float(np.mean(extra["diverging"]))
    days = np.arange(sim_vintage.cutoff + 1)
    rt = np.median(smp["latent::rt_subpop"][:, N_INIT:, 0], axis=0)[days]
    true_rt = truth(outbreak, "rt")[days]
    checks = [56, 84, 112, 140, 168, 196]
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
        dm = np.median(smp["delay_mean"][:, N_INIT:], axis=0)
        summary["delay_true"] = np.round(
            truth(outbreak, "delay_mean")[[42, 168]], 1
        ).tolist()
        summary["delay_median"] = np.round(dm[[42, 168]], 1).tolist()
    if "missing_fraction" in smp:
        pi = np.median(smp["missing_fraction"][:, N_INIT:, 0], axis=0)
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
    # Final national truth for the last data week and the 3 forecast weeks.
    true = outbreak.truth()
    nat = true.loc[true["location"] == "CD"].set_index("target_end_date")[
        "truth"
    ]
    weeks = [
        (sim_vintage.start_date + dt.timedelta(days=7 * i + 6)).isoformat()
        for i in range(n_weeks - 4, n_weeks)
    ]
    lo, hi = np.quantile(draws["national"][:, -4:], [0.05, 0.95], axis=0)
    obs = nat.reindex(weeks).fillna(0).to_numpy()
    inside = (obs >= lo) & (obs <= hi)
    coverage = ppc_coverage(
        sim_vintage, draws["insample"], vintage_date(outbreak, VINTAGE_DAYS[-1])
    )
    out = RESULTS / f"recovery_{name}.json"
    summary = json.loads(out.read_text()) if out.exists() else {}
    summary |= {
        "forecast_seconds": round(seconds, 1),
        "national_truth_last4": obs.tolist(),
        "national_90pc_low": lo.tolist(),
        "national_90pc_high": hi.tolist(),
        **coverage,
    }
    out.write_text(json.dumps(summary, indent=2))
    print(summary)
    assert inside.sum() >= 3
    assert 0.8 <= coverage["ppc90_province"] <= 1.0
    assert coverage["ppc90_national"] >= 0.75
