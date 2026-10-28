# Copyright (c) 2026 Sam Abbott. MIT licence, see LICENSE.
"""The step-by-step build of the renewal model, and its variants.

Each rung is a `ModelConfig` fitted with the repository's own NUTS set-up
(`models.renewal.run_nuts`) on two simulated vintages and on the
2026-10-09 real vintage, scored against 2026-10-24 as reported. Per rung
it records the prior predictive, fit time, R-hat, ESS, ESS per second,
divergences, the tree-depth cap share, in-sample PPC coverage, Rt error
and turning point (simulated), CRPS (national, and over a fixed set of
provinces with cases) and 50% and 90% coverage for nowcast weeks (-4 to
-1) and forecast weeks (0 to +3). Persistence is the reference.

Usage::

    uv run python scripts/ladder.py <rung index | persistence | collect>

Each run writes `results/ladder/<rung>.json`; `collect` writes
`results/ladder.csv` and `results/ladder.md`.
"""

from __future__ import annotations

import dataclasses
import datetime as dt
import json
import re
import sys
import time
from pathlib import Path

import numpyro

numpyro.set_host_device_count(2)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from jax import random  # noqa: E402
from numpyro.infer import Predictive  # noqa: E402

from bdbv_hub_renewal import model as pr  # noqa: E402
from bdbv_hub_renewal.data import load_delays, load_locations, load_target  # noqa: E402
from bdbv_hub_renewal.models.base import ForecastTask  # noqa: E402
from bdbv_hub_renewal.models.persistence import Persistence  # noqa: E402
from bdbv_hub_renewal.models.renewal import (  # noqa: E402
    VARIANTS,
    RenewalModel,
    bvd_rows,
    chain_diagnostics,
    parameter_diagnostics,
    ppc_rows,
    run_nuts,
    samples_table,
    vintage_inputs,
)
from bdbv_hub_renewal.ppc import summarise_draws  # noqa: E402
from bdbv_hub_renewal.scoring import crps_sample  # noqa: E402
from bdbv_hub_renewal.simulate import simulate_outbreak  # noqa: E402

OUT = Path(__file__).parents[1] / "results"
SIM_DAYS = (166, 206)
SIM_LOCATIONS = ("CD-IT", "CD-NK", "CD-HU", "CD-TO", "CD-BU", "CD-SK")
REAL_FIT, REAL_TRUTH = "2026-10-09", "2026-10-24"
SAMPLER = {
    "num_warmup": 300,
    "num_samples": 300,
    "num_chains": 2,
    "max_tree_depth": 10,
    "target_accept": 0.9,
}
RUNGS: dict[str, pr.ModelConfig | str] = {
    **pr.model.LADDER,
    "renewal fallback": pr.FALLBACK,
    "renewal learnt kernel": dataclasses.replace(pr.MAIN, coupling="learnt"),
    "renewal future Rt held": dataclasses.replace(pr.MAIN, future_rt="hold"),
    "renewal per-origin importation": dataclasses.replace(
        pr.MAIN, importation="per_origin"
    ),
    "renewal linear Rt knots": dataclasses.replace(
        pr.MAIN, rt_cadence="linear"
    ),
    "renewalbvd": pr.BVD,
    "renewalsp": "renewalsp",
}
"""Rungs in order: the build steps (S5 is `renewal`), then variants."""


def slug(name: str) -> str:
    """Turn a rung name into a file name."""
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")


def weeks(vintage: str, back: int = 4, ahead: int = 3) -> tuple[str, ...]:
    """Saturdays from `back` weeks before the vintage week to `ahead` after."""
    end = pd.Timestamp(vintage).date()
    end += dt.timedelta(days=(5 - end.weekday()) % 7)
    return tuple(
        (end + dt.timedelta(weeks=i)).isoformat()
        for i in range(-back, ahead + 1)
    )


def task_for(target, delays, vintage, locations, kind) -> ForecastTask:
    """Build a forecast task scoring weeks -4 to +3 around the vintage."""
    return ForecastTask(
        vintage=vintage,
        reference_date=pd.Timestamp(vintage).date(),
        kind=kind,
        target=target.loc[target["vintage"] == vintage],
        delays=delays,
        locations=locations,
        target_end_dates=weeks(vintage),
    )


def synthetic_bvd(outbreak, vintage: str, seed: int = 7) -> pd.DataFrame:
    """Build a BVDOutbreakSize-like Rt summary from the simulation truth.

    National and province log Rt on `as_of + 6`, `as_of` four days before
    the vintage, with noise at the published scales (log SD 0.17
    national, 0.25 by province).
    """
    rng = np.random.default_rng(seed)
    as_of = pd.Timestamp(vintage).date() - dt.timedelta(days=4)
    day = pd.Timestamp(as_of + dt.timedelta(days=6))
    par = outbreak.parameters
    on_day = par.loc[par["date"] == day]
    w = on_day["onsets"].to_numpy() + 1e-9
    nat = float(np.log((w * on_day["rt"].to_numpy()).sum() / w.sum()))
    nat = nat + rng.normal(0, 0.17)
    rows = [("national", "", as_of.isoformat(), nat, 0.17, nat, 0.17)]
    for loc, r in zip(on_day["location"], on_day["rt"], strict=True):
        rows.append(
            (loc, loc, as_of.isoformat(), np.log(r) + rng.normal(0, 0.2),
             0.25, nat, 0.17)
        )  # fmt: skip
    return pd.DataFrame(
        rows,
        columns=["group", "members", "as_of", "log_mean", "log_sd",
                 "ref_log_mean", "ref_log_sd"],
    )  # fmt: skip


def scores(samples: pd.DataFrame, truth: pd.Series) -> dict[str, float]:
    """CRPS and 50%/90% coverage, nowcast and forecast weeks apart.

    National uses `CD` rows when the samples have them (a model's own
    national fit), else the sum of the provinces. Provinces are the fixed
    set with any case in the scored weeks, with a missing forecast taken
    as zero, so every rung and persistence score the same rows.
    """
    weeks_ = sorted(set(samples["target_end_date"]))
    n = int(samples["sample"].max())
    now = weeks_[-4]
    if (samples["location"] == "CD").any():
        nat = samples.loc[samples["location"] == "CD"]
    else:
        nat = samples.groupby(["target_end_date", "sample"], as_index=False)[
            "value"
        ].sum()
    nat = {w: g["value"].to_numpy() for w, g in nat.groupby("target_end_date")}
    provinces = sorted(
        {
            loc
            for (loc, w), v in truth.items()
            if loc != "CD" and w in weeks_ and v > 0
        }
    )
    by = {
        k: g["value"].to_numpy()
        for k, g in samples.groupby(["location", "target_end_date"])
    }
    rows = []
    for w in weeks_:
        rows.append(("national", w, truth.get(("CD", w), 0.0), nat[w]))
        for loc in provinces:
            x = by.get((loc, w), np.zeros(n))
            rows.append(("province", w, truth.get((loc, w), 0.0), x))
    out: dict[str, float] = {}
    for level in ("national", "province"):
        sel = [r for r in rows if r[0] == level]
        out[f"crps_{level}"] = float(
            np.mean([crps_sample(y, x) for _, _, y, x in sel])
        )
        for part, ahead in (("now", False), ("ahead", True)):
            part_rows = [r for r in sel if (r[1] >= now) == ahead]
            for q in (50, 90):
                lo, hi = (1 - q / 100) / 2, 1 - (1 - q / 100) / 2
                out[f"cov{q}_{level}_{part}"] = float(
                    np.mean(
                        [
                            np.quantile(x, lo) <= y <= np.quantile(x, hi)
                            for _, _, y, x in part_rows
                        ]
                    )
                )
    return out


def fit_one(task, config, table, bvd=None) -> dict:
    """Fit one configuration and return the pieces the metrics need."""
    data = vintage_inputs(task, config, table, bvd=bvd)
    model = pr.build_model(data, config)
    start = time.perf_counter()
    run_nuts(model, data, {"seed": task.seed, **SAMPLER})
    np.asarray(next(iter(model.mcmc.get_samples().values())))
    seconds = time.perf_counter() - start
    draws = pr.forecast_trajectories(model, data, task.n_samples, seed=1)
    return {"data": data, "model": model, "draws": draws, "seconds": seconds}


def fit_metrics(run) -> dict[str, float]:
    """Sampler diagnostics, time and in-sample PPC coverage of one fit."""
    model = run["model"]
    by_chain = model.mcmc.get_samples(group_by_chain=True)
    trace = {k: np.asarray(by_chain[k]) for k in model.mcmc.last_state.z}
    extra = {
        k: np.asarray(v)
        for k, v in model.mcmc.get_extra_fields(group_by_chain=True).items()
    }
    params = parameter_diagnostics(trace)
    rhat = [p["rhat"] for p in params if p["rhat"] is not None]
    ess = [p["ess_bulk"] for p in params if p["ess_bulk"] is not None]
    chains = chain_diagnostics(extra, SAMPLER["max_tree_depth"])
    observed, draws = ppc_rows(run["data"], run["draws"]["insample"])
    rows = summarise_draws(observed, draws, as_of="2100-01-01")["rows"]
    return {
        "seconds": run["seconds"],
        "max_rhat": max(rhat),
        "min_ess": min(ess),
        "min_ess_per_s": min(ess) / run["seconds"],
        "median_ess_per_s": float(np.median(ess)) / run["seconds"],
        "divergent": float(
            np.mean([c["divergences"] / c["draws"] for c in chains])
        ),
        "depth_hits": float(np.mean([c["tree_depth_share"] for c in chains])),
        "ppc90": float(
            np.mean(
                [r["lower_90"] <= r["observed"] <= r["upper_90"] for r in rows]
            )
        ),
        "ppc50": float(
            np.mean(
                [r["lower_50"] <= r["observed"] <= r["upper_50"] for r in rows]
            )
        ),
    }


def rt_metrics(run, outbreak) -> dict[str, float]:
    """Onset-weighted national Rt error every 14 days, and the turn."""
    data, model = run["data"], run["model"]
    smp = model.mcmc.get_samples()
    n_init = model.latent.n_initialization_points
    rt = np.asarray(smp["latent::rt_subpop"])[:, n_init:]
    w = np.asarray(smp["latent_infections_by_subpop"])[:, n_init:]
    est = np.median((rt * w).sum(-1) / np.maximum(w.sum(-1), 1e-12), axis=0)
    par = outbreak.parameters.assign(w=lambda d: d["onsets"] * d["rt"])
    agg = par.groupby("date")[["w", "onsets"]].sum()
    truth = (agg["w"] / agg["onsets"].clip(lower=1e-9)).to_numpy()
    offset = (data.start_date - outbreak.settings.start).days
    days = np.arange(max(56, offset + 14), offset + data.cutoff + 1, 14)
    below = np.nonzero(est[60:] < 1)[0]
    turn = int(below[0]) + 60 + offset if below.size else np.nan
    true_turn = int(np.nonzero(truth[60:] < 1)[0][0]) + 60
    return {
        "rt_mae": float(np.abs(est[days - offset] - truth[days]).mean()),
        "turn_error": float(abs(turn - true_turn)),
    }


def prior_wild(config, outbreak, table) -> float:
    """Share of prior draws whose 3-week forecast passes 10x the peak."""
    vintage = (
        outbreak.settings.start + dt.timedelta(days=SIM_DAYS[-1])
    ).isoformat()
    target, delays = outbreak.vintage(vintage)
    task = task_for(target, delays, vintage, SIM_LOCATIONS, "simulated")
    data = vintage_inputs(task, config, table)
    model = pr.build_model(data, config)
    prior = Predictive(model.sample, num_samples=200)(
        random.key(0), **pr.model_args(model, data, forecast=True)
    )
    ahead = np.asarray(prior["onsets_final_total"])[:, -3:].sum(-1)
    peak = np.nansum(data.known + data.imputed, axis=1).max()
    return float(np.mean(~np.isfinite(ahead) | (ahead > 10 * peak)))


def per_location(task) -> tuple[pd.DataFrame, dict]:
    """Fit `renewalsp` through the repository adapter and summarise."""
    model = RenewalModel(VARIANTS["renewalsp"], sampler=dict(SAMPLER))
    fit = model.forecast_fit(task)
    samples = pd.concat([fit.samples, fit.national], ignore_index=True)
    seconds = sum(r.seconds for r in fit.runs.values())
    params = [
        p for r in fit.runs.values() for p in parameter_diagnostics(r.trace)
    ]
    chains = [
        c
        for r in fit.runs.values()
        for c in chain_diagnostics(r.extra, SAMPLER["max_tree_depth"])
    ]
    ess = [p["ess_bulk"] for p in params if p["ess_bulk"] is not None]
    summary = {
        "seconds": seconds,
        "max_rhat": max(p["rhat"] for p in params if p["rhat"] is not None),
        "min_ess": min(ess),
        "min_ess_per_s": min(ess) / seconds,
        "median_ess_per_s": float(np.median(ess)) / seconds,
        "divergent": float(
            np.mean([c["divergences"] / c["draws"] for c in chains])
        ),
        "depth_hits": float(np.mean([c["tree_depth_share"] for c in chains])),
        "n_fits": len(fit.runs),
    }
    return samples, summary


def run_rung(name, config, outbreak, table, real) -> dict:
    """Fit a rung on the simulated and real tasks and score it."""
    out: dict = {"rung": name}
    single = config == "renewalsp"
    if not single:
        out["prior_wild"] = prior_wild(config, outbreak, table)
    truth = outbreak.truth().set_index(["location", "target_end_date"])["truth"]
    sims = []
    for day in SIM_DAYS:
        vintage = (outbreak.settings.start + dt.timedelta(days=day)).isoformat()
        target, delays = outbreak.vintage(vintage)
        task = task_for(target, delays, vintage, SIM_LOCATIONS, "simulated")
        if single:
            samples, res = per_location(task)
        else:
            bvd = synthetic_bvd(outbreak, vintage) if config.bvd else None
            run = fit_one(task, config, table, bvd)
            samples, _ = samples_table(task, run["data"], run["draws"])
            if config.national:
                samples = national_samples(task, run)
            res = fit_metrics(run) | rt_metrics(run, outbreak)
        res |= scores(samples, truth)
        sims.append(res)
        print(
            name,
            "sim",
            day,
            {k: round(v, 3) for k, v in res.items()},
            flush=True,
        )
    for key in sims[0]:
        out[f"sim_{key}"] = float(np.nanmean([s[key] for s in sims]))
    target, delays, locations = real
    task = task_for(
        target, delays.loc[delays["vintage"] == REAL_FIT], REAL_FIT,
        locations, "real",
    )  # fmt: skip
    later = target.loc[target["vintage"] == REAL_TRUTH].set_index(
        ["location", "target_end_date"]
    )["total"]
    if single:
        samples, res = per_location(task)
    else:
        bvd = bvd_rows(task) if config.bvd else None
        if config.bvd:
            out["real_bvd_as_of"] = (
                None if bvd is None else bvd["as_of"].iloc[0]
            )
        run = fit_one(task, config, table, bvd)
        samples, _ = samples_table(task, run["data"], run["draws"])
        if config.national:
            samples = national_samples(task, run)
        res = fit_metrics(run) | {"n_active": len(run["data"].names)}
    res |= scores(samples, later)
    print(name, "real", {k: round(v, 3) for k, v in res.items()}, flush=True)
    out |= {f"real_{k}": v for k, v in res.items()}
    return out


def national_samples(task, run) -> pd.DataFrame:
    """Sample table of a national-only fit, as location `CD`."""
    data, nat = run["data"], run["draws"]["national"]
    first = data.start_date + dt.timedelta(days=6)
    rows = []
    for date in task.target_end_dates:
        i = (pd.Timestamp(date).date() - first).days // 7
        for k in range(nat.shape[0]):
            value = float(nat[k, i]) if i >= 0 else 0.0
            rows.append(("CD", date, k + 1, value))
    return pd.DataFrame(
        rows, columns=["location", "target_end_date", "sample", "value"]
    )


def persistence_row(outbreak, real) -> dict:
    """Score persistence on the same tasks."""
    target, delays, locations = real
    task = task_for(
        target, delays.loc[delays["vintage"] == REAL_FIT], REAL_FIT,
        locations, "real",
    )  # fmt: skip
    later = target.loc[target["vintage"] == REAL_TRUTH].set_index(
        ["location", "target_end_date"]
    )["total"]
    out = {"rung": "persistence"}
    out |= {
        f"real_{k}": v
        for k, v in scores(Persistence().forecast(task), later).items()
    }
    truth = outbreak.truth().set_index(["location", "target_end_date"])["truth"]
    sims = []
    for day in SIM_DAYS:
        vintage = (outbreak.settings.start + dt.timedelta(days=day)).isoformat()
        t, d = outbreak.vintage(vintage)
        task = task_for(t, d, vintage, SIM_LOCATIONS, "simulated")
        sims.append(scores(Persistence().forecast(task), truth))
    out |= {f"sim_{k}": float(np.mean([s[k] for s in sims])) for k in sims[0]}
    return out


def main(args: list[str]) -> None:
    """Run one rung, persistence, or collect the table."""
    out = OUT / "ladder"
    out.mkdir(parents=True, exist_ok=True)
    if args and args[0] == "collect":
        rows = [json.loads(p.read_text()) for p in sorted(out.glob("*.json"))]
        order = [*RUNGS, "persistence"]
        rows.sort(key=lambda r: order.index(r["rung"]))
        frame = pd.DataFrame(rows)
        frame.to_csv(OUT / "ladder.csv", index=False)
        (OUT / "ladder.md").write_text(
            frame.round(3).to_markdown(index=False) + "\n"
        )
        print(frame.round(3).to_string())
        return
    outbreak = simulate_outbreak()
    table = load_locations()
    target, delays = load_target(), load_delays()
    locations = tuple(sorted(set(target["location"]) - {"CD"}))
    real = (target, delays, locations)
    if args[0] == "persistence":
        row = persistence_row(outbreak, real)
    else:
        name, config = list(RUNGS.items())[int(args[0])]
        start = time.perf_counter()
        row = run_rung(name, config, outbreak, table, real)
        row["rung_seconds"] = time.perf_counter() - start
    (out / f"{slug(row['rung'])}.json").write_text(
        json.dumps(row, indent=1, default=float)
    )


if __name__ == "__main__":
    main(sys.argv[1:])
