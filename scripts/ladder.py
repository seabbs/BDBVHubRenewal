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

import jax  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from jax import random  # noqa: E402
from numpyro.infer import Predictive  # noqa: E402

from bdbv_hub_renewal import model as pr  # noqa: E402
from bdbv_hub_renewal.data import load_delays, load_locations, load_target  # noqa: E402
from bdbv_hub_renewal.data.simulate import simulate_outbreak  # noqa: E402
from bdbv_hub_renewal.evaluation.ppc import summarise_draws  # noqa: E402
from bdbv_hub_renewal.evaluation.scoring import crps_sample  # noqa: E402
from bdbv_hub_renewal.forecasters import renewal as adapter  # noqa: E402
from bdbv_hub_renewal.forecasters.base import ForecastTask  # noqa: E402
from bdbv_hub_renewal.forecasters.persistence import Persistence  # noqa: E402
from bdbv_hub_renewal.forecasters.renewal import (  # noqa: E402
    VARIANTS,
    bvd_rows,
    chain_diagnostics,
    parameter_diagnostics,
    ppc_rows,
    samples_table,
    vintage_inputs,
)

OUT = Path(__file__).parents[1] / "results"


def run_nuts(model, data, sampler):
    """`forecasters.renewal.run_nuts` with vectorised chains.

    Parallel (pmap) chains leak memory on Linux CPU (about 0.7 GB per
    30 s on archie), so the ladder runs chains vectorised until the
    adapter does. It also replaces the adapter's own function, so the
    per-location rung runs the same way.
    """
    from jax import random as jrandom
    from numpyro.infer import MCMC

    # Each fit compiles new executables; clearing them first stops a run
    # of many fits (the per-location rung) exhausting LLVM's memory.
    jax.clear_caches()
    args = pr.model_args(model, data)
    model.validate_data(
        **{k: v for k, v in args.items() if k != "population_size"}
    )
    model.mcmc = MCMC(
        adapter.nuts_kernel(model.sample, sampler),
        num_warmup=int(sampler["num_warmup"]),
        num_samples=int(sampler["num_samples"]),
        num_chains=int(sampler["num_chains"]),
        chain_method="vectorized",
        progress_bar=False,
    )
    model.mcmc.run(
        jrandom.key(int(sampler["seed"])),
        extra_fields=pr.model.EXTRA_FIELDS,
        **args,
    )
    return model


adapter.run_nuts = run_nuts
SIM_DAYS = (166, 206)
SIM_LOCATIONS = ("CD-IT", "CD-NK", "CD-HU", "CD-TO", "CD-BU", "CD-SK")
REAL_ORIGINS = ("2026-08-18", "2026-10-09", "2026-10-17")
REAL_TRUTH = "2026-10-24"
COMPLETE_K = 3
"""Weeks ending at least this many weeks before the truth vintage count as
complete, as `data.completeness.complete_weeks` (`complete_after_weeks`);
the complete-weeks scores use only those."""
SP_MIN_CASES = 10
"""renewalsp fits only locations with at least this many cases."""
SAMPLER = {
    "num_warmup": 300,
    "num_samples": 300,
    "num_chains": 2,
    "max_tree_depth": 10,
    "target_accept": 0.9,
}
RUNGS: dict[str, pr.ModelConfig | str] = {
    **pr.model.LADDER,
    "S6 + report day-of-week": dataclasses.replace(pr.MAIN, report_dow=True),
    "renewal case-weighted Rt": dataclasses.replace(
        pr.MAIN, rt_weighting="cases"
    ),
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

DESCRIPTIONS = {
    "persistence": "The last complete week carried forward, as the reference.",
    "S0 national renewal": "One renewal on the national total, no truncation.",
    "S1 + provinces (pooled Rt)": "A renewal per province with case, Rt a "
    "national walk plus pooled deviations.",
    "S2 + truncation (hub delay)": "Right truncation of the total by the "
    "hub's own delay.",
    "S3 + delays, missingness, imputation": "Known and imputed onsets as two "
    "streams, with time-varying delays and missingness and the hub's "
    "imputation modelled.",
    "S4 + coupling and importation": "Gravity importation between provinces "
    "and an importation-only part for provinces with no cases.",
    "S5 + depletion (main)": "Susceptible depletion; this is `renewal`.",
    "S6 + report day-of-week": "A sum-to-zero report day-of-week effect "
    "on the delay hazard.",
    "renewal case-weighted Rt": "Province Rt deviations sum to zero "
    "weighted by cases, so the national walk is the case-weighted Rt.",
    "renewal learnt kernel": "Gravity decay and destination weights learnt.",
    "renewal future Rt held": "Rt held at its last value past the data.",
    "renewal per-origin importation": "One importation share per origin, "
    "partially pooled.",
    "renewal linear Rt knots": "Weekly Rt knots joined linearly.",
    "renewalbvd": "BVDOutbreakSize Rt as a tempered pseudo-observation.",
    "renewalsp": "An independent single-patch renewal per location, the "
    "nation included.",
}
"""One sentence per rung: what it adds."""

DECISIONS: dict[str, str] = {
    "persistence": "Reference only.",
    "S0 national renewal": "Dropped as a candidate: national scores no "
    "better than persistence, and it gives no provinces.",
    "S1 + provinces (pooled Rt)": "Kept: province forecasts and pooling, "
    "with national scores as S0.",
    "S2 + truncation (hub delay)": "Kept: the largest gain, national CRPS "
    "down by about three quarters on simulated data and by a third on real.",
    "S3 + delays, missingness, imputation": "Kept on the fuller-model rule: "
    "scores level with S2, and it models the imputed stream and changing "
    "delays the target depends on.",
    "S4 + coupling and importation": "Kept on the fuller-model rule: scores "
    "level, it learns across provinces and forecasts spread to new ones.",
    "S5 + depletion (main)": "Kept and chosen as `renewal`: fewer wild prior "
    "draws (0.44 to 0.32) and the best province scores of the build.",
    "S6 + report day-of-week": "An option, not the default: no score gain, "
    "slower and with a higher R-hat.",
    "renewal learnt kernel": "An option for the backtest: scores slightly "
    "worse than gravity and it mixes no better.",
    "renewal future Rt held": "An option for the backtest: level with the "
    "walk, slightly better on the real vintage.",
    "renewal per-origin importation": "An option for the backtest: level "
    "with one shared share; its lower real depth-cap share is one chain "
    "capped and one not, not better mixing.",
    "renewal linear Rt knots": "Dropped: worse on the real vintage and no "
    "gain on simulated data.",
    "renewalbvd": "Shadow candidate: best on simulated data with a "
    "truth-based BVD summary, worse on the one real vintage.",
    "renewalsp": "Comparison variant: province scores level with renewal "
    "on the real vintage, but its own national fit did not converge (R-hat "
    "2.2) and scores far worse.",
}
"""One sentence per rung: kept, dropped or an option, and why. Filled in
from the results."""


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


def scores(
    samples: pd.DataFrame,
    truth: pd.Series,
    truth_as_of: str | None = None,
) -> dict[str, float]:
    """CRPS and 50%/90% coverage, nowcast and forecast weeks apart.

    Only weeks present in the truth are scored. National uses `CD` rows
    when the samples have them (a model's own national fit), else the sum
    of the provinces. Provinces are the fixed set with any case in the
    scored weeks, with a missing forecast taken as zero, so every rung and
    persistence score the same rows. With `truth_as_of`, CRPS is also
    given over complete weeks only: those ending at least `COMPLETE_K`
    weeks before it.
    """
    weeks_ = sorted(set(samples["target_end_date"]))
    n = int(samples["sample"].max())
    now = weeks_[-4]
    in_truth = {w for (_, w) in truth.index}
    scored = [w for w in weeks_ if w in in_truth]
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
            if loc != "CD" and w in scored and v > 0
        }
    )
    by = {
        k: g["value"].to_numpy()
        for k, g in samples.groupby(["location", "target_end_date"])
    }
    rows = []
    for w in scored:
        rows.append(("national", w, truth.get(("CD", w), 0.0), nat[w]))
        for loc in provinces:
            x = by.get((loc, w), np.zeros(n))
            rows.append(("province", w, truth.get((loc, w), 0.0), x))
    complete = None
    if truth_as_of is not None:
        complete = (
            (pd.Timestamp(truth_as_of) - pd.Timedelta(weeks=COMPLETE_K))
            .date()
            .isoformat()
        )
    out: dict[str, float] = {}
    for level in ("national", "province"):
        sel = [r for r in rows if r[0] == level]
        out[f"crps_{level}"] = float(
            np.mean([crps_sample(y, x) for _, _, y, x in sel])
        )
        if complete is not None:
            done = [r for r in sel if r[1] <= complete]
            out[f"crps_{level}_complete"] = (
                float(np.mean([crps_sample(y, x) for _, _, y, x in done]))
                if done
                else float("nan")
            )
        for part, ahead in (("now", False), ("ahead", True)):
            part_rows = [r for r in sel if (r[1] >= now) == ahead]
            for q in (50, 90):
                lo, hi = (1 - q / 100) / 2, 1 - (1 - q / 100) / 2
                out[f"cov{q}_{level}_{part}"] = (
                    float(
                        np.mean(
                            [
                                np.quantile(x, lo) <= y <= np.quantile(x, hi)
                                for _, _, y, x in part_rows
                            ]
                        )
                    )
                    if part_rows
                    else float("nan")
                )
    return out


def fit_one(task, config, table, bvd=None, as_of: str | None = None) -> dict:
    """Fit one configuration and return the pieces the metrics need.

    With `as_of`, a later truth vintage, the draws also hold the counts as
    reported by it (`forecast_trajectories(..., as_of=...)`).
    """
    data = vintage_inputs(task, config, table, bvd=bvd)
    model = pr.build_model(data, config)
    start = time.perf_counter()
    run_nuts(model, data, {"seed": task.seed, **SAMPLER})
    np.asarray(next(iter(model.mcmc.get_samples().values())))
    seconds = time.perf_counter() - start
    cuts = ()
    if as_of is not None:
        cuts = ((pd.Timestamp(as_of).date() - data.start_date).days,)
    draws = pr.forecast_trajectories(
        model, data, task.n_samples, seed=1, as_of=cuts
    )
    return {"data": data, "model": model, "draws": draws, "seconds": seconds}


def as_reported_samples(task, run, as_of: str) -> pd.DataFrame:
    """Sample table of the counts as reported by the truth vintage.

    Like for like with scoring against `as_of` as reported: the draws are
    the counts the model expects that vintage to show.
    """
    data = run["data"]
    cut = (pd.Timestamp(as_of).date() - data.start_date).days
    draws = run["draws"]["as_reported"][cut]
    if data.names == ["CD"]:
        return national_samples(task, run, draws["national"])
    samples, _ = samples_table(task, data, draws)
    return samples


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
    worst = max(
        (p for p in params if p["rhat"] is not None), key=lambda p: p["rhat"]
    )
    per_chain = {}
    for c in chains:
        per_chain[f"depth_share_chain{c['chain']}"] = c.get("tree_depth_share")
        per_chain[f"step_size_chain{c['chain']}"] = c.get("step_size")
    return per_chain | {
        "worst_rhat_param": f"{worst['name']}[{worst['index']}]",
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


def per_location(
    task, table, as_of: str | None = None
) -> tuple[pd.DataFrame, dict]:
    """Fit `renewalsp`: one single-patch fit per location, national its own.

    Locations with at least `SP_MIN_CASES` cases are fitted, the nation
    included; the rest get zeros (their reported counts are below the
    threshold). With `as_of`, samples are the counts as reported by it.
    """
    config = adapter.model_config(VARIANTS["renewalsp"])
    rows = task.target.loc[task.target["vintage"] == task.vintage]
    cases = rows.groupby("location")["total"].sum()
    frames, fits = [], []
    for loc in (*task.locations, "CD"):
        if cases.get(loc, 0) < SP_MIN_CASES:
            continue
        sub = dataclasses.replace(task, locations=(loc,))
        run = fit_one(sub, config, table, as_of=as_of)
        nat = (
            run["draws"]["as_reported"][
                next(iter(run["draws"]["as_reported"]))
            ]["national"]
            if as_of is not None
            else run["draws"]["national"]
        )
        part = national_samples(sub, run, nat).assign(location=loc)
        frames.append(part)
        fits.append((loc, fit_metrics(run)))
    samples = pd.concat(frames, ignore_index=True)
    seconds = sum(f["seconds"] for _, f in fits)
    worst = max(fits, key=lambda f: f[1]["max_rhat"])
    summary = {
        "seconds": seconds,
        "max_rhat": worst[1]["max_rhat"],
        "worst_rhat_param": f"{worst[0]}:{worst[1]['worst_rhat_param']}",
        "min_ess": min(f["min_ess"] for _, f in fits),
        "min_ess_per_s": min(f["min_ess"] for _, f in fits) / seconds,
        "divergent": float(np.mean([f["divergent"] for _, f in fits])),
        "depth_hits": float(np.mean([f["depth_hits"] for _, f in fits])),
        "n_fits": len(fits),
    }
    return samples, summary


def real_scores(name, config, table, real, out) -> dict:
    """Fit each real origin, score as reported by `REAL_TRUTH`, average."""
    target, delays, locations = real
    later = target.loc[target["vintage"] == REAL_TRUTH].set_index(
        ["location", "target_end_date"]
    )["total"]
    results = []
    for origin in REAL_ORIGINS:
        jax.clear_caches()
        task = task_for(
            target, delays.loc[delays["vintage"] == origin], origin,
            locations, "real",
        )  # fmt: skip
        if config == "renewalsp":
            samples, res = per_location(task, table, as_of=REAL_TRUTH)
        else:
            bvd = bvd_rows(task) if config.bvd else None
            if config.bvd:
                out[f"real_bvd_as_of_{origin}"] = (
                    None if bvd is None else bvd["as_of"].iloc[0]
                )
            run = fit_one(task, config, table, bvd, as_of=REAL_TRUTH)
            samples = as_reported_samples(task, run, REAL_TRUTH)
            res = fit_metrics(run) | {"n_active": len(run["data"].names)}
        res |= scores(samples, later, REAL_TRUTH)
        print(name, "real", origin, res, flush=True)
        results.append((origin, res))
    return summarise_origins(results)


def summarise_origins(results) -> dict:
    """Average numeric results over origins, keep per-origin CRPS."""
    out: dict = {}
    keys = {k for _, r in results for k in r}
    for k in sorted(keys):
        values = [r.get(k) for _, r in results]
        numbers = [v for v in values if isinstance(v, (int, float))]
        if k == "max_rhat" and numbers:
            out[f"real_{k}"] = max(numbers)
        elif k == "worst_rhat_param":
            worst = max(results, key=lambda r: r[1].get("max_rhat", 0))
            out[f"real_{k}"] = f"{worst[0]}:{worst[1].get(k)}"
        elif numbers and len(numbers) == len(values):
            out[f"real_{k}"] = float(np.nanmean(numbers))
    for origin, r in results:
        tag = origin.replace("-", "")
        for k in ("crps_national", "crps_province", "seconds", "max_rhat"):
            if k in r:
                out[f"real_{k}_{tag}"] = r[k]
    return out


def run_rung(name, config, outbreak, table, real) -> dict:
    """Fit a rung on the simulated and real tasks and score it."""
    out: dict = {"rung": name}
    single = config == "renewalsp"
    if not single:
        out["prior_wild"] = prior_wild(config, outbreak, table)
    truth = outbreak.truth().set_index(["location", "target_end_date"])["truth"]
    sims = []
    for day in SIM_DAYS:
        # Each fit compiles new executables; without clearing, the
        # per-location rung runs out of mappable memory in LLVM.
        jax.clear_caches()
        vintage = (outbreak.settings.start + dt.timedelta(days=day)).isoformat()
        target, delays = outbreak.vintage(vintage)
        task = task_for(target, delays, vintage, SIM_LOCATIONS, "simulated")
        if single:
            samples, res = per_location(task, table)
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
    out |= real_scores(name, config, table, real, out)
    return out


def national_samples(task, run, nat=None) -> pd.DataFrame:
    """Sample table of a national-only fit, as location `CD`."""
    data = run["data"]
    nat = run["draws"]["national"] if nat is None else nat
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
    later = target.loc[target["vintage"] == REAL_TRUTH].set_index(
        ["location", "target_end_date"]
    )["total"]
    out = {"rung": "persistence"}
    results = []
    for origin in REAL_ORIGINS:
        task = task_for(
            target, delays.loc[delays["vintage"] == origin], origin,
            locations, "real",
        )  # fmt: skip
        results.append(
            (origin, scores(Persistence().forecast(task), later, REAL_TRUTH))
        )
    out |= summarise_origins(results)
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
        order = ["persistence", *RUNGS]
        rows.sort(key=lambda r: order.index(r["rung"]))
        frame = pd.DataFrame(rows)
        frame.insert(0, "step", [order.index(r) for r in frame["rung"]])
        frame.insert(2, "description", frame["rung"].map(DESCRIPTIONS))
        frame.insert(3, "decision", frame["rung"].map(DECISIONS))
        frame.to_csv(OUT / "ladder.csv", index=False, na_rep="")
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
    elif "--real-only" in args:
        # Keep the simulated results and refit the real origins only.
        name, config = list(RUNGS.items())[int(args[0])]
        path = out / f"{slug(name)}.json"
        old = json.loads(path.read_text()) if path.exists() else {"rung": name}
        start = time.perf_counter()
        row = {k: v for k, v in old.items() if not k.startswith("real_")}
        row |= real_scores(name, config, table, real, row)
        row["real_rung_seconds"] = time.perf_counter() - start
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
