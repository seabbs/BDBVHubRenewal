"""Couplings of the foundation model's quantile draws (#356), exploration.

    explore_coupling.py OUT [--dates N] [--models M ...] [--variants V ...]

On the German development half (every Nth nowcast date), each model
forecasts each date's states and age-group tasks once per coupling of the
uniform levels its trajectories read Chronos-2's quantiles at. Chronos-2
is called once per input: its quantiles are cached in OUT/q/ by a hash of
the input, so the couplings differ only in `u`. Couplings:
- iid: every series and day its own uniform (today's sampler);
- comonotone: one uniform per sample and series, shared across days;
- ar<rho>: a Gaussian AR(1) copula over days, correlation rho;
- ar<rho>s<r>: the same with exchangeable correlation r across series;
- schaake: iid draws reordered to the ranks of the series' own past
  growth over windows of the horizon's length.
Scores are `evaluation.pairing.variant_scores` with Germany its own level:
CRPS and coverage per cell and of each place's sum over its weeks, and the
energy score (counts and log counts) per level. OUT/scores.csv.gz has one
row per model, coupling, date, level, target and metric.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import ndtr

from bdbvhubrenewal.data.germany import (
    AGE_GROUPS,
    hub_forecasts,
    hub_target,
    hub_triangle,
    nowcast_tasks,
)
from bdbvhubrenewal.data.tidy import write_csv
from bdbvhubrenewal.evaluation import germany as de
from bdbvhubrenewal.evaluation.nowcast import rolling_truth
from bdbvhubrenewal.evaluation.pairing import variant_scores
from bdbvhubrenewal.forecasters import foundation
from bdbvhubrenewal.forecasters.base import get_model

MODELS = ("fm_chronos_daily_ragged", "fm_chronos_weekly_ragged")
VARIANTS = (
    "iid",
    "comonotone",
    "ar0.5",
    "ar0.8",
    "ar0.95",
    "ar0.9s0.3",
    "schaake",
)


def _uniforms(variant, rng, n, v, h, target):
    """Return the uniform levels `[n, v, h]` of one coupling."""
    if variant == "iid":
        return rng.uniform(size=(n, v, h))
    if variant == "comonotone":
        return np.broadcast_to(rng.uniform(size=(n, v, 1)), (n, v, h)).copy()
    if variant.startswith("ar"):
        rho, _, rs = variant[2:].partition("s")
        rho, rs = float(rho), float(rs or 0.0)
        e = rng.standard_normal((n, v, h))
        if rs:
            common = rng.standard_normal((n, 1, h))
            e = np.sqrt(rs) * common + np.sqrt(1 - rs) * e
        z = np.empty_like(e)
        z[..., 0] = e[..., 0]
        for t in range(1, h):
            z[..., t] = rho * z[..., t - 1] + np.sqrt(1 - rho**2) * e[..., t]
        return ndtr(z)
    if variant == "schaake":
        u = np.sort(rng.uniform(size=(n, v, h)), axis=0)
        out = np.empty_like(u)
        for i in range(v):
            x = np.asarray(target[i], dtype=float)
            x = x[np.isfinite(x)]
            starts = np.arange(max(len(x) - h, 0))
            if len(starts) == 0:
                out[:, i] = rng.permutation(u[:, i])
                continue
            pick = rng.choice(starts, size=n, replace=len(starts) < n)
            win = np.stack([x[s : s + h] for s in pick])
            growth = np.log1p(win) - np.log1p(win[:, :1])
            # Ties broken at random, so flat windows give no order.
            ranks = np.argsort(
                np.argsort(growth + 1e-9 * rng.random(growth.shape), axis=0),
                axis=0,
            )
            out[:, i] = np.take_along_axis(u[:, i], ranks, axis=0)
        return out
    raise ValueError(variant)


def sampler(variant, cache, checkpoint=None):
    """Return a Sampler drawing with `variant`, Chronos-2 called once."""

    def sample(groups, horizon, n, seed):
        rng = np.random.default_rng(seed)
        out = []
        for g in groups:
            key = hashlib.sha1(
                np.ascontiguousarray(g["target"], dtype=float).tobytes()
                + str(horizon).encode()
            ).hexdigest()[:16]
            path = cache / f"{key}.npz"
            if path.exists():
                got = np.load(path)
                levels, q = got["levels"], got["q"]
            else:
                pipeline = foundation._chronos(checkpoint)
                levels = np.asarray(pipeline.quantiles, dtype=float)
                (q,) = foundation._predict(pipeline, [g], horizon, levels)
                q = q.float().cpu().numpy()
                np.savez(path, levels=levels, q=q)
            v, h, _ = q.shape
            u = _uniforms(variant, rng, n, v, h, g["target"])
            out.append(
                foundation.quantile_draws(
                    levels, np.broadcast_to(q, (n, *q.shape)), u
                )
            )
        return out

    return sample


def main() -> None:
    """Score every coupling on the chosen dates."""
    p = argparse.ArgumentParser()
    p.add_argument("out", type=Path)
    p.add_argument("--dates", type=int, default=4, help="every Nth date")
    p.add_argument("--models", nargs="+", default=list(MODELS))
    p.add_argument("--variants", nargs="+", default=list(VARIANTS))
    p.add_argument("--config", default="config/germany.toml")
    p.add_argument("--data", default="data/raw/germany")
    a = p.parse_args()
    cache = a.out / "q"
    cache.mkdir(parents=True, exist_ok=True)
    cfg = de.load_germany_config(a.config)
    triangle = hub_triangle(Path(a.data) / Path(cfg.triangle).name)
    dates = de.nowcast_dates(cfg)[:: a.dates]
    levels = dict.fromkeys(AGE_GROUPS, "ages")
    done = a.out / "dates"
    done.mkdir(exist_ok=True)
    for date in dates:
        path = done / f"{date}.csv.gz"
        if path.exists():
            continue
        frames = []
        for task in nowcast_tasks(triangle, date, cfg.n_samples, cfg.seed):
            for name in a.models:
                base = get_model(name)
                for variant in a.variants:
                    model = dataclasses.replace(
                        base, sample=sampler(variant, cache)
                    )
                    s = model.forecast(task).assign(model=name, as_of=np.nan)
                    moved = hub_forecasts(s, task, date)
                    ends = sorted(set(moved["target_end_date"]))
                    truth = rolling_truth(
                        hub_target(triangle, ends), cfg.truth_delay
                    )
                    places = {
                        q: "states"
                        for q in set(moved["location"])
                        if q not in levels and q != "DE"
                    }
                    frames.append(
                        variant_scores(
                            moved,
                            truth,
                            study="germany",
                            model=name,
                            variant=variant,
                            origin=date,
                            levels={**levels, **places},
                            national="DE",
                        )
                    )
        write_csv(pd.concat(frames, ignore_index=True), path)
        print(date, "done", flush=True)
    scores = pd.concat(
        [pd.read_csv(done / f"{d}.csv.gz") for d in dates], ignore_index=True
    )
    write_csv(scores, a.out / "scores.csv.gz")


if __name__ == "__main__":
    main()
