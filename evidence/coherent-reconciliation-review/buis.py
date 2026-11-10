"""Trajectory BUIS prototype on randomwalknowcast zone and province samples.

Run from the repo root after `draws.py`:

    uv run python evidence/coherent-reconciliation-review/buis.py DRAWS OUT

For each origin and each province with forecast zones:

1. The zone pool has `N_POOL` joint draws.
   Summing zones gives bottom-up province trajectories u_k.
2. The direct province draws (100) are summarised by a negative binomial
   per week (moment matched, variance floored at mean + 1), or per summary
   for the summary variants.
3. Weights w_k ∝ f(u_k)^tau, normalised.
   ESS = 1 / sum(w_k^2).
4. 100 zone blocks are drawn by systematic resampling with these weights.
   The province is their sum, so zones sum to it exactly and paths are kept.

Provinces with no forecast zones keep their direct forecast in every
method (they have no zone data to combine).
Zones are independent across provinces in this model, as are the direct
provinces, so per-province weighting is the exact joint weighting and no
cross-province reordering is needed.
National is the sum of the provinces.

Methods
-------
- `direct`: the direct province forecast (also the province margin of the
  current `reconcile_zones`).
- `zonesum`: the first 100 pool draws, summed (pure bottom-up).
- `buis_joint_t{tau}`: weight on all 8 weeks jointly, tempered by tau.
- `buis_summary`: weight on two summaries, the reference week (horizon 0)
  and the total of the three forecast weeks.
- `buis_ess`: joint weights with the largest tau in TAUS giving
  ESS >= ESS_TARGET.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import scoringrules as sr
from scipy import stats

from bdbvhubrenewal.data import load_daily, load_delays, load_target
from bdbvhubrenewal.data.zones import province_of
from bdbvhubrenewal.evaluation import backtest as bt
from bdbvhubrenewal.evaluation import store
from bdbvhubrenewal.evaluation.nowcast import (
    complete_weeks,
    release_backtest,
    reported_samples,
)
from bdbvhubrenewal.forecasters.base import NATIONAL

N_OUT = 100
TAUS = (1.0, 0.5, 0.25, 0.1)
ESS_TARGET = 200
CORE = ("CD-IT", "CD-NK", "CD-HU")


def nb_logpmf(x: np.ndarray, draws: np.ndarray) -> np.ndarray:
    """Log pmf at `x` [K, D] of independent NBs fitted to `draws` [S, D]."""
    m = draws.mean(axis=0)
    v = np.maximum(draws.var(axis=0, ddof=1), m) + 1.0
    out = np.zeros(x.shape[0])
    for d in range(x.shape[1]):
        if m[d] == 0 and v[d] <= 1.0:
            out += np.where(
                x[:, d] == 0, 0.0, stats.poisson.logpmf(x[:, d], 1e-3)
            )
            continue
        rd = m[d] ** 2 / (v[d] - m[d])
        out += stats.nbinom.logpmf(x[:, d], rd, rd / (rd + m[d]))
    return out


def weights(logf: np.ndarray, tau: float) -> tuple[np.ndarray, float]:
    lw = tau * logf
    lw = lw - lw.max()
    w = np.exp(lw)
    w /= w.sum()
    return w, float(1.0 / np.sum(w**2))


def systematic(w: np.ndarray, n: int, rng: np.random.Generator) -> np.ndarray:
    u = (rng.random() + np.arange(n)) / n
    idx = np.searchsorted(np.cumsum(w), u)
    idx = np.minimum(idx, len(w) - 1)
    rng.shuffle(idx)
    return idx


def cube(rows: pd.DataFrame, places: list[str], weeks: list[str]) -> np.ndarray:
    """Return values as [sample, place, week]."""
    pv = rows.pivot_table(
        index="sample", columns=["location", "target_end_date"], values="value"
    )
    cols = pd.MultiIndex.from_product([places, weeks])
    arr = pv.reindex(columns=cols).fillna(0.0).to_numpy()
    return arr.reshape(len(pv), len(places), len(weeks))


def frame(arr: np.ndarray, places: list[str], weeks: list[str]) -> pd.DataFrame:
    s, p, w = np.meshgrid(
        np.arange(arr.shape[0]),
        np.arange(len(places)),
        np.arange(len(weeks)),
        indexing="ij",
    )
    return pd.DataFrame(
        {
            "location": np.array(places)[p.ravel()],
            "target_end_date": np.array(weeks)[w.ravel()],
            "sample": s.ravel() + 1,
            "value": arr.ravel(),
        }
    )


def reconcile(prov: pd.DataFrame, zones: pd.DataFrame, origin: str):
    rng = np.random.default_rng(20261110)
    weeks = sorted(prov["target_end_date"].unique())
    provs = sorted(prov.loc[prov["location"] != NATIONAL, "location"].unique())
    direct = cube(prov, provs, weeks)
    zones = zones.assign(prov=zones["location"].map(province_of))
    out = {"direct": direct.copy(), "zonesum": direct.copy()}
    names = [f"buis_joint_t{t:g}" for t in TAUS] + ["buis_summary", "buis_ess"]
    for n in names:
        out[n] = direct.copy()
    ess_rows, zone_out = [], []
    for j, p in enumerate(provs):
        mine = zones.loc[zones["prov"] == p]
        if mine.empty:
            continue
        zl = sorted(mine["location"].unique())
        pool = cube(mine, zl, weeks)  # [K, Z, W]
        u = pool.sum(axis=1)  # [K, W]
        d = direct[:, j, :]
        out["zonesum"][:, j, :] = u[:N_OUT]
        logf = nb_logpmf(u, d)
        for t in TAUS:
            w, ess = weights(logf, t)
            idx = systematic(w, N_OUT, rng)
            out[f"buis_joint_t{t:g}"][:, j, :] = u[idx]
            ess_rows.append(
                {"origin": origin, "location": p, "method": f"joint_t{t:g}",
                 "tau": t, "ess": ess, "ess_frac": ess / len(u)}
            )  # fmt: skip
            if t == 1.0:
                zone_out.append(
                    frame(pool[idx], zl, weeks).assign(model="buis_joint_t1")
                )
        ok = [
            t
            for t in TAUS
            if 1 / np.sum(weights(logf, t)[0] ** 2) >= ESS_TARGET
        ]
        t_ess = ok[0] if ok else TAUS[-1]
        w, ess = weights(logf, t_ess)
        out["buis_ess"][:, j, :] = u[systematic(w, N_OUT, rng)]
        ess_rows.append(
            {"origin": origin, "location": p, "method": "ess",
             "tau": t_ess, "ess": ess, "ess_frac": ess / len(u)}
        )  # fmt: skip
        su = np.stack([u[:, 4], u[:, 5:].sum(axis=1)], axis=1)
        sd = np.stack([d[:, 4], d[:, 5:].sum(axis=1)], axis=1)
        w, ess = weights(nb_logpmf(su, sd), 1.0)
        out["buis_summary"][:, j, :] = u[systematic(w, N_OUT, rng)]
        ess_rows.append(
            {"origin": origin, "location": p, "method": "summary",
             "tau": 1.0, "ess": ess, "ess_frac": ess / len(u)}
        )  # fmt: skip
    frames = []
    for name, arr in out.items():
        f = frame(arr, provs, weeks)
        nat = f.groupby(["target_end_date", "sample"], as_index=False)[
            "value"
        ].sum()
        f = pd.concat([f, nat.assign(location=NATIONAL)], ignore_index=True)
        frames.append(f.assign(model=name, vintage=origin))
    return pd.concat(frames), pd.DataFrame(ess_rows), pd.concat(zone_out)


def multivariate(stored, tasks, daily, target, delays, origins):
    """Log energy and variogram scores, provinces x weeks and national."""
    real = target.loc[target["kind"] == "real"]
    releases = sorted(real["vintage"].astype(str).unique())
    latest = releases[-1]
    done = complete_weeks(target, daily, latest)
    done = done.loc[done["complete"], ["location", "target_end_date"]]
    rows = []
    for origin in origins:
        later = [v for v in releases if v > origin]
        if not later:
            continue
        task = tasks[origin]
        at = real.loc[real["vintage"] == origin]
        reported = at[["location", "target_end_date", "total"]].rename(
            columns={"total": "reported"}
        )
        summ = delays.loc[delays["vintage"] == origin]
        for model, grp in stored.loc[stored["vintage"] == origin].groupby(
            "model"
        ):
            cols = ["location", "target_end_date", "sample", "value"]
            drawn = reported_samples(
                grp[cols], summ.assign(model=model), model, origin, later,
                daily, reported, task.rng(model),
            )  # fmt: skip
            sets = [("reported", o2, s) for o2, s in drawn.items()]
            sets.append(("complete", latest, grp[cols].merge(done)))
            for tgt, o2, s in sets:
                truth = real.loc[real["vintage"] == o2].set_index(
                    ["location", "target_end_date"]
                )["total"]
                for scale, sub in (
                    ("province", s[s["location"] != NATIONAL]),
                    ("national", s[s["location"] == NATIONAL]),
                ):
                    x = sub.pivot_table(
                        index="sample",
                        columns=["location", "target_end_date"],
                        values="value",
                    )
                    keep = [c for c in x.columns if c in truth.index]
                    if len(keep) < 2:
                        continue
                    x = np.log1p(x[keep].to_numpy())
                    y = np.log1p(truth.loc[keep].to_numpy().astype(float))
                    rows.append(
                        {"origin": origin, "target": tgt, "truth_vintage": o2,
                         "scale": scale, "model": model, "n_cells": len(keep),
                         "log_energy": float(sr.es_ensemble(y, x)),
                         "log_variogram": float(sr.vs_ensemble(y, x))}
                    )  # fmt: skip
    return pd.DataFrame(rows)


def main(draws: Path, out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    prov = pd.read_csv(draws / "province.csv.gz", dtype={"vintage": str})
    zones = pd.read_csv(draws / "zones.csv.gz", dtype={"vintage": str})
    delays = pd.read_csv(draws / "delays.csv.gz", dtype={"vintage": str})
    origins = sorted(prov["vintage"].unique())
    stored, ess, zout = [], [], []
    for o in origins:
        s, e, z = reconcile(
            prov.loc[prov["vintage"] == o], zones.loc[zones["vintage"] == o], o
        )
        stored.append(s)
        ess.append(e)
        zout.append(z.assign(vintage=o))
    stored = pd.concat(stored, ignore_index=True)
    ess = pd.concat(ess, ignore_index=True)
    ess.to_csv(out / "ess.csv", index=False)
    pd.concat(zout).to_csv(draws / "buis_zones.csv.gz", index=False)
    stored.to_csv(draws / "stored.csv.gz", index=False)

    data = "data"
    target, daily = load_target(data), load_daily(data)
    config = bt.load_config()
    tasks = store.origin_tasks(target, load_delays(data), config, daily)
    models = sorted(stored["model"].unique())
    learnt = pd.concat([delays.assign(model=m) for m in models])
    res = release_backtest(
        [tasks[o] for o in origins], daily, target, models=(),
        stored=stored, delays=learnt,
    )  # fmt: skip
    sc = res.scores
    sc["scale"] = np.where(sc["location"] == NATIONAL, "national", "province")
    sc.to_csv(draws / "scores.csv.gz", index=False)
    summ = (
        sc.groupby(["target", "scale", "model"])[
            ["wis", "log_wis", "coverage_50", "coverage_90"]
        ]
        .mean()
        .join(sc.groupby(["target", "scale", "model"]).size().rename("n"))
        .round(3)
    )
    summ.to_csv(out / "scores_summary.csv")
    core = (
        sc.loc[sc["location"].isin([*CORE, NATIONAL])]
        .groupby(["target", "location", "model"])[
            ["wis", "log_wis", "coverage_50", "coverage_90"]
        ]
        .mean()
        .round(3)
    )
    core.to_csv(out / "scores_core.csv")
    mv = multivariate(stored, tasks, daily, target, delays, origins)
    mv.to_csv(out / "multivariate.csv", index=False)
    mvs = (
        mv.groupby(["target", "scale", "model"])[
            ["log_energy", "log_variogram"]
        ]
        .mean()
        .round(3)
    )
    mvs.to_csv(out / "multivariate_summary.csv")
    last = stored.loc[
        (stored["vintage"] == origins[-1])
        & stored["location"].isin([*CORE, NATIONAL])
    ]
    q = (
        last.groupby(["model", "location", "target_end_date"])["value"]
        .quantile([0.05, 0.25, 0.5, 0.75, 0.95])
        .unstack()
        .round(0)
    )
    q.columns = ["q05", "q25", "q50", "q75", "q95"]
    q.to_csv(out / f"intervals_{origins[-1]}.csv")
    print(ess.groupby(["method", "origin"])["ess"].median().unstack().round(1))
    print(summ)
    print(mvs)


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]))
