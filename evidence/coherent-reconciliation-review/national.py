"""BUIS at the top level: provinces conditioned on a direct national forecast.

Run from a checkout of the `national-direct` branch (it has
`Settings.national = "direct"`):

    uv run python PATH/national.py OUT

For each origin:

1. A province pool: randomwalknowcast with national as the sum, run with
   `N_POOL` draws. The random walks draw each province independently, so
   `N_JOINT` joint draws are made by picking a pool draw at random for
   each province.
2. The direct national forecast: randomwalknowcast with
   `Settings.national = "direct"` (100 draws).
3. Each joint draw's national trajectory (the province sum) is weighted
   by the direct national's density, a product of weekly NB pmfs
   (all 8 weeks, tempered by tau) or of the reference week and the
   3-week forecast total (`summary`).
4. 100 joint draws are resampled. Provinces and national are coherent and
   paths are kept. Zones would follow from the resampled provinces with
   `reconcile_zones`, so they are not scored here.

Methods:
- `sum`: 100 joint draws, national as their sum (main's method).
- `direct_national`: national from the direct run, provinces as `sum`
  (not coherent; national scored on its own delay).
- `natbuis_t{tau}`, `natbuis_summary`, `natbuis_ess` as in `buis.py`.

Scored on the reported target (each later release, through the delays)
and the complete-week target, at province and national level.
"""

from __future__ import annotations

import dataclasses
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import scoringrules as sr

sys.path.insert(0, str(Path(__file__).parent))
from buis import ESS_TARGET, TAUS, nb_logpmf, systematic, weights  # noqa: E402

from bdbvhubrenewal.data import load_daily, load_delays, load_target
from bdbvhubrenewal.evaluation import backtest as bt
from bdbvhubrenewal.evaluation import store
from bdbvhubrenewal.evaluation.nowcast import complete_weeks, reported_samples
from bdbvhubrenewal.evaluation.scoring import score_sample
from bdbvhubrenewal.forecasters.base import NATIONAL
from bdbvhubrenewal.forecasters.chain import Settings
from bdbvhubrenewal.forecasters.randomwalk import RandomWalkNowcast

ORIGINS = ("2026-10-17", "2026-10-24", "2026-11-01", "2026-11-08")
N_POOL = 2000
N_JOINT = 20000
N_OUT = 100
SHOW = ("CD", "CD-IT", "CD-NK")


def cube(rows, places, weeks):
    pv = rows.pivot_table(
        index="sample", columns=["location", "target_end_date"], values="value"
    )
    cols = pd.MultiIndex.from_product([places, weeks])
    arr = pv.reindex(columns=cols).fillna(0.0).to_numpy()
    return arr.reshape(len(pv), len(places), len(weeks))


def frame(arr, places, weeks):
    s, p, w = np.meshgrid(
        np.arange(arr.shape[0]), np.arange(len(places)),
        np.arange(len(weeks)), indexing="ij",
    )  # fmt: skip
    return pd.DataFrame(
        {
            "location": np.array(places)[p.ravel()],
            "target_end_date": np.array(weeks)[w.ravel()],
            "sample": s.ravel() + 1,
            "value": arr.ravel(),
        }
    )


def with_national(arr, provs, weeks, nat=None):
    f = frame(arr, provs, weeks)
    if nat is None:
        nat = arr.sum(axis=1)
    n = frame(nat[:, None, :], [NATIONAL], weeks)
    return pd.concat([f, n], ignore_index=True)


def methods(task, rng):
    pmodel = RandomWalkNowcast()
    pool_rows = pmodel.forecast(dataclasses.replace(task, n_samples=N_POOL))
    dmodel = RandomWalkNowcast(
        settings=dataclasses.replace(Settings(), national="direct")
    )
    drows = dmodel.forecast(task)
    weeks = sorted(pool_rows["target_end_date"].unique())
    provs = sorted(
        pool_rows.loc[pool_rows["location"] != NATIONAL, "location"].unique()
    )
    pool = cube(pool_rows, provs, weeks)  # [K, P, W]
    pick = rng.integers(0, N_POOL, size=(N_JOINT, len(provs)))
    joint = pool[pick, np.arange(len(provs))[None, :], :]  # [J, P, W]
    u = joint.sum(axis=1)
    dnat = cube(drows.loc[drows["location"] == NATIONAL], [NATIONAL], weeks)
    dnat = dnat[:, 0, :]
    out = {
        "sum": (with_national(joint[:N_OUT], provs, weeks), "sum"),
        "direct_national": (
            with_national(joint[:N_OUT], provs, weeks, dnat[:N_OUT]),
            "direct",
        ),
    }
    logf = nb_logpmf(u, dnat)
    ess = []
    for t in TAUS:
        w, e = weights(logf, t)
        idx = systematic(w, N_OUT, rng)
        out[f"natbuis_t{t:g}"] = (with_national(joint[idx], provs, weeks), "sum")
        ess.append({"method": f"joint_t{t:g}", "tau": t, "ess": e})
    ok = [t for t in TAUS if weights(logf, t)[1] >= ESS_TARGET]
    t_ess = ok[0] if ok else TAUS[-1]
    w, e = weights(logf, t_ess)
    idx = systematic(w, N_OUT, rng)
    out["natbuis_ess"] = (with_national(joint[idx], provs, weeks), "sum")
    ess.append({"method": "ess", "tau": t_ess, "ess": e})
    su = np.stack([u[:, 4], u[:, 5:].sum(axis=1)], axis=1)
    sd = np.stack([dnat[:, 4], dnat[:, 5:].sum(axis=1)], axis=1)
    w, e = weights(nb_logpmf(su, sd), 1.0)
    idx = systematic(w, N_OUT, rng)
    out["natbuis_summary"] = (with_national(joint[idx], provs, weeks), "sum")
    ess.append({"method": "summary", "tau": 1.0, "ess": e})
    summary = dmodel.delay_summary
    return out, pd.DataFrame(ess), summary


def score(origin, task, out, summary, target, daily, real, releases, done):
    later = [v for v in releases if v > origin]
    latest = releases[-1]
    at = real.loc[real["vintage"] == origin]
    reported = at[["location", "target_end_date", "total"]].rename(
        columns={"total": "reported"}
    )
    cells, mv = [], []
    for name, (s, national) in out.items():
        drawn = reported_samples(
            s, summary, name, origin, later, daily, reported,
            task.rng(name), national=national,
        )  # fmt: skip
        sets = [("reported", o2, d) for o2, d in drawn.items()]
        sets.append(("complete", latest, s.merge(done)))
        for tgt, o2, d in sets:
            truth = real.loc[real["vintage"] == o2].set_index(
                ["location", "target_end_date"]
            )["total"]
            x = d.pivot_table(
                index="sample",
                columns=["location", "target_end_date"],
                values="value",
            )
            keep = [c for c in x.columns if c in truth.index]
            xa = x[keep].to_numpy(float)
            y = truth.loc[keep].to_numpy(float)
            for j, (loc, week) in enumerate(keep):
                cells.append(
                    {"origin": origin, "target": tgt, "truth_vintage": o2,
                     "model": name, "location": loc,
                     "target_end_date": week,
                     **score_sample(y[j], xa[:, j])}
                )  # fmt: skip
            nat = np.array([c[0] == NATIONAL for c in keep])
            for scale, k in (("province", ~nat), ("national", nat)):
                if k.sum() < 2:
                    continue
                lx, ly = np.log1p(xa[:, k]), np.log1p(y[k])
                mv.append(
                    {"origin": origin, "target": tgt, "truth_vintage": o2,
                     "scale": scale, "model": name, "n_cells": int(k.sum()),
                     "log_energy": float(sr.es_ensemble(ly, lx)),
                     "log_variogram": float(sr.vs_ensemble(ly, lx))}
                )  # fmt: skip
    return cells, mv


def main(out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    data = "data"
    target, daily = load_target(data), load_daily(data)
    tasks = store.origin_tasks(
        target, load_delays(data), bt.load_config(), daily
    )
    real = target.loc[target["kind"] == "real"]
    releases = sorted(real["vintage"].astype(str).unique())
    done = complete_weeks(target, daily, releases[-1])
    done = done.loc[done["complete"], ["location", "target_end_date"]]
    cells, mv, ess, last = [], [], [], None
    for origin in ORIGINS:
        rng = np.random.default_rng(20261111)
        out, e, summary = methods(tasks[origin], rng)
        ess.append(e.assign(origin=origin))
        print(origin, e.round(0).to_dict("records"), flush=True)
        if origin == releases[-1]:
            last = out
            continue
        c, m = score(
            origin, tasks[origin], out, summary, target, daily, real,
            releases, done,
        )  # fmt: skip
        cells += c
        mv += m
    cells = pd.DataFrame(cells)
    cells["scale"] = np.where(
        cells["location"] == NATIONAL, "national", "province"
    )
    mv = pd.DataFrame(mv)
    pd.concat(ess).to_csv(out_dir / "national_ess.csv", index=False)
    mv.to_csv(out_dir / "national_multivariate.csv", index=False)
    cols = ["wis", "log_wis", "coverage_50", "coverage_90"]
    g = cells.groupby(["target", "scale", "model"])
    summ = g[cols].mean().join(g.size().rename("n")).round(3)
    summ.to_csv(out_dir / "national_scores_summary.csv")
    mvs = mv.groupby(["target", "scale", "model"])[
        ["log_energy", "log_variogram"]
    ].mean()
    mvs.round(3).to_csv(out_dir / "national_multivariate_summary.csv")
    if last is not None:
        q = []
        for name, (s, _) in last.items():
            sub = s.loc[s["location"].isin(SHOW)]
            qq = (
                sub.groupby(["location", "target_end_date"])["value"]
                .quantile([0.05, 0.25, 0.5, 0.75, 0.95])
                .unstack()
                .round(0)
            )
            qq.columns = ["q05", "q25", "q50", "q75", "q95"]
            q.append(qq.assign(model=name))
        pd.concat(q).to_csv(out_dir / f"national_intervals_{releases[-1]}.csv")
    print(summ)
    print(mvs.round(3))


if __name__ == "__main__":
    main(Path(sys.argv[1]))
