"""Rank-paired `reconcile_zones` against sample-k pairing, scored at zones.

Run from the repo root after `draws.py` and `buis.py`:

    uv run python evidence/coherent-reconciliation-review/pairing.py DRAWS OUT

Variants, all using the first 100 zone draws of the pool and the direct
province draws:
- `samplek`: `reconcile_zones` as it is, zone draw k with province draw k.
- `ranked`: zone draws relabelled so the zone block whose total over all
  weeks and zones has rank r goes with the province draw whose total over
  all weeks has rank r, then `reconcile_zones`.
- `buis_joint_t1`: the BUIS zone blocks from `buis.py`, for reference.

Province and national margins of `samplek` and `ranked` equal the direct
forecast by construction.
Zones are scored as in the zone backtest on the reported target: final
counts are mapped to each later release with the zone run's delays
(`reported_samples`) and scored against that release's zone counts.
Per-zone WIS, log WIS and coverage, plus the log energy and log variogram
scores of the zones-by-weeks vector within each province.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import scoringrules as sr

from bdbvhubrenewal.data import (
    load_daily,
    load_daily_zones,
    load_delays,
    load_target,
)
from bdbvhubrenewal.data.zones import province_of, weekly_zone_counts
from bdbvhubrenewal.evaluation import backtest as bt
from bdbvhubrenewal.evaluation import store
from bdbvhubrenewal.evaluation.nowcast import reported_samples
from bdbvhubrenewal.evaluation.scoring import score_sample
from bdbvhubrenewal.forecasters.base import NATIONAL
from bdbvhubrenewal.forecasters.zones import reconcile_zones

N_OUT = 100
COLS = ["location", "target_end_date", "sample", "value"]


def ranked_labels(prov: pd.DataFrame, zones: pd.DataFrame) -> pd.DataFrame:
    """Relabel zone samples so block ranks match province ranks."""
    out = []
    zones = zones.assign(prov=zones["location"].map(province_of))
    for p, grp in zones.groupby("prov"):
        ptot = (
            prov.loc[prov["location"] == p]
            .groupby("sample")["value"]
            .sum()
            .sort_values(kind="stable")
        )
        ztot = grp.groupby("sample")["value"].sum().sort_values(kind="stable")
        if ptot.empty:
            out.append(grp)
            continue
        relabel = dict(zip(ztot.index, ptot.index, strict=True))
        out.append(grp.assign(sample=grp["sample"].map(relabel)))
    return pd.concat(out, ignore_index=True)[COLS]


def main(draws: Path, out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    rd = {"dtype": {"vintage": str}}
    prov = pd.read_csv(draws / "province.csv.gz", **rd)
    zones = pd.read_csv(draws / "zones.csv.gz", **rd)
    zdel = pd.read_csv(draws / "zone_delays.csv.gz", **rd)
    buis = pd.read_csv(draws / "buis_zones.csv.gz", **rd)
    data = "data"
    target, daily = load_target(data), load_daily(data)
    daily_zones = load_daily_zones(data)
    tasks = store.origin_tasks(
        target, load_delays(data), bt.load_config(), daily
    )
    real = target.loc[target["kind"] == "real"]
    releases = sorted(real["vintage"].astype(str).unique())
    cells, mv = [], []
    for origin in sorted(prov["vintage"].unique()):
        later = [v for v in releases if v > origin]
        if not later:
            continue
        task = tasks[origin]
        counts = weekly_zone_counts(daily_zones, origin)
        p = prov.loc[(prov["vintage"] == origin), COLS]
        p = p.loc[p["location"] != NATIONAL]
        z = zones.loc[(zones["vintage"] == origin) & (zones["sample"] <= N_OUT)]
        z = z[COLS]
        active = tuple(sorted(z["location"].unique()))
        variants = {
            "samplek": reconcile_zones(z, p, task, counts, active),
            "ranked": reconcile_zones(
                ranked_labels(p, z), p, task, counts, active
            ),
            "buis_joint_t1": buis.loc[buis["vintage"] == origin, COLS],
        }
        summ = zdel.loc[zdel["vintage"] == origin]
        reported = counts.rename(columns={"total": "reported"})[
            ["location", "target_end_date", "reported"]
        ]
        for name, s in variants.items():
            drawn = reported_samples(
                s, summ.assign(model=name), name, origin, later, daily,
                reported, task.rng(name, "zones"),
            )  # fmt: skip
            for o2, d in drawn.items():
                truth = (
                    weekly_zone_counts(daily_zones, o2)
                    .set_index(["location", "target_end_date"])["total"]
                )
                d = d.loc[d["target_end_date"] <= o2]
                x = d.pivot_table(
                    index="sample",
                    columns=["location", "target_end_date"],
                    values="value",
                )
                y = truth.reindex(x.columns, fill_value=0).to_numpy(float)
                xa = x.to_numpy(float)
                for j, (loc, week) in enumerate(x.columns):
                    cells.append(
                        {"origin": origin, "truth_vintage": o2,
                         "model": name, "location": loc,
                         "target_end_date": week,
                         **score_sample(y[j], xa[:, j])}
                    )  # fmt: skip
                provs = np.array([province_of(c[0]) for c in x.columns])
                for pr in sorted(set(provs)):
                    k = provs == pr
                    if k.sum() < 2:
                        continue
                    lx, ly = np.log1p(xa[:, k]), np.log1p(y[k])
                    mv.append(
                        {"origin": origin, "truth_vintage": o2,
                         "model": name, "province": pr,
                         "n_cells": int(k.sum()),
                         "log_energy": float(sr.es_ensemble(ly, lx)),
                         "log_variogram": float(sr.vs_ensemble(ly, lx))}
                    )  # fmt: skip
    cells = pd.DataFrame(cells)
    mv = pd.DataFrame(mv)
    cells.to_csv(draws / "zone_scores.csv.gz", index=False)
    mv.to_csv(out / "zone_multivariate.csv", index=False)
    cols = ["wis", "log_wis", "coverage_50", "coverage_90"]
    summ = cells.groupby("model")[cols].mean()
    summ["n"] = cells.groupby("model").size()
    case = cells.loc[cells["location"].map(province_of).isin(
        ["CD-IT", "CD-NK", "CD-HU", "CD-TO"]
    )]  # fmt: skip
    by = case.groupby([case["location"].map(province_of), "model"])[cols]
    summ.round(3).to_csv(out / "zone_scores_summary.csv")
    by.mean().round(3).to_csv(out / "zone_scores_by_province.csv")
    mvs = mv.groupby(["province", "model"])[
        ["log_energy", "log_variogram"]
    ].mean()
    mvs.round(3).to_csv(out / "zone_multivariate_summary.csv")
    print(summ.round(3))
    print(by.mean().round(3))
    print(mvs.round(3))
    print(mv.groupby("model")[["log_energy", "log_variogram"]].mean().round(3))


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]))
