"""Tables for REPORT.md from the rwnc-eval2 runs."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from bdbvhubrenewal.evaluation.nowcast import BASELINE

out = Path(sys.argv[1])
tab = out.parent / "tables"
tab.mkdir(exist_ok=True)
REAL = ["CD-IT", "CD-NK", "CD-HU", "CD-TO"]
ORDER = [BASELINE, "randomwalk", "randomwalknowcast",
         "randomwalknowcast_skellam_rem", "randomwalknowcast_nb",
         "renewal", "renewalsp", "renewalbvd"]


def horizon(wb):
    wb = int(wb)
    if wb < 0:
        return f"f{-wb}"
    return f"k{wb}" if wb <= 3 else "k4+"


def prep(s):
    s = s.loc[s["origin"].isin(["2026-10-17", "2026-10-24"])].copy()
    s["h"] = s["weeks_back"].map(horizon)
    return s


def level_rows(s, unit="provinces"):
    nat = s["location"] == "CD"
    parts = [s.loc[nat].assign(level="national"),
             s.loc[~nat].assign(level=f"{unit} pooled"),
             s.loc[~nat & ~s["near_zero"]].assign(level=f"{unit} w/o near-zero")]
    if unit == "provinces":
        for p in [*REAL, "CD-BU"]:
            parts.append(s.loc[s["location"] == p].assign(level=p))
    return pd.concat(parts, ignore_index=True)


def cells(s, by):
    """Mean scores and relative WIS on matched forecasts within each cell."""
    match = ["origin", "truth_vintage", "location", "target_end_date"]
    rows = []
    for key, g in s.groupby(by, sort=False):
        ids = dict(zip(by, key if isinstance(key, tuple) else (key,), strict=True))
        ref = "randomwalk" if "randomwalk" in set(g["model"]) else BASELINE
        base = g.loc[g["model"] == ref, [*match, "wis"]]
        for m, r in g.groupby("model", sort=False):
            pair = r[match + ["wis"]].merge(base, on=match, suffixes=("", "_b"))
            bw = pair["wis_b"].mean()
            rows.append({**ids, "model": m, "n": len(r), "wis": r["wis"].mean(),
                         "rel_wis": pair["wis"].mean() / bw if bw > 0 else np.nan,
                         "n_rel": len(pair), "rel_to": ref,
                         "cov50": r["coverage_50"].mean(),
                         "cov90": r["coverage_90"].mean(),
                         "bias": r["bias"].mean()})
    t = pd.DataFrame(rows)
    t["_o"] = t["model"].map({m: i for i, m in enumerate(ORDER)}).fillna(99)
    return t.sort_values([*by, "_o", "model"]).drop(columns="_o")


def md(t, cols=None):
    t = t.copy()
    for c in ("wis", "rel_wis", "cov50", "cov90", "bias"):
        if c in t:
            t[c] = t[c].map(lambda x: "" if pd.isna(x) else f"{x:.2f}" if c != "wis" else f"{x:.1f}")
    t = t[cols] if cols else t
    return t.to_markdown(index=False)


main = prep(pd.read_csv(out / "main-scores.csv"))
var = [prep(pd.read_csv(out / f"var-{v}-scores.csv")) for v in ("skellam_rem", "nb")]
var = pd.concat(var)
var = var.loc[~var["model"].eq(BASELINE)]
age6 = prep(pd.read_csv(out / "age6-scores.csv"))
zones = prep(pd.read_csv(out / "zones-scores.csv"))

ours = main.loc[main["model"].isin(ORDER)]
lv = level_rows(ours)
full = cells(lv, ["target", "level", "origin", "h"])
full.to_csv(tab / "cells-by-origin.csv", index=False)
pooled_o = cells(lv, ["target", "level", "h"])
pooled_o.to_csv(tab / "cells-origins-pooled.csv", index=False)
summary = cells(lv, ["target", "level"])
summary.to_csv(tab / "cells-summary.csv", index=False)
both = pd.concat([ours, var])
vlv = level_rows(both)
vt = cells(vlv.loc[vlv["model"].str.startswith("randomwalk") | vlv["model"].eq(BASELINE)],
           ["target", "level", "origin", "hgrp"] if False else ["target", "level", "origin"])
vt.to_csv(tab / "variants.csv", index=False)
vth = cells(vlv.loc[vlv["model"].str.startswith("randomwalk") | vlv["model"].eq(BASELINE)],
            ["target", "level", "origin", "h"])
vth.to_csv(tab / "variants-by-h.csv", index=False)
a6 = cells(level_rows(age6.loc[age6["model"].isin(ORDER)]), ["target", "level", "origin"])
a6 = a6.loc[a6["target"] == "complete"]
a6.to_csv(tab / "complete-age6.csv", index=False)
zl = level_rows(zones.loc[zones["location"] != "CD"], unit="zones")
zt = cells(zl, ["target", "level", "origin", "h"])
zt.to_csv(tab / "zones-by-h.csv", index=False)
zs = cells(zl, ["target", "level", "origin"])
zs.to_csv(tab / "zones.csv", index=False)
# hub comparison at origin 10-24, matched cells with every hub model
hub = main.loc[main["origin"] == "2026-10-24"]
hl = level_rows(hub)
ht = cells(hl.loc[hl["level"].isin(["national", "provinces w/o near-zero"])],
           ["target", "level", "h"])
ht.to_csv(tab / "hub-by-h.csv", index=False)
hs = cells(hl.loc[hl["level"].isin(["national", "provinces pooled", "provinces w/o near-zero"])],
           ["target", "level"])
hs.to_csv(tab / "hub.csv", index=False)

C = ["model", "n", "wis", "rel_wis", "cov50", "cov90", "bias"]
with open(tab / "tables.md", "w") as f:
    for name, t, by in [("summary", summary, ["target", "level"]),
                        ("by-origin", full, ["target", "level", "origin", "h"]),
                        ("variants", vt, ["target", "level", "origin"]),
                        ("complete-age6", a6, ["target", "level", "origin"]),
                        ("zones", zs, ["target", "level", "origin"]),
                        ("zones-by-h", zt, ["target", "level", "origin", "h"]),
                        ("hub", hs, ["target", "level"]),
                        ("hub-by-h", ht, ["target", "level", "h"])]:
        f.write(f"\n## {name}\n\n")
        f.write(md(t, by + C) + "\n")
print("ok")
