"""Markdown tables for REPORT.md."""
import sys
from pathlib import Path

import pandas as pd

out = Path(sys.argv[1]); tab = out.parent / "tables"; hubfile = sys.argv[2]
H = ["k4+", "k3", "k2", "k1", "k0", "f1", "f2"]
MO = {"as reported": 0, "randomwalk": 1, "randomwalknowcast": 2,
      "randomwalknowcast_skellam_rem": 3, "randomwalknowcast_nb": 4,
      "renewal": 5, "renewalsp": 6, "renewalbvd": 7}
LV = ["national", "provinces w/o near-zero", "provinces pooled", "CD-IT", "CD-NK", "CD-HU", "CD-TO", "CD-BU",
      "zones w/o near-zero", "zones pooled"]


def f(x, d=2):
    return "–" if pd.isna(x) else f"{x:.{d}f}"


def cell(r):
    return f"{f(r.wis, 1)} / {f(r.rel_wis)} / {f(r.cov50)} / {f(r.cov90)} / {f(r.bias)} / {int(r.n)}"


def order(t, keys):
    t = t.assign(_m=t.model.map(MO).fillna(50), _l=t.level.map({l: i for i, l in enumerate(LV)}).fillna(99))
    return t.sort_values([*keys, "_m", "model"]).drop(columns=["_m", "_l"])


def flat(t, by):
    t = order(t, [k for k in by if k != "level"] if "level" not in by else ["_l", *[k for k in by if k != "level"]] if False else by)
    rows = t[[*by, "model", "n", "wis", "rel_wis", "rel_to", "cov50", "cov90", "bias"]].copy()
    for c in ("wis",):
        rows[c] = rows[c].map(lambda x: f(x, 1))
    for c in ("rel_wis", "cov50", "cov90", "bias"):
        rows[c] = rows[c].map(f)
    return rows.to_markdown(index=False)


def wide(t, keys):
    t = t.copy(); t["cell"] = t.apply(cell, axis=1)
    p = t.pivot_table(index=[*keys, "model"], columns="h", values="cell", aggfunc="first")
    p = p[[h for h in H if h in p.columns]].fillna("").reset_index()
    p["_m"] = p.model.map(MO).fillna(50)
    return p.sort_values([*keys, "_m", "model"]).drop(columns="_m").to_markdown(index=False)


sec = []
summ = pd.read_csv(tab / "cells-summary.csv")
summ["_l"] = summ.level.map({l: i for i, l in enumerate(LV)})
summ = summ.sort_values(["target", "_l"]).drop(columns="_l")
for tgt in ("next", "complete", "latest"):
    sec.append((f"T1-{tgt}", flat(summ[summ.target == tgt], ["level"])))
byo = pd.read_csv(tab / "cells-by-origin.csv")
for lvl in ["national", "provinces w/o near-zero", "provinces pooled", "CD-IT", "CD-NK", "CD-HU", "CD-TO", "CD-BU"]:
    s = byo[(byo.level == lvl) & byo.model.isin(["as reported", "randomwalk", "randomwalknowcast", "renewal"])]
    sec.append((f"T2-{lvl}", wide(s, ["target", "origin"])))
var = pd.read_csv(tab / "variants.csv")
var = var[var.level.isin(["national", "provinces w/o near-zero", "provinces pooled"]) & var.model.str.startswith("randomwalknowcast")]
sec.append(("T3-variants", flat(var.sort_values(["target", "level", "origin"]), ["target", "level", "origin"])))
vh = pd.read_csv(tab / "variants-by-h.csv")
vh = vh[vh.level.isin(["national", "provinces w/o near-zero"]) & vh.model.str.startswith("randomwalknowcast") & (vh.target == "next")]
sec.append(("T3-variants-next-by-h", wide(vh, ["level", "origin"])))
hub = pd.read_csv(tab / "hub.csv")
sec.append(("T4-hub", flat(hub.sort_values(["target", "level"]), ["target", "level"])))
z = pd.read_csv(tab / "zones.csv")
sec.append(("T5-zones", flat(z.sort_values(["target", "level", "origin"]), ["target", "level", "origin"])))
zh = pd.read_csv(tab / "zones-by-h.csv")
sec.append(("T5-zones-by-h", wide(zh[zh.level == "zones w/o near-zero"], ["target", "origin"])))
c = pd.read_csv(out / "clock.csv")
c = c[c.location.isin(["CD", "CD-IT", "CD-NK", "CD-HU", "CD-TO"]) & (c.model == "randomwalknowcast")]
c["cell"] = c.apply(lambda r: f"{r.q50:.0f} ({r.q5:.0f}–{r.q95:.0f})", axis=1)
p = c.pivot_table(index=["origin", "location", "weeks_back"], columns="clock_shift", values="cell", aggfunc="first")
p.columns = [f"shift {s:+d}" for s in p.columns]
extra = c.groupby(["origin", "location", "weeks_back"])[["reported", "truth_next", "truth_latest"]].first()
p = p.join(extra).reset_index()
sec.append(("T6-clock", p.to_markdown(index=False)))
h = pd.read_csv(hubfile); h = h[h.location.str.count("-") <= 1]
q = h.groupby(["location", "target_end_date"]).value.quantile([0.05, 0.5, 0.95]).unstack().reset_index()
q["cell"] = q.apply(lambda r: f"{r[0.5]:.0f} ({r[0.05]:.0f}–{r[0.95]:.0f})", axis=1)
keep = q.groupby("location")[0.95].max(); q = q[q.location.isin(keep[keep > 0].index)]
w = q.pivot(index="location", columns="target_end_date", values="cell")
w.columns = [x[5:] for x in w.columns]
ordr = ["CD", "CD-IT", "CD-NK", "CD-HU", "CD-TO", "CD-BU"]
w = w.reindex([o for o in ordr if o in w.index] + sorted(set(w.index) - set(ordr)))
t = pd.read_csv("data/target.csv"); t = t[t.vintage == "2026-11-01"]
rep = t.pivot(index="location", columns="target_end_date", values="total").reindex(w.index)
w.insert(0, "reported 10-24, 10-31", rep["2026-10-24"].astype(int).astype(str) + ", " + rep["2026-10-31"].astype(int).astype(str))
sec.append(("T7-submission", w.to_markdown()))
fa = pd.read_csv(out / "fans.csv"); fa = fa[(fa.origin == "2026-11-01") & (fa.model == "randomwalk") & fa.location.isin(w.index)]
fa["cell"] = fa.apply(lambda r: f"{r.q50:.0f} ({r.q5:.0f}–{r.q95:.0f})", axis=1)
g = fa.pivot(index="location", columns="target_end_date", values="cell"); g.columns = [x[5:] for x in g.columns]
g = g.reindex(w.index)[[x for x in w.columns if x in g.columns]]
sec.append(("T7-randomwalk-1101", g.to_markdown()))
with open(tab / "report-tables.md", "w") as fh:
    for name, body in sec:
        fh.write(f"\n<!-- {name} -->\n\n{body}\n")
print("ok")
