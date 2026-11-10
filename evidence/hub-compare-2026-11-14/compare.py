"""Put our 2026-11-14 forecasts beside the hub's and flag the gaps.

Usage: compare.py <S> <evidence dir>

Inputs:
- `<E>/hub_quantiles.csv` from `hub_quantiles.py`;
- randomwalknowcast and randomwalk quantiles from the plausibility
  evidence (`S/ours/plaus_quantiles.csv.gz`, main 96ac128d; the source
  of both models is unchanged on main af48b070);
- `S/ours/pprandomwalknowcast.csv`, run on main af48b070 with
  `nowcast --vintage 2026-11-08 --reference-date 2026-11-14`;
- `S/ours/<model>_zones.csv` for renewal, renewalsp and renewalbvd, made
  by `store_zones.py` from the fit-store fits at b12df214;
- the 2026-11-08 release in `data/` (target.csv, health_zones.csv.gz).

Rules, fixed before looking at the numbers:
- The hub ensemble is the median, quantile by quantile, of the merged and
  PR #114 models other than BDBVHub-baseline, where two or more give the
  place and week. The baseline is shown but kept out of the ensemble and
  of the hub range.
- The hub range of a place and week is [lowest hub q05, highest hub q95]
  over the same models.
- `median_outside`: our median is outside the hub range.
- `no_overlap`: our [q05, q95] does not meet the hub range.
- `tail_x2`: our q95 is more than twice the highest hub q95.
- `not_in_hub`: no hub model (bar the baseline) gives the place.
- Data check: for weeks ending on or before 2026-10-24 (two weeks or more
  before the release's last week) each model's median is set against the
  count reported in the 2026-11-08 release. Later weeks are still filling.

Writes `quantiles.csv`, `table.csv` (median [q05, q95] wide), `flags.csv`,
`data_check.csv`.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

S, E = Path(sys.argv[1]), Path(sys.argv[2])
Q = ["q05", "q25", "q50", "q75", "q95"]
LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
WEEKS = [
    "2026-10-17", "2026-10-24", "2026-10-31", "2026-11-07",
    "2026-11-14", "2026-11-21", "2026-11-28", "2026-12-05",
]
OURS = [
    "randomwalknowcast", "pprandomwalknowcast", "randomwalk",
    "renewal", "renewalsp", "renewalbvd",
]


def from_samples(d, model):
    g = d.groupby(["location", "target_end_date"]).value
    q = pd.concat(
        {k: g.quantile(v) for k, v in zip(Q, LEVELS, strict=True)}, axis=1
    ).reset_index()
    q["model"] = model
    return q


hub = pd.read_csv(E / "hub_quantiles.csv")
hub["group"] = np.where(hub.model == "BDBVHub-baseline", "baseline", "hub")
ours = []
p = pd.read_csv(S / "ours" / "plaus_quantiles.csv.gz")
ours.append(p[p.model.isin(["randomwalknowcast", "randomwalk"])])
if (S / "ours" / "pprandomwalknowcast.csv").exists():
    pp = pd.read_csv(S / "ours" / "pprandomwalknowcast.csv")
    pp = pp[pp.output_type == "sample"]
    ours.append(from_samples(pp, "pprandomwalknowcast"))
for m in ["renewal", "renewalsp", "renewalbvd"]:
    ours.append(from_samples(pd.read_csv(S / "ours" / f"{m}_zones.csv"), m))
ours = pd.concat(ours)
ours["group"] = "ours"
ours["status"] = "ours"

hubm = hub[hub.group == "hub"]
ens = hubm.groupby(["location", "target_end_date"])
ensq = ens[Q].median()
ensq["n_models"] = ens.model.nunique()
ensq = ensq[ensq.n_models >= 2].reset_index()
ensq["model"] = "hub-ensemble"
ensq["group"] = "ensemble"
ensq["status"] = "derived"

cols = ["group", "model", "status", "location", "target_end_date", *Q]
allq = pd.concat([hub[cols], ensq[cols], ours[cols]])
allq = allq[allq.target_end_date.isin(WEEKS)]

# Places: the nation, provinces with a case in the release or given by a
# hub model, and zones given by a hub model or with a case.
t = pd.read_csv("data/target.csv")
t = t[t.vintage == "2026-11-08"]
z = pd.read_csv("data/health_zones.csv.gz")
z = z[z.vintage == "2026-11-08"]
data = pd.concat([
    t[["location", "location_name", "target_end_date", "total"]],
    z[["location", "location_name", "target_end_date", "total"]],
])
names = dict(zip(data.location, data.location_name, strict=False))
active = set(data.groupby("location").total.sum().loc[lambda x: x > 0].index)
places = sorted(active | set(hubm.location))
allq = allq[allq.location.isin(places)]
allq["location_name"] = allq.location.map(names).fillna("")
allq.to_csv(E / "quantiles.csv", index=False, float_format="%.1f")

# Wide table: one row per place, week and model.
allq["cell"] = allq.apply(
    lambda r: f"{r.q50:.0f} [{r.q05:.0f}, {r.q95:.0f}]", axis=1
)
order = {m: i for i, m in enumerate(
    ["hub-ensemble", *sorted(hubm.model.unique()), "BDBVHub-baseline", *OURS]
)}
table = allq.pivot_table(
    index=["location", "location_name", "target_end_date"],
    columns="model", values="cell", aggfunc="first",
)
table = table[sorted(table.columns, key=order.get)]
table.to_csv(E / "table.csv")

# Flags: each of our models against the hub range.
rng = hubm.groupby(["location", "target_end_date"]).agg(
    hub_lo=("q05", "min"), hub_hi=("q95", "max"),
    hub_med_lo=("q50", "min"), hub_med_hi=("q50", "max"),
    n_hub=("model", "nunique"),
).reset_index()
o = allq[allq.group == "ours"].merge(
    rng, on=["location", "target_end_date"], how="left"
)
o["not_in_hub"] = o.n_hub.isna()
o["median_outside"] = (o.q50 < o.hub_lo) | (o.q50 > o.hub_hi)
o["no_overlap"] = (o.q05 > o.hub_hi) | (o.q95 < o.hub_lo)
o["tail_x2"] = o.q95 > 2 * o.hub_hi
o["position"] = np.select(
    [o.not_in_hub, o.q50 > o.hub_hi, o.q50 < o.hub_lo,
     o.q50 > o.hub_med_hi, o.q50 < o.hub_med_lo],
    ["not in hub", "above hub range", "below hub range",
     "above all hub medians", "below all hub medians"],
    "within hub medians",
)
o = o.drop(columns=["cell", "group", "status"])
o.to_csv(E / "flags.csv", index=False, float_format="%.1f")

# Data check on the settled weeks.
obs = data.rename(columns={"total": "reported"})[
    ["location", "target_end_date", "reported"]
]
settled = allq[allq.target_end_date <= "2026-10-24"].merge(
    obs, on=["location", "target_end_date"], how="left"
)
settled["reported"] = settled.reported.fillna(0)
settled["in_90"] = (settled.reported >= settled.q05) & (
    settled.reported <= settled.q95
)
settled["median_over_reported"] = (settled.q50 + 1) / (settled.reported + 1)
settled[[
    "location", "location_name", "target_end_date", "model", "group",
    "reported", "q05", "q50", "q95", "in_90", "median_over_reported",
]].to_csv(E / "data_check.csv", index=False, float_format="%.2f")
print("places", len(places), "rows", len(allq))

# Never-seeded zones: zones with no case in the release. Count those
# where a model's q95 or q50 is 1 or more in some forecast week.
seen = set(z.location)
fw = ["2026-11-14", "2026-11-21", "2026-11-28", "2026-12-05"]
every = pd.concat([hub[cols], ours[cols]])
nz = every[
    (every.location.str.count("-") == 2)
    & ~every.location.isin(seen)
    & every.target_end_date.isin(fw)
]
nz = nz.assign(
    q95_ge1=nz.location.where(nz.q95 >= 1),
    q50_ge1=nz.location.where(nz.q50 >= 1),
)
seed = nz.groupby(["group", "model"]).agg(
    zones_given=("location", "nunique"),
    zones_q95_ge1=("q95_ge1", "nunique"),
    zones_q50_ge1=("q50_ge1", "nunique"),
).reset_index()
seed.to_csv(E / "never_seeded.csv", index=False)
print(seed)
