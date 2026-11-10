"""Check the 2026-11-07 round against the 2026-11-08 release.

Inputs (in <S>): `hub/ts.csv` (make_ts.py), `q1107.csv` and `flags2_1107.csv`
and `flags_1107.csv` (quantiles.csv, flags2.csv and flags.csv of the
2026-11-07 evidence at 971b2572), and the two submitted files.
Weeks up to 10-17 are at least three weeks old in the 2026-11-08 release, so
near complete; 10-24 and 10-31 are still filling in and week 11-07 holds
onsets to 11-05 only, so for those weeks a count above the 95% quantile is a
miss but a count below the 5% quantile is not.
"""

import sys
from pathlib import Path

import pandas as pd

S, SUBS = Path(sys.argv[1]), Path(sys.argv[2])  # argv[3]: data dir
ts = pd.read_csv(S / "hub/ts.csv")
sys.path.insert(0, str(Path(__file__).parent))
from names import location_names  # noqa: E402

names = location_names(S, Path(sys.argv[3]))
obs = ts[ts.as_of == "2026-11-08"].set_index(
    ["location", "target_end_date"]
).observation
old = ts[ts.as_of == "2026-11-01"].set_index(
    ["location", "target_end_date"]
).observation
COMPLETE = ["2026-09-26", "2026-10-03", "2026-10-10", "2026-10-17"]
LATE = ["2026-10-24", "2026-10-31", "2026-11-07"]

q = pd.read_csv(S / "q1107.csv")
subs = {
    "seabbs-renewal (submitted)": SUBS
    / "seabbs-renewal/2026-11-07-seabbs-renewal.csv.gz",
    "seabbs-rwnowcast (submitted)": SUBS
    / "seabbs-rwnowcast/2026-11-07-seabbs-rwnowcast.csv.gz",
}
for model, path in subs.items():
    d = pd.read_csv(path)
    s = d.groupby(["location", "target_end_date"]).value.quantile(
        [0.05, 0.5, 0.95, 0.975]
    ).unstack()
    s.columns = ["q05", "q50", "q95", "q975"]
    q = pd.concat([q, s.reset_index().assign(model=model)])
q = q[q.target_end_date.isin(COMPLETE + LATE)]
# zones the hub has no row for in a week have reported 0
q["obs"] = [obs.get((loc, t), 0) for loc, t in zip(q.location, q.target_end_date)]
q["scale"] = q.location.map(
    lambda x: "national" if x == "CD"
    else "province" if x.count("-") == 1 else "zone"
)
q["below"] = q.obs < q.q05
q["above"] = q.obs > q.q95
q["inside"] = ~(q.below | q.above)
q["is90"] = (q.q95 - q.q05) + 20 * (q.q05 - q.obs).clip(lower=0) \
    + 20 * (q.obs - q.q95).clip(lower=0)
q["ae"] = (q.q50 - q.obs).abs()
q.to_csv(S / "check1107.csv", index=False)

pd.set_option("display.width", 250)
c = q[q.target_end_date.isin(COMPLETE)]
tab = c.groupby(["scale", "model"]).agg(
    n=("obs", "size"), cover90=("inside", "mean"), below=("below", "sum"),
    above=("above", "sum"), mae=("ae", "mean"), is90=("is90", "mean"),
).round(2)
print("Weeks 09-26 to 10-17 (near complete):")
print(tab.to_string())
late = q[q.target_end_date.isin(LATE)]
tab = late.groupby(["scale", "model", "target_end_date"]).agg(
    n=("obs", "size"), above=("above", "sum")
).unstack("target_end_date")
print("\nWeeks 10-24 to 11-07, count already above the 95% quantile:")
print(tab.to_string())

# national and province medians against the data
for loc in ["CD", "CD-IT", "CD-NK", "CD-HU", "CD-TO"]:
    g = q[q.location == loc].pivot_table(
        index="target_end_date", columns="model", values="q50"
    ).round(0)
    g["obs 11-08"] = [obs.get((loc, t)) for t in g.index]
    g["obs 11-01"] = [old.get((loc, t)) for t in g.index]
    print(f"\n{loc} median by week")
    print(g.to_string())

# the flagged places of the earlier round, against the new release
for fname in ["flags2_1107.csv", "flags_1107.csv"]:
    f = pd.read_csv(S / fname)
    f = f.drop_duplicates(["model", "location", "flag"])
    rows = []
    for r in f.itertuples():
        g = q[(q.model == r.model) & (q.location == r.location)].set_index(
            "target_end_date"
        )
        rows.append(dict(
            model=r.model, name=names.get(r.location, r.location), flag=r.flag,
            **{f"obs {t[5:]}": obs.get((r.location, t), 0) for t in
               ["2026-10-17", "2026-10-24", "2026-10-31", "2026-11-07"]},
            **{f"fc {t[5:]}": f"{g.q50[t]:.0f} ({g.q05[t]:.0f}-{g.q95[t]:.0f})"
               for t in ["2026-10-31", "2026-11-07"] if t in g.index},
        ))
    out = pd.DataFrame(rows)
    out.to_csv(S / f"held_{fname}", index=False)
    print(f"\n{fname}")
    print(out.to_string())
