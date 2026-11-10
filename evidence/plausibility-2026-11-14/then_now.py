"""Last week's forecasts against this week's, for CD and the provinces.

Usage: then_now.py <S> <subs dir> now:model=file ... then:model=file ...

`then` files are last week's (2026-11-07 round, data to 11-01); a `then`
entry may also be `q1107` to take the model from the 11-07 evidence
quantiles (`<S>/q1107.csv`). `now` files are the 2026-11-14 round (data to
11-08). Writes `<S>/then_now.csv` and prints a markdown table per model.
"""

import sys
from pathlib import Path

import pandas as pd

S = Path(sys.argv[1])
WEEKS = ["2026-10-24", "2026-10-31", "2026-11-07", "2026-11-14",
         "2026-11-28"]
ts = pd.read_csv(S / "hub/ts.csv")
obs = {
    v: ts[ts.as_of == v].set_index(["location", "target_end_date"]).observation
    for v in ["2026-11-01", "2026-11-08"]
}


def quant(path: str) -> pd.DataFrame:
    d = pd.read_csv(path)
    if "output_type_id" in d.columns:
        d = d.rename(columns={"output_type_id": "sample"})
    d = d[d.location.str.count("-") <= 1]
    q = d.groupby(["location", "target_end_date"]).value.quantile(
        [0.05, 0.5, 0.95]
    ).unstack()
    q.columns = ["q05", "q50", "q95"]
    return q


q1107 = pd.read_csv(S / "q1107.csv").set_index(["location", "target_end_date"])
pairs: dict[str, dict[str, pd.DataFrame]] = {}
for arg in sys.argv[3:]:
    when, rest = arg.split(":", 1)
    model, path = rest.split("=", 1)
    if path == "q1107":
        q = q1107[q1107.model == model][["q05", "q50", "q95"]]
    else:
        q = quant(path)
    pairs.setdefault(model, {})[when] = q

rows = []
for model, p in pairs.items():
    then, now = p["then"], p["now"]
    locs = sorted(set(now.index.get_level_values(0)))
    for loc in locs:
        for w in WEEKS:
            def fmt(q: pd.DataFrame) -> str:
                if (loc, w) not in q.index:
                    return ""
                r = q.loc[(loc, w)]
                return f"{r.q50:.0f} [{r.q05:.0f}, {r.q95:.0f}]"
            rows.append(dict(
                model=model, location=loc, week=w, then=fmt(then),
                now=fmt(now),
                then_med=then.q50.get((loc, w)), now_med=now.q50.get((loc, w)),
                rep_1101=obs["2026-11-01"].get((loc, w)),
                rep_1108=obs["2026-11-08"].get((loc, w)),
            ))
out = pd.DataFrame(rows)
out.to_csv(S / "then_now.csv", index=False)
# provinces with a case in the last 8 weeks
recent = ts[(ts.as_of == "2026-11-08") & (ts.target_end_date >= "2026-09-19")]
active = set(recent.loc[recent.observation > 0, "location"])
for model, g in out.groupby("model", sort=False):
    g = g[g.location.isin(active) & (g.location.str.count("-") <= 1)]
    print(f"\n#### {model}\n")
    print("| place | week | then (11-07 round) | now (11-14 round) "
          "| reported 11-01 | reported 11-08 |")
    print("| --- | --- | --- | --- | --- | --- |")
    for r in g.itertuples():
        a = "" if pd.isna(r.rep_1101) else f"{r.rep_1101:.0f}"
        b = "" if pd.isna(r.rep_1108) else f"{r.rep_1108:.0f}"
        print(f"| {r.location} | {r.week[5:]} | {r.then} | {r.now} "
              f"| {a} | {b} |")
