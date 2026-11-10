"""Tables of the 2026-11-07 round audit (`audit1107.py`).

Usage: audit_report.py <S> <out md>

Reads `<S>/audit1107_{scores,quantiles}_{then,now}.csv` and writes:
- `<S>/audit1107_cells.csv`: one row per model, place and week, then and
  now, on the reported target (what the 2026-11-08 release shows);
- `<S>/audit1107_summary.csv`: mean WIS, log WIS and 50%/90% coverage by
  model, scale and week group, then and now, over the cells both score
  (the then files hold fewer provinces and the stored fits fewer weeks);
- the markdown report.
"""

import sys
from pathlib import Path

import pandas as pd

S, OUT = Path(sys.argv[1]), Path(sys.argv[2])
NAMES = pd.read_csv("data/locations.csv").set_index("location").location_name


def load(kind: str) -> pd.DataFrame:
    parts = [pd.read_csv(S / f"audit1107_{kind}_{w}.csv") for w in
             ("then", "now")]
    d = pd.concat(parts, ignore_index=True)
    return d[d.target == "reported"]


q, s = load("quantiles"), load("scores")
key = ["model", "location", "target_end_date"]
# compare like with like: cells with a truth that both then and now score
q = q[q.truth.notna()]
both = s.groupby(key).when.nunique()
both = both[both == 2].index
s = s.set_index(key).loc[lambda x: x.index.isin(both)].reset_index()
cell = q.pivot_table(index=key, columns="when",
                     values=["q5", "q50", "q95"]).reset_index()
cell.columns = ["_".join(c).strip("_") for c in cell.columns]
truth = q.drop_duplicates(key).set_index(key)[["truth", "reported"]]
cell = cell.join(truth, on=key)
w = s.pivot_table(index=key, columns="when", values="wis")
w.columns = [f"wis_{c}" for c in w.columns]
cell = cell.join(w, on=key)


def fmt(r: pd.Series, when: str) -> str:
    if pd.isna(r.get(f"q50_{when}")):
        return "-"
    return (f"{r[f'q50_{when}']:.0f} [{r[f'q5_{when}']:.0f}, "
            f"{r[f'q95_{when}']:.0f}]")


def towards(r: pd.Series) -> str:
    if pd.isna(r.get("q50_then")) or pd.isna(r.get("q50_now")):
        return ""
    a, b = abs(r.q50_then - r.truth), abs(r.q50_now - r.truth)
    if abs(a - b) < max(1.0, 0.02 * r.truth):
        return "same"
    return "towards" if b < a else "away"


cell["then"] = cell.apply(fmt, axis=1, when="then")
cell["now"] = cell.apply(fmt, axis=1, when="now")
cell["move"] = cell.apply(towards, axis=1)
cell.to_csv(S / "audit1107_cells.csv", index=False)

s["scale"] = s.location.map(lambda x: "national" if x == "CD"
                            else "province")
s["weeks"] = pd.cut(
    pd.to_datetime(s.target_end_date),
    pd.to_datetime(["2026-09-01", "2026-10-17", "2026-10-31",
                    "2026-11-07"]),
    labels=["to 10-17", "10-24 to 10-31", "11-07"],
)
summ = s.groupby(["model", "scale", "weeks", "when"], observed=True).agg(
    n=("wis", "size"), wis=("wis", "mean"), log_wis=("log_wis", "mean"),
    cov50=("coverage_50", "mean"), cov90=("coverage_90", "mean"),
).reset_index()
summ.to_csv(S / "audit1107_summary.csv", index=False)

lines = ["# 2026-11-07 round audit: then against now", ""]
lines += [
    "Origin: the 2026-11-01 release. Truth: the 2026-11-08 release.",
    "Then: the files we had at the round (run 41691221605; submitted: "
    "randomwalknowcast as seabbs-rwnowcast, renewalbvd as seabbs-renewal). "
    "Now: main's code at the same origin (random walks run at 96ac128d; "
    "renewal models from the 11-01 fits at fa04ff08 on `fit-store`). "
    "pprandomwalknowcast has no then samples.",
    "All values are on the reported target: the forecast mapped to what "
    "the 2026-11-08 release shows through the model's delay now "
    "(`randomwalk` as it is), as the release backtest scores. "
    "Median [5%, 95%].",
    "",
    "## Scores",
    "",
    "| model | scale | weeks | n | WIS then | WIS now | log WIS then "
    "| log WIS now | 50% cov then | now | 90% cov then | now |",
    "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- "
    "| --- |",
]
sp = summ.pivot_table(index=["model", "scale", "weeks"], columns="when",
                      values=["n", "wis", "log_wis", "cov50", "cov90"],
                      observed=True)
for (m, sc, wk), r in sp.iterrows():
    def g(c: str, wh: str, f: str = ".1f") -> str:
        v = r.get((c, wh))
        return "-" if pd.isna(v) else format(v, f)
    lines.append(
        f"| {m} | {sc} | {wk} | {g('n', 'now', '.0f')} | {g('wis', 'then')} "
        f"| {g('wis', 'now')} | {g('log_wis', 'then', '.3f')} "
        f"| {g('log_wis', 'now', '.3f')} | {g('cov50', 'then', '.2f')} "
        f"| {g('cov50', 'now', '.2f')} | {g('cov90', 'then', '.2f')} "
        f"| {g('cov90', 'now', '.2f')} |"
    )
lines += ["", "## By place and week", ""]
for m, g in cell.groupby("model"):
    lines += [f"### {m}", "",
              "| place | week | then | now | 11-08 | move | WIS then "
              "| WIS now |", "| --- | --- | --- | --- | --- | --- | --- "
              "| --- |"]
    g = g[(g.truth > 0) | (g.location == "CD")]
    for r in g.sort_values(["location", "target_end_date"]).itertuples():
        wt = "-" if pd.isna(r.wis_then) else f"{r.wis_then:.1f}"
        wn = "-" if pd.isna(r.wis_now) else f"{r.wis_now:.1f}"
        lines.append(
            f"| {NAMES.get(r.location, r.location)} | {r.target_end_date[5:]}"
            f" | {r.then} | {r.now} | {r.truth:.0f} | {r.move} | {wt} | {wn} |"
        )
    lines.append("")
OUT.write_text("\n".join(lines) + "\n")
pd.set_option("display.width", 220)
print(sp.round(3).to_string())
mv = cell[cell.move != ""].groupby(["model", "move"]).size().unstack(
    fill_value=0)
print(mv)
