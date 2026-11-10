"""Turn the hub's 2026-11-14 sample files into quantiles.

Usage: hub_quantiles.py <S> <out csv>

Reads `S/hubclone/model-output/*/2026-11-14-*.csv` (the hub's main
branch) and `S/pr114/model-output/*/2026-11-14-*.csv` (the UQPSU files
in the open hub PR #114, dates written day first). Writes q05, q25,
q50, q75 and q95 of each model, location and target week, with a
`status` column (`merged` or `open PR #114`).
"""

import sys
from pathlib import Path

import pandas as pd

S, out = Path(sys.argv[1]), Path(sys.argv[2])
LEVELS = {"q05": 0.05, "q25": 0.25, "q50": 0.5, "q75": 0.75, "q95": 0.95}
frames = []
for root, status in [(S / "hubclone", "merged"), (S / "pr114", "open PR #114")]:
    for path in sorted(root.glob("model-output/*/2026-11-14-*.csv")):
        d = pd.read_csv(path, dtype=str)
        d = d[(d.target == "wk inc confirmed cases") & (d.output_type == "sample")]
        dayfirst = "/" in d.target_end_date.iloc[0]
        d["target_end_date"] = pd.to_datetime(
            d.target_end_date, dayfirst=dayfirst
        ).dt.strftime("%Y-%m-%d")
        d["value"] = d.value.astype(float)
        d["model"] = path.parent.name
        d["status"] = status
        frames.append(d[["model", "status", "location", "target_end_date", "value"]])
s = pd.concat(frames)
g = s.groupby(["model", "status", "location", "target_end_date"]).value
q = pd.concat({k: g.quantile(v) for k, v in LEVELS.items()}, axis=1)
q["n_samples"] = g.size()
q.reset_index().to_csv(out, index=False)
print(q.reset_index().groupby("model").agg(
    locations=("location", "nunique"),
    first=("target_end_date", "min"), last=("target_end_date", "max"),
    samples=("n_samples", "max"),
))
