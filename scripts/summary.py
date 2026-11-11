"""Scores of the Chronos-2 models and the random walks on matched cells.

Each target (reported, complete) is scored on the cells every model
forecasts, split into nowcast weeks (weeks back >= 0) and forecast weeks.
National is also shown on its own: on the complete target it is each
model's own national series.
"""
import sys
from pathlib import Path

import pandas as pd

from bdbvhubrenewal.evaluation.nowcast import relative_wis

res = Path(sys.argv[1])
s = pd.read_csv(res / "nowcast-scores.csv")
s["weeks"] = s["weeks_back"].map(lambda k: "nowcast" if k >= 0 else "forecast")
keys = ["target", "origin", "truth_vintage", "location", "target_end_date"]
models = sorted(s["model"].unique())
common = s.groupby(keys)["model"].nunique()
common = common[common == len(models)].reset_index()[keys]
s = s.merge(common, on=keys)
cols = ["target", "weeks", "model", "n", "rel_wis", "rel_log_wis",
        "coverage_50", "coverage_90", "bias"]
out = []
for scale, rows in (("all", s), ("national", s[s["location"] == "CD"])):
    t = relative_wis(rows, by=("target", "weeks"))
    t.insert(0, "scale", scale)
    out.append(t[["scale", *cols]])
out = pd.concat(out).round(2)
print(out.to_string(index=False))
out.to_csv(res / "fm-summary.csv", index=False)
print("origins", sorted(s["origin"].unique()), "weeks back",
      sorted(s["weeks_back"].unique()))
