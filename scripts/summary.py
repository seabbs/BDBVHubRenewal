"""Scores of the Chronos-2 models and the random walks on matched cells.

Relative WIS is against randomwalknowcast.

Each target (reported, complete) is scored on the cells every model
scored on that target forecasts, split into nowcast weeks (weeks back >= 0) and forecast weeks.
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
# Within each target, the cells every model scored on that target has.
n_models = s.groupby("target")["model"].nunique().rename("n_models")
common = s.groupby(keys)["model"].nunique().rename("n").reset_index()
common = common.merge(n_models, on="target")
common = common.loc[common["n"] == common["n_models"], keys]
s = s.merge(common, on=keys)
cols = ["target", "weeks", "model", "n", "rel_wis", "rel_log_wis",
        "coverage_50", "coverage_90", "bias"]
out = []
for scale, rows in (("all", s), ("national", s[s["location"] == "CD"])):
    t = relative_wis(
        rows, by=("target", "weeks"), baseline="randomwalknowcast"
    )
    t.insert(0, "scale", scale)
    out.append(t[["scale", *cols]])
out = pd.concat(out).round(2)
print(out.to_string(index=False))
out.to_csv(res / "fm-summary.csv", index=False)
print("origins", sorted(s["origin"].unique()), "weeks back",
      sorted(s["weeks_back"].unique()))
