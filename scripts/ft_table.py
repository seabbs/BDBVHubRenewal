"""#193's table: relative WIS (relative log WIS) against a baseline, all
places, on summary.py's matched cells, for reported nowcast, reported
forecast and complete nowcast.

Usage: ft_table.py BASELINE RESULTS... (one results directory per arm,
named results-<arm>), from the fm-finetune worktree's root.
"""
import sys
from pathlib import Path

import pandas as pd

from bdbvhubrenewal.evaluation.nowcast import relative_wis

baseline, dirs = sys.argv[1], sys.argv[2:]
keys = ["target", "origin", "truth_vintage", "location", "target_end_date"]
cols = [("reported", "nowcast"), ("reported", "forecast"),
        ("complete", "nowcast")]
out = []
for d in dirs:
    s = pd.read_csv(Path(d) / "nowcast-scores.csv")
    s["weeks"] = s["weeks_back"].map(lambda k: "nowcast" if k >= 0 else "forecast")
    n_models = s.groupby("target")["model"].nunique().rename("n_models")
    common = s.groupby(keys)["model"].nunique().rename("n").reset_index()
    common = common.merge(n_models, on="target")
    s = s.merge(common.loc[common["n"] == common["n_models"], keys], on=keys)
    t = relative_wis(s, by=("target", "weeks"), baseline=baseline)
    t["cell"] = t.apply(
        lambda r: f"{r.rel_wis:.2f} ({r.rel_log_wis:.2f})", axis=1)
    w = t.pivot(index="model", columns=["target", "weeks"], values="cell")
    w = w[cols]
    w.columns = [f"{a} {b}" for a, b in cols]
    w.insert(0, "D", Path(d).name.removeprefix("results-"))
    out.append(w.reset_index())
out = pd.concat(out, ignore_index=True)
print(out.to_markdown(index=False))
