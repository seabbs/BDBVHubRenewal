"""Each Chronos-2 model's own zones against its allocator zones.

Mean WIS and log WIS of `own` over `allocator`, on every zone forecast
both made (results/zone-scores.csv.gz), so origins randomwalknowcast
cannot forecast from are kept.
"""

import sys

import pandas as pd

scores = pd.read_csv(sys.argv[1] if len(sys.argv) > 1 else "results/zone-scores.csv.gz")
fm = scores.loc[scores["model"].str.contains(":")].copy()
fm[["name", "variant"]] = fm["model"].str.split(":", expand=True)
keys = ["name", "origin", "location", "target_end_date"]
wide = fm.pivot_table(index=keys, columns="variant", values=["wis", "log_wis"])
mean = wide.groupby(level="name").mean()
out = pd.DataFrame(
    {
        "n": wide.groupby(level="name").size(),
        "own_over_allocator_wis": mean[("wis", "own")] / mean[("wis", "allocator")],
        "own_over_allocator_log_wis": mean[("log_wis", "own")]
        / mean[("log_wis", "allocator")],
    }
).round(3)
print(out.to_string())
out.to_csv("results/own-vs-allocator.csv")
