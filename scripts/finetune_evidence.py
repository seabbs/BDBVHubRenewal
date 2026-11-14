"""Fine-tuning evidence for the paper (#223): the held-out loss curve and
one row per simulated outbreak of the r3 pilot.

Usage: finetune_evidence.py CKPT_DIR RUN PILOT_DIR OUT_DIR, from the
fm-finetune-r3 worktree's root (for `scripts/finetune_realism.py`).
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, "scripts")
import finetune_realism as realism  # noqa: E402

ckpt, run, pilot, out = Path(sys.argv[1]), sys.argv[2], Path(sys.argv[3]), Path(sys.argv[4])

# Simulations seen by each step, from the consumed order.
seen, total, by_step = set(), [], {}
for line in (ckpt / "consumed.txt").read_text().splitlines():
    step, ids = line.split("\t")
    for draw in ids.split(","):
        seen.add("-".join(draw.split(":")[0].split("-")[:2]))
    by_step[int(step)] = len(seen)
rows = []
for h in json.loads((ckpt / "history.json").read_text()):
    for key, kind in (("held_out_loss", "all held-out blocks"), ("eval_loss", "first 100 held-out blocks")):
        if key in h and h[key] is not None and np.isfinite(h[key]):
            rows.append({"run": run, "outbreaks_seen": by_step.get(h["step"], 0), "step": h["step"], "heldout_loss": h[key], "blocks": kind})
pd.DataFrame(rows).to_csv(out / "finetune-heldout.csv", index=False)

# One row per simulated outbreak of the pilot, with its cell.
sims = realism.simulated([pilot], np.random.default_rng(1), sample=None)
settings = pd.concat(pd.read_csv(p) for p in pilot.glob("settings-*.csv"))
factors = ["process", "gt_shape", "trajectory", "delay_family", "delay_form", "delay_change", "reporting", "reference_dow", "observation", "strata"]
sims["simulation"] = sims["id"].str.split("-").str[:2].str.join("-")
merged = sims.merge(settings[["outbreak", *factors]].rename(columns={"outbreak": "simulation"}), on="simulation")
merged.drop(columns=["source"]).rename(columns={"id": "outbreak"}).to_csv(out / "sim-summaries-r3-pilot.csv", index=False)
print(len(rows), "loss rows;", len(merged), "outbreaks")
