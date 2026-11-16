# Copyright (c) 2026 Sam Abbott. MIT licence, see LICENSE.
"""Summarise covariate ablation shards as paired tables (markdown).

Every comparison is paired: only blocks scored under both `given` and the
condition. For each set, model, condition and window (nowcast, forecast)
it gives the mean loss under `given` and the condition, their ratio, the
share of blocks worse than `given` and a bootstrap 90% interval of the
ratio over blocks.

    python scripts/covariate_ablation_summary.py OUT_DIR
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd


def table(d: pd.DataFrame, window: str) -> pd.DataFrame:
    """Return the paired table of one window."""
    rng = np.random.default_rng(1)
    given = d[d["condition"] == "given"].set_index(["group", "model", "block"])
    rows = []
    for (group, model, cond), part in d[d["condition"] != "given"].groupby(
        ["group", "model", "condition"], sort=False
    ):
        part = part.set_index(["group", "model", "block"])
        g = given.loc[part.index, window].to_numpy()
        c = part[window].to_numpy()
        ok = np.isfinite(g) & np.isfinite(c)
        g, c = g[ok], c[ok]
        if len(g) == 0:
            continue
        idx = rng.integers(0, len(g), (2000, len(g)))
        boot = c[idx].mean(1) / g[idx].mean(1)
        rows.append(
            {
                "set": group,
                "model": model,
                "condition": cond,
                "blocks": len(g),
                "given": g.mean(),
                "condition loss": c.mean(),
                "ratio": c.mean() / g.mean(),
                "90% CI": f"{np.quantile(boot, 0.05):.2f}-"
                f"{np.quantile(boot, 0.95):.2f}",
                "share worse": (c > g + 1e-9).mean(),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    """Print the tables."""
    out = Path(sys.argv[1])
    d = pd.concat(pd.read_csv(f) for f in sorted(out.glob("shard-*.csv")))
    multi = np.where(d["n_targets"] > 1, "multi-place", "univariate")
    d["group"] = np.where(
        d["set"] == "bdbv", "BDBV pseudo-origins", "sims " + multi
    )
    alls = d.assign(group=np.where(d["set"] == "bdbv", "BDBV", "sims all"))
    d = pd.concat([d, alls[alls["group"] == "sims all"]])
    order = {
        "removed": 0,
        "no_horizon": 1,
        "shuffle_time": 2,
        "shuffle_time_all": 3,
        "shuffle_places": 4,
        "swap_places": 5,
    }
    print(
        "blocks:",
        d[d["condition"] == "given"]
        .groupby(["group", "source"])["block"]
        .nunique()
        .div(2)
        .astype(int)
        .to_dict()
        if False
        else d[(d["condition"] == "given") & (d["model"] == "n1000")]
        .groupby(["group", "source"])
        .size()
        .to_dict(),
    )
    for window in ("nowcast", "forecast"):
        t = table(d, window)
        t["o"] = t["condition"].map(order)
        t = t.sort_values(["set", "model", "o"]).drop(columns="o")
        print(f"\n### {window} steps\n")
        print(t.to_markdown(index=False, floatfmt=".3f"))
    # Zero-shot vs n1000 with covariates given, paired.
    g = d[d["condition"] == "given"].pivot_table(
        index=["group", "block"],
        columns="model",
        values=["nowcast", "forecast"],
    )
    print("\n### n1000 / zero-shot, covariates given\n")
    for window in ("nowcast", "forecast"):
        r = (
            g[(window, "n1000")].groupby("group").mean()
            / g[(window, "zero-shot")].groupby("group").mean()
        )
        print(window, r.round(3).to_dict())
    bd = d[(d["set"] == "bdbv") & (d["group"] == "BDBV pseudo-origins")]
    print("\n### BDBV per origin\n")
    print(
        bd.pivot_table(
            index=["id", "model"],
            columns="condition",
            values=["nowcast", "forecast"],
        )
        .round(2)
        .to_markdown()
    )


if __name__ == "__main__":
    main()
