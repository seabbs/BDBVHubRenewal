"""Which grid factors explain predicted width on the held-out sims.

Regresses each simulation's log median nowcast 90% width on its grid
factors and reports each factor's drop in R^2 when left out.

    python scripts/ftpractice/width_factors.py COND_DIR SIM_DIR \
        --seeds 32000 32039
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

FACTORS = {
    "sqrt_phi": "num",
    "shocks": "cat",
    "delay_change": "cat",
    "spread": "cat",
    "log_cases": "num",
    "delay_median": "num",
    "delay_sdlog": "num",
}


def design(d: pd.DataFrame, factors: list[str]) -> np.ndarray:
    """Return an intercept plus the factors, categorical ones as dummies."""
    cols = [np.ones(len(d))]
    for f in factors:
        if FACTORS[f] == "num":
            cols.append(d[f].to_numpy(float))
        else:
            dummies = pd.get_dummies(d[f], drop_first=True, dtype=float)
            cols.extend(dummies.to_numpy().T)
    return np.column_stack(cols)


def r2(x: np.ndarray, y: np.ndarray) -> float:
    """Return the R^2 of a least-squares fit."""
    beta, *_ = np.linalg.lstsq(x, y, rcond=None)
    resid = y - x @ beta
    return float(1 - resid.var() / y.var())


def main() -> None:
    """Print each checkpoint's full R^2 and each factor's drop."""
    p = argparse.ArgumentParser()
    p.add_argument("cond", type=Path)
    p.add_argument("sim_dir", type=Path)
    p.add_argument("--seeds", type=int, nargs=2, required=True)
    args = p.parse_args()
    seeds = range(args.seeds[0], args.seeds[1] + 1)
    settings = pd.concat(
        pd.read_csv(args.sim_dir / f"settings-{s}.csv") for s in seeds
    ).rename(columns={"outbreak": "simulation"})
    extra = settings[["simulation", "delay_median", "delay_sdlog"]]
    per = pd.read_csv(args.cond / "per-sim.csv").merge(extra, on="simulation")
    per = per[per["part"] == "nowcast"].copy()
    per["log_cases"] = np.log1p(pd.to_numeric(per["cases"], errors="coerce"))
    names = list(FACTORS)
    rows = []
    targets = dict(tuple(per.groupby("checkpoint")))
    own = per.drop_duplicates("simulation").assign(
        checkpoint="sims' own spread"
    )
    own["width90"] = own["own_spread"]
    targets["sims' own spread"] = own
    for name, g in targets.items():
        g = g.dropna(subset=["width90", *names])
        g = g[g["width90"] > 0]
        y = np.log(g["width90"].to_numpy())
        full = r2(design(g, names), y)
        rec = {"target": name, "sims": len(g), "R2": full}
        for f in names:
            rec[f] = full - r2(design(g, [n for n in names if n != f]), y)
        rows.append(rec)
    print(pd.DataFrame(rows).round(3).to_markdown(index=False))


if __name__ == "__main__":
    main()
