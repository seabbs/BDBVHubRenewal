"""Record one round of simulations' summaries.

Summarises each simulation of `ROUND_DIR/sims` (`summaries`): cases,
strata with cases, the largest stratum's cases over its population, the
largest week-on-week rise in weekly cases by reference date and the
reporting delay. The grid sets no thresholds, so no simulation is
excluded. `excluded.csv` is written empty because the trainer and the
releases read it.

Records the input table, the exclusions (`excluded.csv`: simulation,
reason, evidence), `decisions.json` and a one-line summary in
`ROUND_DIR/qc/`. Simulations are never deleted. Run from the repo root:

    uv run python scripts/sim_qc.py ROUND_DIR
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

CELL = [
    "strata",
    "n_places",
    "n_ages",
    "spread",
    "trajectory",
    "process",
    "delay_family",
    "delay_form",
    "delay_change",
    "report_days",
    "shocks",
    "mixing",
    "imports",
    "seeding",
    "observation",
    "obs_strata",
    "delay_places",
    "strata_link",
]


def cases_over_population(counts: pd.DataFrame, settings: Any) -> float:
    """Return the largest ratio of a stratum's cases to its population.

    Cases are each stratum's median total over draws. NaN unless the
    settings give each stratum's location and population
    (`stratum_location`, `stratum_population`, ";"-separated).
    """
    try:
        where = str(settings["stratum_location"]).split(";")
        size = [
            float(x) for x in str(settings["stratum_population"]).split(";")
        ]
    except (KeyError, TypeError, ValueError):
        return float("nan")
    if len(where) != len(size):
        return float("nan")
    population = pd.Series(size, index=where)
    cases = (
        counts.groupby(["location", "outbreak"])["count"]
        .sum()
        .groupby("location")
        .median()
    )
    ratio = cases / population.reindex(cases.index)
    return float(ratio.max()) if ratio.notna().any() else float("nan")


def summaries(sim_dir: Path) -> pd.DataFrame:
    """Return one row per simulation: its cell and count summaries."""
    settings = pd.concat(
        pd.read_csv(p) for p in sorted(sim_dir.glob("settings-*.csv"))
    ).set_index("outbreak")
    rows = []
    for path in sorted(sim_dir.glob("counts-*.csv.gz")):
        counts = pd.read_csv(path)
        counts["simulation"] = counts["outbreak"].str.rsplit("-", n=1).str[0]
        for simulation, c in counts.groupby("simulation"):
            ref = pd.to_datetime(c["reference_date"])
            delay = (pd.to_datetime(c["report_date"]) - ref).dt.days
            w = c["count"].to_numpy(float)
            per_draw = c.groupby("outbreak")["count"].sum()
            week = ref.dt.to_period("W")
            weekly = (
                c.assign(week=week).groupby(["outbreak", "week"])["count"].sum()
            )
            first = weekly.loc[per_draw.index[0]]
            growth = np.diff(np.log(first.to_numpy(float) + 1))
            order = np.argsort(delay.to_numpy())
            cdf = np.cumsum(w[order]) / w.sum()
            s = settings.loc[simulation] if simulation in settings.index else {}
            rows.append(
                {
                    "simulation": simulation,
                    **{
                        k: s.get(k) for k in CELL if k in getattr(s, "index", s)
                    },
                    "draws": per_draw.size,
                    "cases_median": float(per_draw.median()),
                    "cases_min": float(per_draw.min()),
                    "cases_max": float(per_draw.max()),
                    "strata_with_cases": c["location"].nunique(),
                    "cases_over_population": cases_over_population(c, s),
                    "weeks": int(first.size),
                    "peak_weekly": float(first.max()),
                    "max_weekly_log_growth": float(growth.max())
                    if growth.size
                    else np.nan,
                    "delay_mean_days": float((delay * w).sum() / w.sum()),
                    "delay_q99_days": float(
                        delay.to_numpy()[order][np.searchsorted(cdf, 0.99)]
                    ),
                    "reported_within_7d": float(
                        w[delay.to_numpy() <= 7].sum() / w.sum()
                    ),
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    """Check the round and record everything in its qc/ directory."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("round_dir", type=Path)
    args = parser.parse_args()
    qc = args.round_dir / "qc"
    qc.mkdir(parents=True, exist_ok=True)
    table = summaries(args.round_dir / "sims")
    table.to_csv(qc / "input.csv", index=False)
    excluded = pd.DataFrame(columns=["simulation", "reason", "evidence"])
    excluded.to_csv(qc / "excluded.csv", index=False)
    n_out = 0
    (qc / "decisions.json").write_text(
        json.dumps(
            {
                "check": "no thresholds (scripts/sim_qc.py)",
                "n_simulations": len(table),
                "excluded": [],
            },
            indent=1,
        )
    )
    (qc / "summary.txt").write_text(
        f"sim QC excluded {n_out} of {len(table)} simulations"
    )
    print((qc / "summary.txt").read_text())


if __name__ == "__main__":
    main()
