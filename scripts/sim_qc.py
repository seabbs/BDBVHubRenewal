"""Check one round of simulations against fixed plausibility thresholds.

Summarises each simulation of `ROUND_DIR/sims` (`summaries`) and excludes
those past a threshold (`check`), each with its reason and evidence: more
cases in a stratum than its population (the median total over draws; only
where the settings give each stratum's population), the largest
week-on-week rise in weekly cases by reference date, and the mean
reporting delay. Size has no other threshold (#408 item 10). The
thresholds and why they are set as they are live in the grid file
(`qc`, the one `simulate.grid_file` names); the workflow passes them.

Records the input table, the exclusions (`excluded.csv`: simulation,
reason, evidence), the thresholds (`decisions.json`) and a one-line
summary in `ROUND_DIR/qc/`. Simulations are never deleted; the trainer
skips the excluded ones. Run from the repo root:

    uv run python scripts/sim_qc.py ROUND_DIR --max-weekly-jump N
        --max-mean-delay-days N
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

LIMITS = ("max_weekly_jump", "max_mean_delay_days")
"""The thresholds `check` takes, each a required option."""

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
    "reporting",
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


def check(table: pd.DataFrame, limits: dict[str, float]) -> pd.DataFrame:
    """Return the simulations past a threshold, with reason and evidence.

    `table` is `summaries`' output; `limits` holds `max_weekly_jump` and
    `max_mean_delay_days`. A simulation past several
    thresholds is listed once per threshold.
    """
    table = table.reset_index(drop=True)
    jump = math.log(limits["max_weekly_jump"])
    rules = [
        (
            table["cases_over_population"] > 1,
            "cases in a stratum above its population",
            lambda i: (
                f"cases_over_population={table['cases_over_population'][i]:.2f}"
            ),
        ),
        (
            table["max_weekly_log_growth"] > jump,
            f"weekly cases rise more than {limits['max_weekly_jump']:,.0f}-"
            "fold in one week",
            lambda i: (
                "max_weekly_fold="
                f"{math.exp(table['max_weekly_log_growth'][i]):.0f}"
            ),
        ),
        (
            table["delay_mean_days"] > limits["max_mean_delay_days"],
            "mean reporting delay over "
            f"{limits['max_mean_delay_days']:,.0f} days",
            lambda i: f"delay_mean_days={table['delay_mean_days'][i]:.1f}",
        ),
    ]
    out = []
    for mask, reason, evidence in rules:
        for i in mask[mask].index:
            out.append(
                {
                    "simulation": table["simulation"][i],
                    "reason": reason,
                    "evidence": evidence(i),
                }
            )
    return pd.DataFrame(out, columns=["simulation", "reason", "evidence"])


def main() -> None:
    """Check the round and record everything in its qc/ directory."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("round_dir", type=Path)
    for key in LIMITS:
        parser.add_argument(
            "--" + key.replace("_", "-"), type=float, required=True
        )
    args = parser.parse_args()
    limits = {k: getattr(args, k) for k in LIMITS}
    qc = args.round_dir / "qc"
    qc.mkdir(parents=True, exist_ok=True)
    table = summaries(args.round_dir / "sims")
    table.to_csv(qc / "input.csv", index=False)
    excluded = check(table, limits)
    excluded.to_csv(qc / "excluded.csv", index=False)
    n_out = excluded["simulation"].nunique()
    (qc / "decisions.json").write_text(
        json.dumps(
            {
                "check": "fixed thresholds (scripts/sim_qc.py)",
                "limits": limits,
                "n_simulations": len(table),
                "excluded": excluded.to_dict("records"),
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
