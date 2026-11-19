"""Check one round of simulations against fixed plausibility thresholds.

Summarises each simulation of `ROUND_DIR/sims` (`summaries`) and excludes
those past a fixed threshold (`check`), each with its reason and evidence:

- cases per stratum (the median total over draws, over the strata with
  cases) above `--max-cases-per-stratum`: past the size the grid's
  sampling distribution allows a stratum (5e5);
- weekly cases rising more than `--max-weekly-jump`-fold in one week with
  no reporting shock: the grid caps daily growth at 0.2 (at most about
  4-fold in a week), and only a backlog or an outage releases reports in
  bulk;
- a mean reporting delay above `--max-mean-delay-days`: the grid's mean
  delay is at most 21 days.

Records the input table, the exclusions (`excluded.csv`: simulation,
reason, evidence), the thresholds (`decisions.json`) and a one-line
summary in `ROUND_DIR/qc/`. Simulations are never deleted; the trainer
skips the excluded ones. Run from the repo root:

    uv run python scripts/sim_qc.py ROUND_DIR [--max-cases-per-stratum N]
        [--max-weekly-jump N] [--max-mean-delay-days N]
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

LIMITS = {
    "max_cases_per_stratum": 5e5,
    "max_weekly_jump": 1000,
    "max_mean_delay_days": 35,
}
"""The thresholds, as workflow/config.yaml's simulate.qc sets them."""

SHOCKS = {"backlog", "outage"}
"""Reporting shocks that release reports in bulk."""
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

    `table` is `summaries`' output; `limits` holds `max_cases_per_stratum`,
    `max_weekly_jump` and `max_mean_delay_days`. A simulation past several
    thresholds is listed once per threshold.
    """
    table = table.reset_index(drop=True)
    if "shocks" not in table:
        table = table.assign(shocks="none")
    per_stratum = table["cases_median"] / table["strata_with_cases"].clip(
        lower=1
    )
    jump = math.log(limits["max_weekly_jump"])
    no_shock = ~table["shocks"].isin(SHOCKS)
    rules = [
        (
            per_stratum > limits["max_cases_per_stratum"],
            f"cases per stratum above {limits['max_cases_per_stratum']:,.0f}",
            lambda i: (
                f"cases_per_stratum={per_stratum[i]:.0f}, "
                f"strata_with_cases={table['strata_with_cases'][i]}"
            ),
        ),
        (
            (table["max_weekly_log_growth"] > jump) & no_shock,
            f"weekly cases rise more than {limits['max_weekly_jump']:,.0f}-"
            "fold in one week with no reporting shock",
            lambda i: (
                "max_weekly_fold="
                f"{math.exp(table['max_weekly_log_growth'][i]):.0f}, "
                f"shocks={table['shocks'][i]}"
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
    for key, value in LIMITS.items():
        parser.add_argument(
            "--" + key.replace("_", "-"), type=float, default=value
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
