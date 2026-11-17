"""Review one round of simulations with the fixed sim-QC prompt (A).

Summarises each simulation of `ROUND_DIR/sims` (`summaries`), gives the
table to Claude with `config/finetune/sim-qc-prompt-a.md` through the
`claude` CLI, and records the prompt and its sha256, the input table, the
raw output, the exclusions (`excluded.csv`: simulation, reason,
evidence) and a one-line summary in `ROUND_DIR/qc/`. Exclusion is
reluctant: the prompt defaults to keeping, and an exclusion naming a
simulation not in the round is dropped and noted. Simulations are never
deleted; the trainer skips the excluded ones. Run from the repo root:

    uv run python scripts/sim_qc.py ROUND_DIR
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROMPT = Path("config/finetune/sim-qc-prompt-a.md")
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


def review(table: pd.DataFrame, prompt: str) -> dict:
    """Return Claude's review of the table, as the prompt's JSON."""
    text = (
        prompt + "\n\n## Simulations\n\n" + table.round(3).to_csv(index=False)
    )
    run = subprocess.run(
        ["claude", "-p", "--output-format", "json"],
        input=text,
        capture_output=True,
        text=True,
        check=True,
        timeout=1800,
    )
    raw = json.loads(run.stdout)
    body = raw.get("result", "")
    start, end = body.find("{"), body.rfind("}")
    return {"raw": raw, "review": json.loads(body[start : end + 1])}


def main() -> None:
    """Review the round and record everything in its qc/ directory."""
    round_dir = Path(sys.argv[1])
    qc = round_dir / "qc"
    qc.mkdir(parents=True, exist_ok=True)
    prompt = PROMPT.read_text()
    (qc / "prompt-a.md").write_text(prompt)
    digest = hashlib.sha256(prompt.encode()).hexdigest()
    table = summaries(round_dir / "sims")
    table.to_csv(qc / "input.csv", index=False)
    result = review(table, prompt)
    (qc / "output.json").write_text(json.dumps(result["raw"], indent=1))
    excluded = pd.DataFrame(
        result["review"].get("excluded", []),
        columns=["simulation", "reason", "evidence"],
    )
    unknown = excluded.loc[~excluded["simulation"].isin(table["simulation"])]
    excluded = excluded.loc[excluded["simulation"].isin(table["simulation"])]
    excluded.to_csv(qc / "excluded.csv", index=False)
    notes = result["review"].get("notes", "")
    (qc / "decisions.json").write_text(
        json.dumps(
            {
                "prompt_sha256": digest,
                "n_simulations": len(table),
                "excluded": excluded.to_dict("records"),
                "dropped_unknown_ids": unknown["simulation"].tolist(),
                "notes": notes,
            },
            indent=1,
        )
    )
    (qc / "summary.txt").write_text(
        f"sim QC excluded {len(excluded)} of {len(table)} simulations"
        + (f" ({len(unknown)} unknown ids dropped)" if len(unknown) else "")
    )
    print((qc / "summary.txt").read_text())


if __name__ == "__main__":
    main()
