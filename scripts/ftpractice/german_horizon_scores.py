"""WIS by horizon on the German development half, nowcast and forecast.

Scores `german_horizons.py` output with the scorer's own `score_quantiles`
and writes, per checkpoint, model and horizon, the mean WIS, its ratio to
the same model zero-shot on the same forecasts, 50% and 95% coverage and
the median's bias. Also the continuity at the cut-off: the forecast's
change from the 7-day sum ending on the nowcast date to the one a week
later (median at +7 over median at 0) against the truth's.

    python scripts/ftpractice/german_horizon_scores.py IN_DIR OUT.csv \
        reference=zero-shot slots-6bb954c7ffe7=r6-n500 ...

Daily zero-shot sums independent daily draws to 7-day sums, so its
intervals are narrow by construction; the weekly model is the clean
comparison.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

from bdbvhubrenewal.evaluation.germany import QUANTILE_LEVELS
from bdbvhubrenewal.evaluation.scoring import score_quantiles

KEYS = ["model", "origin", "location", "target_end_date"]


def read(root: Path) -> pd.DataFrame:
    """Return one label's rows, with the model's `_ft` suffix dropped."""
    rows = pd.concat(
        (pd.read_csv(f) for f in sorted(root.glob("*.csv.gz"))),
        ignore_index=True,
    )
    if "target" in rows.columns:
        rows = rows[rows["target"].fillna("7day") == "7day"]
    rows["model"] = rows["model"].str.removesuffix("_ft")
    return rows


def scores(rows: pd.DataFrame) -> pd.DataFrame:
    """Return each forecast's WIS, coverage and median."""
    truth = rows[[*KEYS, "truth"]].drop_duplicates(KEYS)
    out = score_quantiles(
        rows[[*KEYS, "horizon", "quantile", "value"]],
        truth,
        KEYS,
        QUANTILE_LEVELS,
    )
    wis = [c for c in out.columns if c.startswith("wis")]
    out["wis"] = out[wis[0]] if len(wis) == 1 else out["wis_quantile"]
    median = rows[np.isclose(rows["quantile"], 0.5)][[*KEYS, "value"]]
    return out.merge(median.rename(columns={"value": "median"}), on=KEYS)


def continuity(s: pd.DataFrame) -> pd.DataFrame:
    """Return the median log ratio of predicted to true change, 0 to +7."""
    at = s.pivot_table(
        index=["model", "origin", "location"],
        columns="horizon",
        values=["median", "truth"],
    )
    pred = np.log((at[("median", 7)] + 1) / (at[("median", 0)] + 1))
    true = np.log((at[("truth", 7)] + 1) / (at[("truth", 0)] + 1))
    gap = (pred - true).rename("gap").reset_index()
    return gap.groupby("model")["gap"].agg(
        continuity_bias="median",
        continuity_mae=lambda g: g.abs().median(),
    )


def main() -> None:
    """Score each label and write the table."""
    root, out = Path(sys.argv[1]), Path(sys.argv[2])
    labels = dict(a.split("=", 1) for a in sys.argv[3:])
    frames, conts = [], []
    for label, name in labels.items():
        s = scores(read(root / label))
        s["checkpoint"] = name
        frames.append(s)
        c = continuity(s).reset_index()
        c["checkpoint"] = name
        conts.append(c)
    s = pd.concat(frames, ignore_index=True)
    ref = s[s["checkpoint"] == labels["reference"]][
        ["model", "origin", "location", "horizon", "wis"]
    ].rename(columns={"wis": "wis_zero_shot"})
    s = s.merge(ref, on=["model", "origin", "location", "horizon"])
    s["bias"] = (s["median"] - s["truth"]) / s["truth"].clip(lower=1)
    table = (
        s.groupby(["checkpoint", "model", "horizon"])
        .agg(
            n=("wis", "size"),
            wis=("wis", "mean"),
            wis_zero_shot=("wis_zero_shot", "mean"),
            cover50=("coverage_50", "mean"),
            cover95=("coverage_95", "mean"),
            bias=("bias", "median"),
        )
        .reset_index()
    )
    table["rel_wis"] = table["wis"] / table["wis_zero_shot"]
    table = table.merge(
        pd.concat(conts), on=["checkpoint", "model"], how="left"
    )
    table.to_csv(out, index=False)
    for model in sorted(table["model"].unique()):
        t = table[table["model"] == model]
        print(f"\n{model}: mean WIS relative to zero-shot")
        print(
            t.pivot_table(
                index="checkpoint", columns="horizon", values="rel_wis"
            )
            .round(2)
            .to_string()
        )
        print(f"{model}: 95% coverage")
        print(
            t.pivot_table(
                index="checkpoint", columns="horizon", values="cover95"
            )
            .round(2)
            .to_string()
        )
    print("\ncontinuity (log predicted/true change, 0 to +7):")
    print(
        pd.concat(conts).set_index(["checkpoint", "model"]).round(3).to_string()
    )


if __name__ == "__main__":
    main()
