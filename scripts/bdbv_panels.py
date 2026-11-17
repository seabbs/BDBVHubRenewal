"""BDBV province nowcast panels, one per model (`nowcast_panels`).

Usage: python scripts/bdbv_panels.py DATA ZERO R1 CHECKPOINTS R1_SCORES OUT

DATA is the tidy data (`target.csv`). ZERO holds the release backtest's
`forecasts.csv.gz` with randomwalknowcast and the zero-shot daily ragged
model, R1 the round 1 fine-tuned model's, and CHECKPOINTS
`<hash>/forecasts.csv.gz` per r3 checkpoint (`evidence/checkpoints.csv`).
R1_SCORES is the round 1 release backtest's `nowcast-scores.csv`, for
its relative WIS on the cells of `evidence/checkpoint-scores.csv`.

The panels are at origin 2026-10-17, the one origin with a scored week,
for the three provinces with most cases by then. The black line is the
latest release's count, as most weeks have no 35-day truth yet.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from nowcast_panels import draw_panels

ORIGIN = "2026-10-17"
LEVELS = (0.025, 0.25, 0.5, 0.75, 0.975)
DAILY = "fm_chronos_daily_ragged"
BASELINE = "randomwalknowcast"


def _quantiles(samples: pd.DataFrame, model: str) -> pd.DataFrame:
    """Return a model's sample quantiles at `LEVELS` by origin and week."""
    rows = samples.loc[
        (samples["model"] == model) & (samples["vintage"] == ORIGIN)
    ]
    keys = ["vintage", "location", "target_end_date"]
    q = rows.groupby(keys)["value"].quantile(list(LEVELS)).reset_index()
    return q.rename(columns={"vintage": "origin", "level_3": "quantile"})


def _r1_rel(scores: Path, cells: pd.DataFrame) -> float:
    """Return round 1's province relative WIS on the matched cells."""
    s = pd.read_csv(scores)
    s = s.loc[(s["target"] == "complete") & (s["location"] != "CD")]
    keys = ["origin", "truth_vintage", "location", "target_end_date"]
    s = s.merge(cells, on=keys)
    s = s.loc[s["model"].isin([f"{DAILY}_ft", BASELINE])]
    both = s.groupby(keys)["model"].nunique()
    s = s.set_index(keys).loc[both.index[both == 2]].reset_index()
    wis = s.groupby("model")["wis"].mean()
    return float(wis[f"{DAILY}_ft"] / wis[BASELINE])


def main() -> None:
    """Draw the BDBV province panels."""
    data, zero, r1, ckpts, r1_scores, out = map(Path, sys.argv[1:7])
    target = pd.read_csv(data / "target.csv", dtype={"kind": str})
    real = target.loc[target["kind"] == "real"]
    latest = real["vintage"].max()
    at = real.loc[(real["vintage"] == ORIGIN) & (real["location"] != "CD")]
    places = at.groupby("location")["total"].sum().nlargest(3).index.tolist()
    zs = pd.read_csv(zero / "forecasts.csv.gz")
    models = {
        BASELINE: _quantiles(zs, BASELINE),
        "Chronos-2 zero-shot": _quantiles(zs, DAILY),
        "Chronos-2 ft round 1": _quantiles(
            pd.read_csv(r1 / "forecasts.csv.gz"), f"{DAILY}_ft"
        ),
    }
    curve = pd.read_csv("evidence/checkpoint-scores.csv")
    curve = curve.loc[
        (curve["study"] == "ebola")
        & (curve["level"] == "provinces")
        & (curve["model"] == DAILY)
    ].set_index("checkpoint")
    zero_scores = pd.read_csv(zero / "nowcast-scores.csv")
    cells = zero_scores.loc[
        (zero_scores["model"] == BASELINE)
        & (zero_scores["target"] == "complete")
        & (zero_scores["location"] != "CD")
        & (zero_scores["origin"] == ORIGIN),
        ["origin", "truth_vintage", "location", "target_end_date"],
    ]
    scored = {
        BASELINE: 1.0,
        "Chronos-2 zero-shot": curve.loc["zero-shot", "rel_wis"],
        "Chronos-2 ft round 1": _r1_rel(r1_scores, cells),
    }
    checkpoints = pd.read_csv("evidence/checkpoints.csv", dtype=str)
    for ckpt, outbreaks in checkpoints.itertuples(index=False):
        path = ckpts / ckpt / "forecasts.csv.gz"
        if not path.exists() or ckpt not in curve.index:
            continue
        label = f"Chronos-2 ft r3 n{outbreaks}"
        models[label] = _quantiles(pd.read_csv(path), f"{DAILY}_ft")
        scored[label] = curve.loc[ckpt, "rel_wis"]
    weeks = sorted(set(real.loc[real["vintage"] == ORIGIN, "target_end_date"]))
    weeks = weeks[-10:]
    asof = real.loc[
        (real["vintage"] == ORIGIN) & real["target_end_date"].isin(weeks),
        ["vintage", "location", "target_end_date", "total"],
    ].rename(columns={"vintage": "origin"})
    truth = real.loc[
        (real["vintage"] == latest) & (real["target_end_date"] >= weeks[0]),
        ["location", "target_end_date", "total"],
    ].rename(columns={"total": "truth"})
    names = dict(zip(at["location"], at["location_name"], strict=False))
    draw_panels(
        models,
        scored,
        truth,
        asof,
        [(ORIGIN, p, f"{names.get(p, p)} ({p})") for p in places],
        f"BDBV weekly cases, three provinces with most cases: nowcasts at "
        f"origin {ORIGIN} by model (daily Chronos-2 source)",
        out,
        truth_label=f"latest release ({latest})",
        truth_short="latest",
        scored_on="the 26 province weeks with a 35-day truth (origin "
        f"{ORIGIN})",
    )


if __name__ == "__main__":
    main()
