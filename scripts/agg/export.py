"""Write the paper's aggregation-scores.csv.gz (`aggregation_rows`).

BDBV: main's complete target on the rolling truth (D from
`resolve_truth_delay`), from `bdbv_score.py`'s scores, with the 95%
interval's coverage from the samples (`sample_coverage`). Germany: the
development half, from `germany_score2.py`'s quantile scores.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from bdbvhubrenewal.data import load_daily_zones, load_target
from bdbvhubrenewal.data.zones import is_zone
from bdbvhubrenewal.evaluation import backtest as bt
from bdbvhubrenewal.evaluation.groupings import (
    aggregation_rows,
    rolling_zone_truth,
    sample_coverage,
)
from bdbvhubrenewal.evaluation.nowcast import (
    ReleaseForecasts,
    resolve_truth_delay,
    rolling_truth,
)
from bdbvhubrenewal.evaluation.zones import SUMMED, ZoneForecasts

BASE = "randomwalknowcast"
CELL = ["model", "origin", "location", "target_end_date"]


def bdbv_samples(specs: list[str]) -> pd.DataFrame:
    frames = []
    for spec in specs:
        where, _, ckpt = spec.partition("@")
        parts = sorted(
            p for p in Path(where).iterdir() if p.is_dir() and "__" in p.name
        )
        s = ReleaseForecasts.read(parts).samples
        if ckpt:
            s = s.assign(model=s["model"].str.replace("_ft:", f"_ft@{ckpt}:"))
        frames.append(s)
    out = pd.concat(frames, ignore_index=True)
    if "as_of" in out:
        out = out.loc[out["as_of"].isna()].drop(columns="as_of")
    return out.astype({"origin": str, "target_end_date": str})


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data")
    parser.add_argument("--bdbv-fits", nargs="+", required=True)
    parser.add_argument("--bdbv-scores", required=True)
    parser.add_argument("--germany-scores")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    target = load_target(args.data)
    delay = resolve_truth_delay(bt.load_config("config/backtest.toml"), target)
    real = target.loc[target["kind"] == "real"]
    truth = pd.concat(
        [
            rolling_truth(real, delay),
            rolling_zone_truth(load_daily_zones(args.data), delay),
        ],
        ignore_index=True,
    )
    res = Path(args.bdbv_scores)
    scores = pd.concat(
        [
            pd.read_csv(res / f"scores-{delay}.csv.gz", dtype=str),
            pd.read_csv(res / "zone-scores.csv.gz", dtype=str).assign(
                level="zones"
            ),
        ],
        ignore_index=True,
    )
    scores = scores.loc[scores["target"] == "complete"]
    base = ReleaseForecasts.read([res / "base"]).samples
    base = base.loc[~base["location"].map(is_zone)]
    zones = ZoneForecasts.read([res / "rw-zones"]).samples
    zones = zones.loc[zones["key"] == BASE + SUMMED].assign(model=BASE)
    # randomwalknowcast's country and provinces come from its own run and
    # its zones from its zone run summed to provinces, as scored.
    samples = pd.concat(
        [
            bdbv_samples(args.bdbv_fits),
            base.astype({"origin": str, "target_end_date": str}),
            zones.drop(columns="key").astype(
                {"origin": str, "target_end_date": str}
            ),
        ],
        ignore_index=True,
    )
    cover = sample_coverage(samples, truth, 0.95)
    bdbv = scores.merge(cover, on=CELL, how="left").rename(
        columns={"coverage_50": "cover50"}
    )
    bdbv = bdbv.astype({"wis": float, "log_wis": float, "cover50": float})
    frames = [aggregation_rows(bdbv, "ebola")]
    if args.germany_scores:
        de = pd.read_csv(args.germany_scores, dtype={"origin": str})
        de = de.rename(
            columns={"coverage_50": "cover50", "coverage_95": "cover95"}
        ).assign(level=lambda d: d["pool"].replace({"age": "ages"}))
        frames.append(aggregation_rows(de, "germany"))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    table = pd.concat(frames, ignore_index=True)
    table.to_csv(out / "aggregation-scores.csv.gz", index=False)
    print(
        table.groupby(["study", "level", "option"])
        .agg(n=("wis", "size"), wis=("wis", "mean"))
        .to_string()
    )


if __name__ == "__main__":
    main()
