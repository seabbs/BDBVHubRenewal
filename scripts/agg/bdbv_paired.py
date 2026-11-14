"""Make the paired arm on BDBV from the bottom arm's zone draws.

Reads every `<model>:bottom__<origin>` part of --fits, pairs its zone
draws by the rolling copula on the zones' rolling truth
(`evaluation.groupings.pair_and_sum`), sums them to provinces and the
country, each raised to its count at the origin, and writes
`<model>:paired__<origin>` parts to --out, as `ReleaseForecasts`.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from bdbvhubrenewal.data import load_daily_zones, load_target
from bdbvhubrenewal.data.zones import is_zone, province_of
from bdbvhubrenewal.evaluation import backtest as bt
from bdbvhubrenewal.evaluation.groupings import pair_and_sum, rolling_zone_truth
from bdbvhubrenewal.evaluation.nowcast import (
    ReleaseForecasts,
    resolve_truth_delay,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data")
    parser.add_argument("--fits", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--seed", type=int, default=20261027)
    args = parser.parse_args()
    target = load_target(args.data)
    config = bt.load_config("config/backtest.toml")
    delay = resolve_truth_delay(config, target)
    truth = rolling_zone_truth(load_daily_zones(args.data), delay)
    parts = sorted(Path(args.fits).glob("*:bottom__*"))
    run = ReleaseForecasts.read(parts)
    s = run.samples
    s = s.loc[s["location"].map(is_zone)].copy()
    s["origin"] = s["origin"].astype(str)
    s["target_end_date"] = s["target_end_date"].astype(str)
    s["model"] = s["model"].str.replace(":bottom", ":paired")
    real = target.loc[target["kind"] == "real"]
    floors = real.rename(columns={"vintage": "origin", "total": "floor"})[
        ["origin", "location", "target_end_date", "floor"]
    ].astype({"origin": str, "target_end_date": str})
    paired, fits = pair_and_sum(
        s, truth, args.seed, parent=province_of, floors=floors
    )
    out = Path(args.out)
    for (model, origin), rows in paired.groupby(["model", "origin"]):
        runs = pd.DataFrame(
            [{"origin": origin, "model": model, "status": "forecast",
              "reason": ""}]
        )  # fmt: skip
        empty = pd.DataFrame(columns=["model", "origin"])
        mine = fits.loc[(fits["model"] == model) & (fits["origin"] == origin)]
        ReleaseForecasts(runs, rows, empty, mine).write(
            out / f"{model}__{origin}"
        )
    print(fits.to_string(index=False))


if __name__ == "__main__":
    main()
