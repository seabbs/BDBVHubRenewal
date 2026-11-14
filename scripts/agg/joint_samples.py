"""Write the joint-block Chronos-2 samples in main's forecasts layout.

Columns as main's results/forecasts.csv.gz: model, kind (real), vintage,
reference_date, location, target_end_date, sample, value, as_of (empty)
and horizon (weeks from reference_date). Models fitted with another
checkpoint take a suffix (`DIR=suffix`). Zones are left out, as in
main's file.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from bdbvhubrenewal.data.zones import is_zone
from bdbvhubrenewal.evaluation import backtest as bt
from bdbvhubrenewal.evaluation.nowcast import ReleaseForecasts


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--fits", nargs="+", required=True, help="DIR or DIR=suffix"
    )
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    frames = []
    for spec in args.fits:
        where, _, suffix = spec.partition("=")
        parts = sorted(Path(where).glob("*:joint__*"))
        s = ReleaseForecasts.read(parts).samples
        s = s.assign(model=s["model"].str.removesuffix(":joint") + suffix)
        frames.append(s)
    s = pd.concat(frames, ignore_index=True)
    # Main's forecasts hold the country and provinces only.
    s = s.loc[~s["location"].map(is_zone)]
    s["vintage"] = s["origin"].astype(str)
    ref = {v: bt.reference_date_for_vintage(v) for v in set(s["vintage"])}
    s["reference_date"] = s["vintage"].map(lambda v: str(ref[v]))
    s["horizon"] = (
        pd.to_datetime(s["target_end_date"])
        - pd.to_datetime(s["reference_date"])
    ).dt.days // 7
    out = s.assign(kind="real", as_of=pd.NA)[
        [
            "model", "kind", "vintage", "reference_date", "location",
            "target_end_date", "sample", "value", "as_of", "horizon",
        ]
    ]  # fmt: skip
    out.to_csv(args.out, index=False)
    print(out.groupby("model").size().to_string())


if __name__ == "__main__":
    main()
