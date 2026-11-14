"""Nowcast the German development half per grouping arm, keeping samples.

One file per date and arm, `<date>__<arm>.csv.gz`, from
`evaluation.groupings.germany_grouping_samples`. Arms are given as
`model:grouping` or `model@checkpoint:grouping`; `randomwalknowcast`
alone. Dates on or after the held-out split are refused.
"""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

if os.environ.get("AGG_CPU"):
    import torch

    # CPU only: the laptop GPU is held by the fine-tune trainer.
    _mps = torch.backends.mps.is_available

    def _no_mps() -> bool:
        return False

    _no_mps.__wrapped__ = _mps  # type: ignore[attr-defined]
    torch.backends.mps.is_available = _no_mps

from bdbvhubrenewal.data.germany import hub_triangle
from bdbvhubrenewal.evaluation import germany as de
from bdbvhubrenewal.evaluation.groupings import germany_grouping_samples


def parse(arm: str) -> tuple[str, str, str | None]:
    model, _, grouping = arm.partition(":")
    model, _, checkpoint = model.partition("@")
    return model, grouping or "joint", checkpoint or None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/raw/germany")
    parser.add_argument("--arms", nargs="+", required=True)
    parser.add_argument("--part", default="0/1", help="k/n: every nth date")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    config = de.load_germany_config("config/germany.toml")
    dates = de.nowcast_dates(config)
    assert max(dates) < config.split_date
    k, n = (int(x) for x in args.part.split("/"))
    triangle = hub_triangle(Path(args.data) / Path(config.triangle).name)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for date in dates[k::n]:
        for arm in args.arms:
            path = out / f"{date}__{arm}.csv.gz"
            if path.exists():
                continue
            t0 = time.time()
            model, grouping, checkpoint = parse(arm)
            samples = germany_grouping_samples(
                model, grouping, triangle, date, config.n_samples,
                config.seed, checkpoint,
            )  # fmt: skip
            samples.to_csv(path, index=False)
            print(f"{date} {arm} {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
