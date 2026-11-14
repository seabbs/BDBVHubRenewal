"""Score the German grouping nowcasts of `germany_fits.py`, as #205 does.

Quantile WIS on the 40-day truth (`with_truth`, `score_germany`), and the
same on log(x + 1) of the quantiles and truth, relative to
randomwalknowcast by level, on the forecasts every model made.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from bdbvhubrenewal.data.germany import hub_triangle
from bdbvhubrenewal.evaluation import germany as de

BASE = "randomwalknowcast"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/raw/germany")
    parser.add_argument("--fits", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    config = de.load_germany_config("config/germany.toml")
    files = sorted(Path(args.fits).glob("*.csv.gz"))
    forecasts = pd.concat(
        [
            pd.read_csv(f, dtype={"origin": str, "target_end_date": str})
            for f in files
        ],
        ignore_index=True,
    )
    assert forecasts["origin"].max() < config.split_date
    triangle = hub_triangle(Path(args.data) / Path(config.triangle).name)
    rows = de.with_truth(forecasts, triangle, config.truth_delay)
    natural = de.score_germany(rows, config.truth_delay)
    logged = de.score_germany(
        rows.assign(
            value=np.log1p(rows["value"].clip(lower=0)),
            truth=np.log1p(rows["truth"]),
        ),
        config.truth_delay,
    )
    keys = ["model", "origin", "location", "target_end_date"]
    scores = natural.merge(
        logged[[*keys, "wis"]].rename(columns={"wis": "log_wis"}), on=keys
    )
    # Every model on the same forecasts.
    n_models = scores["model"].nunique()
    common = scores.groupby(keys[1:])["model"].transform("nunique")
    scores = scores.loc[common == n_models]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    scores.to_csv(out / "germany-scores.csv.gz", index=False)
    rel = de.relative_scores(scores, ["pool"], BASE)
    log_rel = de.relative_scores(
        scores.drop(columns="wis").rename(columns={"log_wis": "wis"}),
        ["pool"],
        BASE,
    )
    table = rel[["pool", "model", "n", "rel_wis", "coverage_95"]].merge(
        log_rel[["pool", "model", "rel_wis"]].rename(
            columns={"rel_wis": "rel_log_wis"}
        ),
        on=["pool", "model"],
    )
    table.to_csv(out / "germany-relative.csv", index=False)
    print(f"{scores['origin'].nunique()} dates, {files[0].stem[:10]} on")
    with pd.option_context("display.width", 200, "display.max_rows", 500):
        print(np.round(table, 3).to_string(index=False))


if __name__ == "__main__":
    main()
