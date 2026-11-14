"""Score the German grouping arms from samples, with the paired arm.

The paired arm pairs each `:bottom` arm's states, and apart its age
groups, by the rolling copula on the 40-day truth, and sums the states
to Germany, raised to Germany's count at the date
(`evaluation.groupings.pair_and_sum`). Every arm's samples become the
hub's quantiles (`hub_quantiles`) and are scored as #205 does, with WIS on
log(x + 1) of quantiles and truth, relative to randomwalknowcast by
level on the forecasts every arm made.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from bdbvhubrenewal.data.germany import (
    AGE_GROUPS,
    GERMANY,
    hub_target,
    hub_triangle,
)
from bdbvhubrenewal.evaluation import germany as de
from bdbvhubrenewal.evaluation.groupings import pair_and_sum
from bdbvhubrenewal.evaluation.nowcast import rolling_truth
from bdbvhubrenewal.reporting.hubtables import hub_quantiles

BASE = "randomwalknowcast"
KEYS = ["model", "origin", "location", "target_end_date", "horizon"]


def quantiles(samples: pd.DataFrame) -> pd.DataFrame:
    table = samples.rename(columns={"sample": "output_type_id"}).assign(
        output_type="sample"
    )
    q = hub_quantiles(
        table[[*KEYS, "output_type", "output_type_id", "value"]],
        de.QUANTILE_LEVELS,
    )
    return q.drop(columns="output_type").rename(
        columns={"output_type_id": "quantile"}
    )[[*KEYS, "quantile", "value"]]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/raw/germany")
    parser.add_argument("--samples", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--seed", type=int, default=20261027)
    args = parser.parse_args()
    config = de.load_germany_config("config/germany.toml")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    files = sorted(Path(args.samples).glob("*.csv.gz"))
    samples = pd.concat(
        [
            pd.read_csv(f, dtype={"origin": str, "target_end_date": str})
            for f in files
        ],
        ignore_index=True,
    )
    assert samples["origin"].max() < config.split_date
    triangle = hub_triangle(Path(args.data) / Path(config.triangle).name)
    ends = sorted(set(samples["target_end_date"]))
    target = hub_target(triangle, ends)
    truth = rolling_truth(target, config.truth_delay)
    nat = target.loc[target["location"] == GERMANY]
    floors = nat.rename(columns={"vintage": "origin", "total": "floor"})[
        ["origin", "location", "target_end_date", "floor"]
    ].astype({"origin": str, "target_end_date": str})
    bottom = samples.loc[samples["model"].str.endswith(":bottom")]
    bottom = bottom.assign(
        model=bottom["model"].str.replace(":bottom", ":paired")
    )
    ages = bottom["location"].isin(set(AGE_GROUPS))
    states = ~ages & (bottom["location"] != GERMANY)
    horizon = bottom[["origin", "target_end_date", "horizon"]].drop_duplicates()
    paired_states, fits_s = pair_and_sum(
        bottom.loc[states], truth, args.seed, floors=floors, national=GERMANY
    )
    paired_ages, fits_a = pair_and_sum(
        bottom.loc[ages], truth, args.seed + 1, national=GERMANY
    )
    paired = pd.concat(
        [paired_states, paired_ages.loc[paired_ages["location"] != GERMANY]],
        ignore_index=True,
    ).merge(horizon, on=["origin", "target_end_date"])
    paired.to_csv(out / "paired-samples.csv.gz", index=False)
    pd.concat(
        [fits_s.assign(level="states"), fits_a.assign(level="age")]
    ).to_csv(out / "paired-fits.csv", index=False)
    every = pd.concat([samples, paired[samples.columns]], ignore_index=True)
    forecasts = de.with_truth(quantiles(every), triangle, config.truth_delay)
    forecasts.to_csv(out / "germany-forecasts.csv.gz", index=False)
    natural = de.score_germany(forecasts, config.truth_delay)
    logged = de.score_germany(
        forecasts.assign(
            value=np.log1p(forecasts["value"].clip(lower=0)),
            truth=np.log1p(forecasts["truth"]),
        ),
        config.truth_delay,
    )
    cell = ["model", "origin", "location", "target_end_date"]
    scores = natural.merge(
        logged[[*cell, "wis"]].rename(columns={"wis": "log_wis"}), on=cell
    )
    n_models = scores["model"].nunique()
    common = scores.groupby(cell[1:])["model"].transform("nunique")
    scores = scores.loc[common == n_models]
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
    print(f"{scores['origin'].nunique()} dates, {n_models} arms")
    with pd.option_context("display.width", 200, "display.max_rows", 500):
        print(np.round(table, 3).to_string(index=False))


if __name__ == "__main__":
    main()
