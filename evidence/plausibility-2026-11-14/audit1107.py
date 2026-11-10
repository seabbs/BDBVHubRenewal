"""Audit the 2026-11-07 round: last week's forecasts against main's now.

Usage: audit1107.py <S> <art dir> <fit dir> [model ...]

Origin: the 2026-11-01 release (round 2026-11-07).
- then: the files we had at the round, from run 41691221605's artifacts in
  <art dir>: `randomwalk` and `randomwalknowcast` candidates, `renewal`
  (candidate-main), `renewalsp` and `renewalbvd` (candidate-bvd, the file
  submitted as seabbs-renewal). `randomwalknowcast`'s candidate is the file
  submitted as seabbs-rwnowcast.
- now: main's code at the same origin. The random walk models are run here
  (`forecast_releases`). The renewal models are the stored 11-01 fits in
  <fit dir> (`<model>.json.gz`, from the `fit-store` branch).

Both are scored as the release backtest scores them (`score_releases`)
against the 2026-11-08 release: the reported target, through each model's
delay (`randomwalk` with none). A "then" file has no delay summary of its
own, so it is mapped through the delay of the same model now, at the same
origin. The scores and quantiles are written to <S>/audit1107_*.csv.
"""

import sys
from pathlib import Path

import pandas as pd

from bdbvhubrenewal.data.tidy import load_daily, load_delays, load_target
from bdbvhubrenewal.evaluation.backtest import load_config
from bdbvhubrenewal.evaluation.nowcast import (
    ReleaseForecasts,
    forecast_releases,
    score_releases,
)
from bdbvhubrenewal.evaluation.store import origin_tasks, read_fit

S, ART, FITS = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
WALKS = sys.argv[4:] or ["randomwalk", "randomwalknowcast",
                          "pprandomwalknowcast"]
ORIGIN = "2026-11-01"
COLS = ["location", "target_end_date", "sample", "value"]
THEN = {
    "randomwalk": "inputs/candidates/randomwalk/submission/model-output/"
    "seabbs-randomwalk/2026-11-07-seabbs-randomwalk.csv",
    "randomwalknowcast": "inputs/candidates/randomwalknowcast/submission/"
    "model-output/seabbs-rwnowcast/2026-11-07-seabbs-rwnowcast.csv",
    "renewal": "candidate-main/submission/model-output/seabbs-renewal/"
    "2026-11-07-seabbs-renewal.csv",
    "renewalsp": "candidate-renewalsp/submission/model-output/"
    "seabbs-renewal/2026-11-07-seabbs-renewal.csv",
    "renewalbvd": "candidate-bvd/submission/model-output/seabbs-renewal/"
    "2026-11-07-seabbs-renewal.csv",
}
FITTED = ["renewal", "renewalsp", "renewalbvd"]

target = load_target("data")
daily = load_daily("data")
config = load_config("config/backtest.toml")
task = origin_tasks(target, load_delays("data"), config, daily)[ORIGIN]


def hub_samples(path: Path) -> pd.DataFrame:
    d = pd.read_csv(path)
    d = d[d.location.str.count("-") <= 1]
    return d.rename(columns={"output_type_id": "sample"})[COLS]


# now: the random walks, run here
cache = S / "audit1107_run.pkl"
if cache.exists():
    run = pd.read_pickle(cache)
else:
    run = forecast_releases([task], target, WALKS)
    pd.to_pickle(run, cache)
now_delays = run.delays.drop(columns="origin")
stored, learnt = [], [now_delays]
for m in FITTED:
    fit = read_fit(FITS / f"{m}.json.gz")
    print(m, fit.meta["key"], fit.meta["commit"][:8], fit.meta["fitted_at"])
    fc = fit.forecasts
    fc = fc[fc["as_of"].isna()] if "as_of" in fc else fc
    stored.append(fc[COLS].assign(model=m, vintage=ORIGIN))
    learnt.append(fit.delays.assign(model=m, vintage=ORIGIN))
delays = pd.concat(learnt, ignore_index=True)
now = score_releases(
    [task], daily, target, run,
    stored=pd.concat(stored, ignore_index=True), delays=delays,
)  # fmt: skip

# then: last week's files, each through the same model's delay now
then_samples = pd.concat(
    [hub_samples(ART / p).assign(model=m, vintage=ORIGIN)
     for m, p in THEN.items()],
    ignore_index=True,
)  # fmt: skip
empty = ReleaseForecasts(
    pd.DataFrame(columns=["origin", "model", "status", "reason"]),
    pd.DataFrame(columns=["model", "origin", *COLS]),
    pd.DataFrame(columns=["model", "origin"]),
    pd.DataFrame(columns=["model", "origin"]),
)
then = score_releases(
    [task], daily, target, empty, stored=then_samples, delays=delays
)
for name, res in [("then", then), ("now", now)]:
    res.scores.assign(when=name).to_csv(
        S / f"audit1107_scores_{name}.csv", index=False
    )
    res.quantiles.assign(when=name).to_csv(
        S / f"audit1107_quantiles_{name}.csv", index=False
    )
print("done")
