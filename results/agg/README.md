# Evidence for #222: joint block or aggregation for Chronos-2

Code: `bdbvhubrenewal.evaluation.groupings` and `scripts/agg/` on this branch.
Fine-tuned models are round 1 (`chronos-ft-2c2b22a03107`) unless named `_n1000` (`chronos-ft-0cea0acef1ce`).

Options:

- `joint`: country, provinces and zones (Germany: Germany, states and age groups) in one Chronos-2 block.
- `level`: one block per level.
- `zones_summed`: the lowest level alone, summed draw by draw to the levels above.
- `zones_copula`: the same draws paired by the rolling copula of fm-copula-rolling (`RollingCopula`), then summed.
- `reference`: randomwalknowcast.

BDBV: release backtest origins 2026-10-09 to 2026-11-01, complete target on the rolling truth at D = 35 days (98% level).
Zones with cases at the origin are scored on each zone-week's count in the first release 35 days after it.
Germany: development half, 2021-11-22 to 2022-02-09, quantile scores on the 40-day truth.

Files:

- `aggregation-scores.csv.gz`: one row per scored cell, in the paper's schema (`aggregation_rows`).
- `bdbv-joint-samples.csv.gz`: the `joint` samples of the six Chronos-2 models, country and provinces, in main's forecasts layout.
- `bdbv-relative.csv`: relative WIS and log WIS against randomwalknowcast, by level, on the rolling truth and against the latest release (partly incomplete).
- `germany-relative-round1-quantiles.csv`: relative WIS of the round-1 arms on Germany, from quantile files.
