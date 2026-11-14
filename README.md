# Evidence for #200: health zones in the Chronos-2 block

Code: branch `fm-zones` (on `fm-chronos` at `313e5b9`), commit `756635b`.
Data: hub releases 2026-08-18 to 2026-11-08.

## Does a model's own zone forecast beat the allocator?

Zone backtest origins 2026-10-09 to 2026-11-01 (2026-08-18 is a `skip_origins` release).
Each Chronos-2 model is fitted twice at each origin (`scripts/zone_fits.py`).

- `own`: the block holds every hub target, the 519 zones included, and the zones are the model's own, which the model makes sum to its provinces with each zone keeping its published count, as in its hub file.
- `allocator`: the block holds the country and provinces only, as on main, and `ZoneAllocator` splits the provinces across the zones.

`randomwalknowcast` is its zone-level run made to sum to its provinces, as in its hub file.
Only zones with cases at the origin are scored, as in main's zone backtest.

Scoring is main's complete target on the rolling truth (`scripts/score_zones.py`).
A zone-week's truth is its count in the first release whose cut-off is at least $D = 35$ days after the week (`resolve_truth_delay`, 98% level), 0 where that release has none.
Weeks no release has reached yet are not scored.

Relative WIS against `randomwalknowcast`, with relative log WIS in brackets, on the 65 zone-weeks all models forecast (origin 2026-10-17, horizon -4).
`results/zone-relative.csv`.

| Model | own | allocator |
|---|---|---|
| fm_chronos_weekly_complete | 2.18 (1.75) | 2.19 (1.29) |
| fm_chronos_weekly_ragged | 1.29 (1.09) | 1.42 (0.98) |
| fm_chronos_daily_complete | 2.94 (2.59) | 2.69 (1.58) |
| fm_chronos_daily_ragged | 2.11 (1.93) | 2.57 (1.65) |

The fine-tuned models, apart, as they were trained on blocks of at most 26 provinces (five daily) and here see 546 targets:

| Model | own | allocator |
|---|---|---|
| fm_chronos_weekly_ragged_ft | 1.38 (1.18) | 1.67 (1.06) |
| fm_chronos_daily_ragged_ft | 2.79 (2.44) | 2.67 (1.63) |

The rolling truth leaves few zone-weeks: `randomwalknowcast` forecasts from 4 weeks before the reference date and has no delay at 2026-10-09.
On every zone-week both arms forecast, own over allocator (`scripts/own_vs_allocator.py`, `results/own-vs-allocator.csv`):

| Model | n | WIS | log WIS |
|---|---|---|---|
| fm_chronos_weekly_complete | 261 | 0.96 | 1.21 |
| fm_chronos_weekly_ragged | 517 | 0.84 | 1.05 |
| fm_chronos_weekly_ragged_ft | 517 | 0.71 | 0.96 |
| fm_chronos_daily_complete | 261 | 1.04 | 1.36 |
| fm_chronos_daily_ragged | 517 | 0.85 | 1.24 |
| fm_chronos_daily_ragged_ft | 517 | 0.99 | 1.44 |

Own zones have lower WIS than the allocator for five of six models, and higher log WIS for all but `fm_chronos_weekly_ragged_ft`.
No model's zones beat `randomwalknowcast`'s.

## Time and memory per fit

One fit at the 2026-11-08 release, as the fit store builds the task, in its own process (`scripts/fit_one.py`, `scripts/time_all.sh`).
`series` counts the target series and the lag series.
archie's CPU (16 cores, 62 GB) was shared, at load 16 to 59, so its times are upper bounds.
`results/timings-cpu.txt`, `results/timings-mps.txt`.

| Model | series, zones | CPU fit s, zones | CPU peak GB, zones | CPU fit s, no zones | MPS fit s | MPS GPU GB |
|---|---|---|---|---|---|---|
| fm_chronos_weekly_complete | 546 | 12 | 1.7 | 5 | 0.8 | 1.6 |
| fm_chronos_weekly_ragged | 846 | 16 | 1.6 | 5 | 0.5 | 1.6 |
| fm_chronos_weekly_ragged_ft | 846 | 15 | 1.7 | 4 | 0.5 | 1.6 |
| fm_chronos_daily_complete | 546 | 38 | 2.1 | 6 | 2.1 | 2.7 |
| fm_chronos_daily_ragged | 2,646 | 138 | 4.3 | 15 | 10.4 | 14.9 |
| fm_chronos_daily_ragged_ft | 2,646 | 181 | 4.2 | 683* | 9.7 | 14.9 |

\* At load 59, while other jobs filled archie's memory.
Loading Chronos-2 takes another 4 to 13 s.
Lag series are kept only for the 75 places with cases, so daily ragged has 546 + 75 × 28 = 2,646 series, not the 15,834 of every place.

The backtest fits (`results/backtest-fit-timings-mps.csv`) ran on the laptop's MPS, as archie stopped answering while other jobs filled its memory.

## Reproduce

From an `fm-zones` checkout with the hub data in `data/` and `uv sync --all-groups`:

```sh
export GH_TOKEN=$(gh auth token)
uv run python scripts/zone_fits.py --out fits --models fm_chronos_weekly_complete fm_chronos_weekly_ragged fm_chronos_weekly_ragged_ft fm_chronos_daily_complete fm_chronos_daily_ragged fm_chronos_daily_ragged_ft
uv run python scripts/score_zones.py --fits fits --rw results/rw-zones --out results
uv run python scripts/own_vs_allocator.py results/zone-scores.csv.gz
scripts/time_all.sh results/timings-cpu.txt
```
