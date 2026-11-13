# Evidence for #200: health zones in the Chronos-2 block

Code: branch `fm-zones` (on `fm-chronos` at `313e5b9`), commit `089ee45`.
Data: hub releases 2026-08-18 to 2026-11-08.

## Does a model's own zone forecast beat the allocator?

Zone backtest origins 2026-10-09 to 2026-11-01 (2026-08-18 is a `skip_origins` release).
Each Chronos-2 model is fitted twice at each origin (`scripts/zone_fits.py`).

- `own`: the block holds every hub target, the 519 zones included, and the zones are the model's own, made to sum to its provinces as in a hub file.
- `allocator`: the block holds the country and provinces only, as on main, and `ZoneAllocator` splits the provinces across the zones.
- `own_raw`: the model's own zones before they are made to sum to its provinces.

`randomwalknowcast` is its zone-level run made to sum to its provinces, as in its hub file.
Only zones with cases at the origin are scored, as in main's zone backtest.

Scoring is main's complete target on the rolling truth (`scripts/score_zones.py`).
A zone-week's truth is its count in the first release whose cut-off is at least $D = 35$ days after the week (`resolve_truth_delay`, 98% level), 0 where that release has none.
Weeks no release has reached yet are not scored.

Relative WIS against `randomwalknowcast`, with relative log WIS in brackets, on the 65 zone-weeks all models forecast (origin 2026-10-17, horizon -4).
`results/zone-relative.csv`.

| Model | own | allocator | own_raw |
|---|---|---|---|
| fm_chronos_weekly_complete | 2.87 (2.61) | 2.21 (1.31) | 3.33 (2.83) |
| fm_chronos_weekly_ragged | 2.20 (2.23) | 1.43 (1.00) | 2.24 (2.44) |
| fm_chronos_weekly_ragged_ft | 2.13 (2.22) | 1.68 (1.08) | 2.11 (2.29) |
| fm_chronos_daily_complete | 3.37 (3.46) | 2.71 (1.61) | 5.24 (4.42) |
| fm_chronos_daily_ragged | 2.51 (2.89) | 2.59 (1.69) | 5.00 (4.42) |
| fm_chronos_daily_ragged_ft | 3.16 (3.27) | 2.69 (1.66) | 5.48 (4.52) |

The rolling truth leaves few zone-weeks: `randomwalknowcast` forecasts from ref - 4 weeks and has no delay at 2026-10-09.
On every zone-week both arms forecast, own over allocator (`scripts/own_vs_allocator.py`, `results/own-vs-allocator.csv`):

| Model | n | WIS | log WIS |
|---|---|---|---|
| fm_chronos_weekly_complete | 261 | 1.55 | 2.14 |
| fm_chronos_weekly_ragged | 517 | 1.13 | 1.55 |
| fm_chronos_weekly_ragged_ft | 517 | 0.90 | 1.37 |
| fm_chronos_daily_complete | 261 | 1.25 | 1.89 |
| fm_chronos_daily_ragged | 517 | 1.03 | 1.67 |
| fm_chronos_daily_ragged_ft | 517 | 1.11 | 1.79 |

The allocator beats the models' own zones on log WIS for every model, and on WIS for all but `fm_chronos_weekly_ragged_ft`.
Making the own zones sum to the provinces helps the daily models most.

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
