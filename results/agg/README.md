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

- `aggregation-scores.csv.gz`: one row per scored cell, in the paper's schema (`aggregation_rows`), BDBV (`ebola`) and the German development half (`germany`).
- `aggregation-cost.csv`: one fit per arm, each in its own process, on the laptop CPU with one thread: BDBV at origin 2026-11-01, Germany at 2022-02-09 (both of its tasks, or the merged task for `joint`).
  `series` counts target and lag series per block, `+` between blocks; `memory_gb` is the process's peak resident memory, Chronos-2 loaded; `seconds` is the fit alone.
  `zones_copula` repeats `zones_summed`: its pairing is not timed.
- `bdbv-joint-samples.csv.gz`: the `joint` samples of the six Chronos-2 models, country and provinces, in main's forecasts layout.
- `bdbv-relative.csv`: relative WIS and log WIS against randomwalknowcast, by level, on the rolling truth and against the latest release (partly incomplete).
- `germany-relative.csv`: relative WIS and log WIS against randomwalknowcast, by level, on the forecasts every arm made.

The pairing used fm-copula-rolling's copula at c83d85461, before its per-cell seeding.

## Fine-tuned checkpoints (branch agg-slots)

`aggregation-scores.csv.gz` has a `checkpoint` column: `zero-shot` for the zero-shot models and randomwalknowcast, and a checkpoint's label (`slots-<hash>`, or a plain settings hash) for `fm_chronos_<source>_ragged_ft`.
The zero-shot and reference rows are those above, unchanged.
A `slots_rule` column says how a checkpoint with per-series slots gave them out.
`none`: no slots (zero-shot, the reference, or a checkpoint without slots).
`all`: every place of each block has a slot, as in forecasting; Germany's blocks and BDBV's country and province blocks of `level`.
`cases-only`: a block held more places than the 128 slots, so only places with cases in its context (in their counts or, for cases in the newest weeks only, their lag series) have slots and the others none (slot -1, a zero vector, as base Chronos-2; `groupings.cased_group_slots`).
That is BDBV's `joint` (546 places), `zones_summed` and `zones_copula` (519 zones), and the zone rows of `level`.
These blocks are outside the slots' training range: training blocks held about 95 targets.
Slots follow Chronos-2's row order (`foundation._group_slots`, from slots-loader-roworder).
Runs queue, one checkpoint at a time; each adds or replaces its own rows and keeps every other checkpoint's.
A run takes the newest eligible slots checkpoint when it starts and skips older ones, so not every published checkpoint gets rows.
Eligible checkpoints come from the slots run with random-init slots (settings `series_slots.init` other than `zero`); the zero-init run's checkpoints (chronos-slots-7d7c5d9f247e, chronos-slots-06dbdcc2b48d) are not run. Newest means the most `n_outbreaks`.
Places with cases are those with a case in their counts or in their lag series, so a place whose cases are all in the newest, partly reported weeks keeps a slot.
Each run commits its BDBV rows first (interim, about 2 h), then the German rows.
The round-1 and n1000 fine-tuned rows are dropped, as their checkpoints are not the paper's.
`aggregation-cost.csv` is unchanged: a fit's cost depends on the block sizes and Chronos-2's size, which the LoRA adapter does not change (the round-1 rows above match the zero-shot ones).
`scripts/agg/run_finetuned.sh LABEL` fits, scores and adds one checkpoint's rows, on the same data as the zero-shot rows: the hub at ad1bb4af13bf (the 2026-11-08 release, D = 35), BDBV origins 2026-10-09 to 2026-11-01, the German development half and its 40-day truth.
randomwalknowcast is not rerun; each new arm keeps the cells it has here, and the runner checks every arm has all of them.
The paired arm uses `forecasters/copula.py` as at fm-copula-rolling's c83d85461, before its per-cell seeding; the file on this branch is that version byte for byte.
Before it fits, the runner refits zero-shot `fm_chronos_daily_ragged`, joint, at origins 2026-10-17 (every level scored, zones included) and 2026-11-01, and compares its rows with those here.

Checkpoints added:
- `slots-ea31bba0123e` (release `chronos-slots-ea31bba0123e`): rows added 2026-11-17 by `scripts/agg/run_finetuned.sh slots-ea31bba0123e` from code at d181232c4.
