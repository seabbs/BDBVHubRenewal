# The held-out synthetic set

The held-out set is 100 outbreaks simulated with `scripts/finetune_simulate.R`, 25 for each of the seeds 9001, 9002, 9003 and 9004.
The fine-tune used the seeds 1000 to 1079, so the held-out outbreaks are new draws from the same scenario grid.
They are written to `output/finetune/heldout`, as `counts-<seed>.csv.gz` and `settings-<seed>.csv`.
The protocol is in `paper/synthetic.py`, and `paper/run_synthetic.py` runs the models on the set.

The held-out set was reserved for the final evaluation.
No model was scored on it during development, and no forecast or score from it was looked at.
One early run of the pipeline forecast three held-out outbreaks before this rule was set, and its outputs were deleted unread.
We checked the pipeline end to end on a separate validation set of five outbreaks with seed 7001, in `output/finetune/validation`.

## Protocol

Each outbreak is released weekly, on one weekday from Wednesday to Saturday drawn from its id, as for the fine-tune's training blocks.
A release's cut-off is its origin date.
Each outbreak has four origins, spread evenly from its sixth release to the last release at least 13 weeks before its last report.
Each forecast covers eight weeks of onsets, from four weeks before the week of the cut-off to three weeks after it.
Every model forecasts weekly totals by onset week, for the country and each place, with 200 samples.
The Chronos-2 models keep their own release lag of 4 weeks (weekly) or 28 days (daily).
The scores use rolling truth, the count reported D days after the end of a target week, with D the outbreak's own 99% reporting delay in whole days.
The final simulated count is kept as well.
Scores are relative to `randomwalknowcast`.
