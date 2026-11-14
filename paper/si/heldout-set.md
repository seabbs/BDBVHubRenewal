#### The held-out synthetic set

The held-out set is 100 outbreaks simulated with `scripts/finetune_simulate.R` on the r2-cal5 grid, 25 for each of the seeds 9001, 9002, 9003 and 9004.
The fine-tune never used these seeds, so the held-out outbreaks are new draws from the grid it was trained on.
If an outbreak's design failed five times and was skipped, a further seed tops the set up to 100.
They are written to `output/finetune/heldout`, as `counts-<seed>.csv.gz` and `settings-<seed>.csv`.
The evaluation workflow (`paper/workflow/Snakefile`) runs the models on the set in its `paper` target only.

The held-out set was reserved for the final evaluation.
No model was scored on it during development, and no forecast or score from it was looked at.
One early run of the pipeline forecast three held-out outbreaks before this rule was set, and its outputs were deleted unread.
We checked the pipeline end to end on a separate validation set of five outbreaks with seed 7001, in `output/finetune/validation`.

The German benchmark is split the same way.
Its nowcast dates from 2022-02-10 are held out, and development runs read no forecast from them.
Development runs read the German data only up to the release D = 40 days after the last development date, 2022-03-21, which the truth of that date needs.
The evaluation workflow checks both, and fails a development run that reads later data.

##### Protocol

Each outbreak is released weekly, on one weekday from Wednesday to Saturday drawn from its id, as for the fine-tune's training blocks.
A release's cut-off is its origin date.
Each outbreak has four origins, spread evenly from its sixth release to the last release at least its longest delay (3 to 12 weeks) plus 5 weeks before its last report.
Each forecast covers eight weeks of onsets, from four weeks before the week of the cut-off to three weeks after it.
Every model forecasts weekly totals by onset week, for the country and each place, with 200 samples (`n_samples` in the workflow config).
The Chronos-2 models keep their own release lag of 4 weeks (weekly) or 28 days (daily).
The scores use rolling truth, the count reported D days after the end of a target week, with D the outbreak's own 98% reporting delay in whole days.
The final simulated count is kept as well.
Scores are relative to `randomwalknowcast`.
