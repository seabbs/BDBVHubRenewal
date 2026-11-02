# Release backtest scratch scripts (rwnc-eval2)

Scratch drivers for the scored evaluation of `randomwalknowcast` against `randomwalk`, the renewal models and the hub teams.
They are not part of the package.

- `renewal_run.py OUT MODEL...` fits renewal models at the 10-24 release into a forecast cache.
- `run_eval.py OUT HUB_CLONE RENEWAL_CACHE [parts]` runs the release backtest (provinces, zones, the observation-model variants, the age-6 complete rule, the clock shift and the fan quantiles).
- `analyse.py OUT` writes the score tables.
- `figures.py OUT HUB_FILE` writes the figures.
- `report_tables.py OUT HUB_FILE` writes the Markdown tables for the report.

`HUB_CLONE` is a shallow clone of the public hub repository.
Run each from the repository root with `uv run --frozen python scripts/eval2/<script>`.
