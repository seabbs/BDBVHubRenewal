# Contributing

Set up with `task setup`.
`task --list` shows every task.

## Checks

Run these before sending a change:

```bash
task lint             # ruff and strict mypy
task test-all         # every test, the model fit tests included
task hub:ci-test      # the submission workflow tests
task dry:submit       # the submission workflow dry runs
task docs             # the site, built in strict mode
```

`task test` leaves out the model fit tests (marked `slow`).
The CI runs them weekly and on demand.
Any change to `packages/nowcastrenewal/`, the simulator or the renewal forecaster needs `task test-all` to pass first.

## Tasks

The root Taskfile keeps the top-level tasks (setup, test, lint, format, docs, serve, getting-started, paper) and includes one Taskfile per project from `taskfiles/`, by namespace: `nowcast-utils` (`utils`), `nowcast-eval` (`eval`), `nowcast-foundation` (`foundation`), `nowcast-workflow` (`workflow`), `bdbv-hub` (`hub`) and `paper` (`paper/Taskfile.yml`).
The short names work too: `task foundation:finetune` is `task nowcast-foundation:finetune`.
The named dry runs are `task dry:tick`, `task dry:paper`, `task dry:submit` and `task dry:fit-store`.

The tasks' names before the project Taskfiles:

| Old | New |
|---|---|
| `paper:render` | `paper` |
| `data` | `nowcast-utils:data` |
| `data-bvd` | `nowcast-utils:data-bvd` |
| `backtest` | `nowcast-eval:backtest` |
| `backtest-lopo` | `nowcast-eval:backtest-lopo` |
| `backtest-scenarios` | `nowcast-eval:backtest-scenarios` |
| `backtest-zones` | `nowcast-eval:backtest-zones` |
| `results` | `nowcast-eval:results` |
| `score-history` | `nowcast-eval:score-history` |
| `germany` | `nowcast-eval:germany` |
| `compare-hub` | `nowcast-eval:compare-hub` |
| `plots` | `nowcast-eval:plots` |
| `simulate` | `nowcast-foundation:simulate` |
| `push` | `nowcast-foundation:push` |
| `release` | `nowcast-foundation:release` |
| `finetune` | `nowcast-foundation:finetune` |
| `publish` | `nowcast-foundation:publish` |
| `score` | `nowcast-foundation:score` |
| `land` | `nowcast-foundation:land` |
| `dashboard` | `nowcast-foundation:dashboard` |
| `dashboard:static` | `nowcast-foundation:dashboard-static` |
| `all` | `nowcast-workflow:all` |
| `tick` | `nowcast-workflow:tick` |
| `run` | `nowcast-workflow:run` |
| `schedule` | `nowcast-workflow:schedule` |
| `monitor` | `nowcast-workflow:monitor` |
| `dag` | `nowcast-workflow:dag` |
| `progress` | `nowcast-workflow:progress` |
| `report` | `nowcast-workflow:report` |
| `workflow:check` | `nowcast-workflow:check` |
| `nowcast` | `bdbv-hub:nowcast` |
| `metadata` | `bdbv-hub:metadata` |
| `validate-hub` | `bdbv-hub:validate` |
| `submission-preflight` | `bdbv-hub:preflight` |
| `ci-test` | `bdbv-hub:ci-test` |
| `ci-dry` | `bdbv-hub:ci-dry` |
| `ci-fit` | `bdbv-hub:ci-fit` |
| `fit-store` | `bdbv-hub:fit-store` |
| `fit-store-plan` | `bdbv-hub:fit-store-plan` |

## Commits

Write a failing test, commit it, make it pass, commit again.
Keep each commit to one change.

## Development log

`results/development-log.csv` records each decision about the model, the evaluation, the submission or the docs, with its date, reason and evidence (a commit, issue or results file).
Add a row in the same change when you make a decision of this kind.
The Model development section of the docs shows the log, newest first.
