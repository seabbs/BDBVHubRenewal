# Contributing

Set up with `task setup`.
`task --list` shows every task.

## Checks

Run these before sending a change:

```bash
task lint       # ruff and strict mypy
task test-all   # every test, the model fit tests included
task ci-test    # the submission workflow tests
task ci-dry     # the submission workflow dry runs
task docs       # the site, built in strict mode
```

`task test` leaves out the model fit tests (marked `slow`).
The CI runs them on every push to main.
Any change to `src/bdbv_hub_renewal/model/`, the simulator or the renewal forecaster needs `task test-all` to pass first.

## Commits

Write a failing test, commit it, make it pass, commit again.
Keep each commit to one change.
