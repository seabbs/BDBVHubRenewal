# Contributing

Setting up, checking and documenting changes to the package.
{ .page-subtitle }

| Task | What it does |
|---|---|
| `task setup` | Install the package with the dev and docs groups. |
| `task test` | Run the tests with coverage. |
| `task test-slow` | Run the tiny real model fits. |
| `task lint` | Lint with ruff, check formatting and type-check with mypy. |
| `task format` | Format and fix lint where safe. |
| `task docs` | Build the docs strictly into `_site/`. |
| `task ci-test` | Lint, type-check and test the submission CI scripts in `ci/`. |

Pre-commit hooks run the same checks on each commit.
New behaviour comes with a test, and docs follow UK English with one sentence per line.
New models are added as on [Adding a model](adding-a-model.md).
