# Submission workflow

How a round goes from new hub data to a pull request.
{ .page-subtitle }

`.github/workflows/submit.yml` runs while the hub's window is open (Sunday to Tuesday, Kinshasa time) and on demand.
It submits only the hub models opted in ([Opting in](opting-in.md)), and pushes nothing while `DRY_RUN` is on.

## Jobs

| job | what it does |
| --- | --- |
| `detect` | Stops unless the open round has new hub data. |
| `freeze` | Freezes the page of every round whose pull request has merged. |
| `prepare` | Pins the data, builds `persistence` and `persistencenow`, and decides which hub models run. |
| `baseline` | The baseline hub model: checks and submits `persistencenow`, or `persistence` if it is ruled out. |
| `fit` | The renewal hub model: fits `main`, `renewalsp` and the `bvd` shadow in parallel. |
| `triage`, `retry` | Refits `main` once if needed and if the window leaves time. |
| `gate`, `submit` | Walks the chain, validates, and opens or updates the pull request. |
| `record` | Live runs only: writes the round to the `submission-ledger` branch. |
| `outcome` | Fails the run if a round was skipped or a job broke. |

## renewal chain

The first candidate that clears the checks is submitted, under `seabbs-renewal`.

1. `main`: the renewal model.
2. `retry-1`: one refit with target acceptance 0.95 and jittered inits.
3. `renewalsp`: independent single-patch fits, fitted beside `main`.
4. `persistencenow`: persistence on a nowcast.
5. `persistence`.

Every fit is 2 chains of 500 warm-up and 500 draws.
Fit timeouts and the retry are set against the window's close (`[timing]` in `ci/policy.toml`).

## Checks

Only an invalid file or a clearly broken fit rules a candidate out.
Thresholds are in `ci/policy.toml`.

| check | rules out | refits once | warns |
| --- | --- | --- | --- |
| max R-hat | > 1.5 | > 1.1 | > 1.05 |
| min bulk and tail ESS | < 10 | < 25 | < 100 |
| divergent share | > 20% | > 5% | > 1% |
| 90% predictive coverage | | | < 0.70 |
| schema, values, 100 samples per trajectory | any | | |
| hub validation (Python and hubValidations) | any error | | |
| bounds, growth, against observed, persistence and last week | | | yes |

The national rows are rebuilt as the sum of the provinces plus the unassigned component.
The [hub specification](hub-spec.md) lists the hub's rules.

## Pull request

It goes from `seabbs-bot/BDBV-Modeling-Hub`, branch `<model_id>-<reference_date>`, into the hub's `main`.
The body shows what was submitted, one status line, the plot and a link to the round's page.
The details sit in collapsed sections.

A round's page is frozen once its pull request merges, and later docs builds do not change it ([Rounds](rounds/index.md)).

<div class="pr-body-example" markdown>

--8<-- "pr-body.md"

</div>

## Running it locally

| command | what it does |
| --- | --- |
| `task ci-dry` | Every path through the workflow on fixture data. |
| `task ci-test` | Tests of the CI scripts. |
| `task ci-fit` | Fits and forecasts one config as a candidate does. |
| `task submission-preflight` | Lists what a live run still needs. |
| `gh workflow run submit.yml -R seabbs/BDBVHubRenewal -f force=true -f ignore_window=true -f reference_date=<date>` | A dry run on a past round. |
