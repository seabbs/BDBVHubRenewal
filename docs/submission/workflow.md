# Submission workflow

How a round goes from new hub data to a pull request.
{ .page-subtitle }

`.github/workflows/submit.yml` runs while the hub's window is open (Sunday to Tuesday, Kinshasa time) and on demand.
It submits only the hub models opted in ([Opting in](opting-in.md)), and pushes nothing while `DRY_RUN` is on.

## Jobs

| job | what it does |
| --- | --- |
| `detect` | Stops unless the open round has new hub data, or if its PR is already open or merged. A change to the model code on main since the round's PR reruns it and updates the PR, as does a manual dispatch. |
| `freeze` | Freezes the page of every round whose pull request has merged. |
| `prepare` | Pins the data, builds `randomwalk` and `randomwalknowcast`, and decides which hub models run. |
| `baseline` | The baseline hub model: checks and submits `randomwalknowcast`, or skips the round if it is ruled out. |
| `fit` | The renewal hub model: fits `bvd`, `renewalsp` and `main` in parallel. |
| `gate`, `submit` | Walks the chain, validates, and opens or updates the pull request. |
| `record` | Live runs only: writes the round to the `submission-ledger` branch. |
| `outcome` | Fails the run if a round was skipped or a job broke. |

## renewal chain

The first candidate that clears the checks is submitted, under `seabbs-renewal`.

The chain runs from the most to the least complex renewal model, then the random walks.

1. `bvd`: `renewalbvd`, the renewal model informed by BVDOutbreakSize Rt.
2. `renewalsp`: independent single-patch fits.
3. `main`: `renewal`, the province renewal model.
4. `randomwalknowcast`: random walk on a nowcast.
5. `randomwalk`.

`[submission] model` in `config/hub.toml` says where the chain starts.
Every fit is 2 chains of 500 warm-up, run in parallel, with no refit.
`renewalbvd` takes 1000 draws at target acceptance 0.95, and the others 500 at 0.9.
Fit timeouts are set against the window's close (`[timing]` in `ci/policy.toml`).

## Checks

An invalid file, a fit past the convergence failure thresholds or a forecast far from the recent data rules a candidate out.
Thresholds are in `ci/policy.toml`.

| check | rules out | warns |
| --- | --- | --- |
| max R-hat | > 1.1 | > 1.05 |
| min bulk and tail ESS | < 25 | < 100 |
| divergent share | > 5% | > 1% |
| 90% predictive coverage | | < 0.70 |
| schema, values, 100 samples per trajectory | any | |
| hub validation (Python and hubValidations) | any error | |
| plausible (median and 97.5% quantile against the recent maximum) | any | |
| national equals the provinces plus the unassigned part in `national.json` | any | |
| bounds, growth, against observed, random walk and last week | | yes |

The [hub specification](hub-spec.md) lists the hub's rules.

## Pull request

It goes from `seabbs-bot/BDBV-Modeling-Hub`, branch `<model_id>-<reference_date>`, into the hub's `main`.
The body shows what was submitted, one status line, the plot and a link to the round's page.
The details sit in collapsed sections.

Every row carries a `forecast_date`, the day the PR is opened.
If the hub's `main` already has this model's file for the round, the gate reads it (read only) and writes the file the PR holds as the hub's rows unchanged plus the new rows with one new `forecast_date` ([hub specification](hub-spec.md)).
The same file goes through both validators.
The submit step stops, without pushing, if the hub's file changed after the gate.

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
