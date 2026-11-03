# Setting up automated hub submissions

This bundle submits each week's forecast to the [BDBV Modeling Hub](https://github.com/InsightNet-US/BDBV-Modeling-Hub) as an automated pull request from `seabbs-bot`'s fork.
Nobody approves the PR in CI.
The gate decides, and the PR body states what was submitted and why.
It is safe to install as is, because `DRY_RUN` is on until you turn it off.

## What it does

| job | what |
| --- | --- |
| `detect` | Runs on schedule from Saturday 23:00 UTC (Sunday 00:00 Kinshasa) to Tuesday. It compares the hub's newest data commit and `as_of` with the ledger and stops unless there is new data for the open round. |
| `prepare` | Pins the hub data at one commit and builds the random walk candidate, which is also the baseline for the checks. |
| `fit` | A matrix over the renewal candidates (`bvd`, `renewalsp` and `main`), fitted in parallel. Each fit is cached under a content hash of the model source, the hub commit and its sampler settings. |
| `gate` | Walks the chain `bvd` → `renewalsp` → `main` → `randomwalknowcast` → `randomwalk`. A candidate is ruled out if its fit is past the convergence failure thresholds, its forecast is far from the recent data, or its file is invalid (format checks or hubValidations). |
| `submit` | Publishes the plot, builds the PR body, and opens or updates the PR. A dry run prints the body instead. |
| `record` | Live runs only. Writes the outcome to `state.json` on the `submission-ledger` branch and archives the round's files there. |
| `outcome` | Fails the run if the round was skipped or a job broke, so you are notified. |

Each run uploads these artefacts: `inputs`, `candidate-<id>` (7 days), `fit-<id>` (3 days, as in BVDOutbreakSize), and `gate` and `pr` (90 days).

## Before the first live run

### 1. The bot's fork

As `seabbs-bot`, fork the hub once:

```sh
gh repo fork InsightNet-US/BDBV-Modeling-Hub --clone=false
```

Then, in the fork's Settings → Actions → General, disable Actions.
The hub's own workflows should not run there.
The fork's `main` is never used or synced.
Each round's branch, `<model_id>-<reference_date>`, is cut from the hub's `main`.
The plots go on an orphan `submission-assets` branch of the fork.

### 2. The token

Create a **classic** personal access token on `seabbs-bot` with the scopes **`public_repo`** and **`workflow`**.

- `public_repo` is what `epiforecasts/cityforecasts` uses for the same job (`bin/submit_to_hub.sh`, secret `HUB_PR_TOKEN`).
  It pushes to the fork and opens PRs into the hub.
- `workflow` is needed when the hub's `main` has changed a file under `.github/workflows/` since the fork was made.
  The branch we push then carries that change, and GitHub refuses the push without this scope.
  The hub changes its workflows often (for example #65).
- A fine-grained token will not work.
  Fine-grained tokens cannot open pull requests on a repository the bot neither owns nor collaborates on.

Give the token an expiry, for example 90 days, and put a reminder in your calendar.
When it expires, the `submit` job fails with an authentication error, and the run fails visibly.

### 3. Secrets and variables in this repo

Settings → Environments → **New environment** `hub-submission`:

- Deployment branches: **Selected branches**, `main` only.
- Required reviewers: **none**.
  The gate is the approval.
- Environment secret: **`HUB_SUBMIT_TOKEN`**, the token above.
  It is not a repository secret, so only the `submit` job on `main` can read it.

Settings → Secrets and variables → Actions → **Variables**:

| variable | default when unset | meaning |
| --- | --- | --- |
| `DRY_RUN` | dry run | Set to `false` to go live. Any other value, or none, is a dry run. |
| `HUB_MODEL` | latest entry of `results/model-choice.yml` | Model name passed to `bdbvhubrenewal`. |
| `HUB_MODEL_ID` | `config/hub.toml` (`seabbs-renewal`) | Hub model id. Fallbacks are submitted under this id too. |
| `HUB_FORK` | `seabbs-bot/BDBV-Modeling-Hub` | The fork to push to. |
| `EXPECTED_LOCATIONS` | none | For example `CD,CD-IT,CD-NK,CD-HU,CD-TO`. The coverage check fails if any is missing. |
| `HUB_VALIDATIONS` | on | `false` skips the R hubValidations step and keeps the Python checks. |
| `FIT_RUNNER` | `ubuntu-latest` | Runner label for the fit jobs (see Runtimes). |

### 4. Repository settings

- Settings → Actions → General → Workflow permissions: **Read repository contents** is enough.
  Only the `record` job asks for `contents: write`, to push the `submission-ledger` branch.
  If a ruleset covers all branches, exclude `submission-ledger`.
- "Allow GitHub Actions to create and approve pull requests" is not needed.
  PRs are opened with the bot's token on another repo.

### 5. Tell the hub organisers

The first PR from `seabbs-bot` is a first-time contribution from a fork.
The hub's validation workflow will wait for a maintainer to click **Approve and run**.
Ask Justin Lessler or Claire Smith to approve it, and say that weekly bot PRs will follow.
The plan's open questions for the organisers also apply here (issue 3 in `seabbs/EpidemicDates`).

## Switching off DRY_RUN

1. Run **Actions → Submit to the BDBV hub → Run workflow** with the defaults.
   This is a dry run.
   Read the PR body in the `submit` job summary and the `gate` artefact.
2. For a past vintage outside the window, set `force` and `ignore_window`, and give a `reference_date`.
   These are allowed only in a dry run.
   This is also how to time the fits (see Runtimes).
3. Inside a window, with you watching, run it once with `dry_run: false`.
   Check the PR on the hub.
4. Set the repository variable `DRY_RUN` to `false`.
   Scheduled runs are then live.

To pause submissions, set `DRY_RUN` back to `true`, or disable the workflow.

## Being told when something goes wrong

No issues are opened anywhere.
A round that is skipped (every candidate failed), a broken job, or a PR the hub had already closed all make the `outcome` job fail.
The run's summary gives the reason, and the `gate` artefact has the details.
Ways to see the failure:

- **Default.** GitHub emails failed runs.
  For scheduled runs it notifies only the user who last changed the `cron` lines in `submit.yml`.
  Commits default to `seabbs-bot` here, so make the last edit to those lines yourself (`/commit --as-me`).
  Otherwise the bot gets the email.
- Settings → Notifications → System → Actions: set "Notify me on failed workflows only", by email or on the web and mobile.
- Watch the repository with the custom **Actions** event, or check the Actions tab from GitHub Mobile.
- Optional and not wired in: a Slack or email step on failure.
  It needs a webhook or SMTP secret.
  Add one to the `outcome` job if email is not enough.

Warnings do not fail the run.
They are stated in the PR body and the job summary.

Scheduled workflows in a public repository are switched off after 60 days without activity on the repository.
GitHub emails before it does this.
Ledger commits are on a side branch, so an occasional commit to `main` keeps the schedule alive.

## Runtimes and limits

GitHub-hosted `ubuntu-latest` on a public repository: 4 vCPUs, 16 GB RAM, 14 GB disk, and a **6-hour cap per job**.
Each fit job's timeout is at most 350 minutes, and stops 45 minutes before the window closes (`ci/policy.toml` `[timing]`).
The Actions cache holds 10 GB per repository and evicts entries unused for 7 days.
Artefacts are kept for 90 days at most, so the ledger branch is the durable archive.
Scheduled starts can be late by minutes, or dropped when GitHub is busy.
The 3-hourly schedule covers that.

Every fit runs 2 chains of 500 warm-up and 500 draws (`ci/policy.toml`).
On the 2026-10-24 vintage, with vectorised chains on 2 cores of archie, 4 chains of 1000 warm-up and 1000 draws took 130 minutes of sampling for the main fit.
Nearly every draw reaches the maximum tree depth, so cost follows the number of NUTS iterations, and the current settings take about a quarter of that, 35 to 45 minutes.
The single-patch fallback (`renewalsp`) took 21 minutes of sampling for its six independent fits (the country and five provinces), so it is ready before the main fit.
Hosted runners are 1.2 to 1.5 times slower, so each fit job takes about 1 to 1.5 hours with setup.
These times are to be re-measured on the current model.
Time a real fit with a dry run on a past vintage (see above), and read the `Fit took N s` line in the job summary.

The chain runs prepare (random walk), then every renewal fit in parallel, then the gate.
Scheduled runs check once each morning, at 06:00 UTC on Sunday, Monday and Tuesday, after the Saturday 23:00 UTC run as the window opens.
A round is submitted once: when the ledger records its PR, later scheduled runs stop, and a manual dispatch resubmits.
From the Tuesday 06:00 UTC check there are about 17 hours to the close, which also leaves the organisers the US working day to approve the PR's checks.
Every fit job's timeout leaves `reserve_minutes` (45) before the close for the gate and the pull request.
So a fit that runs to its timeout still reaches the fallback or random walk before the close.
`max_tree_depth` is set per candidate in `ci/policy.toml`.

If fits come close to the cap:

1. Make each fit cheaper.
   Use fewer warm-up iterations or draws, start from last week's posterior, and keep the JAX compilation cache warm (`setup-forecast-env` restores it).
2. Split the matrix further where the model allows.
   The single-patch fallback fits each province on its own, so it can fit per province group as separate legs.
   The coupled main model cannot.
3. Start earlier.
   Data usually land on Friday or Saturday, and the window opens Sunday.
   The last scheduled start on Tuesday is 06:00 UTC, which leaves about 17 hours before the window closes for the fits and the gate.
4. Use a bigger machine through `FIT_RUNNER`.
   GitHub's larger runners need a paid Team or Enterprise organisation, for example if the repo moves to `epiforecasts`.
   A self-hosted runner works too.
   The workflow only runs on `schedule` and `workflow_dispatch`, never on fork pull requests, so a self-hosted runner on this public repo is not exposed to outside code.

## The fallback chain

The chain is set in `ci/policy.toml`.
Problems are handled where they can be, and the PR body leads with a `Handled:` line saying what was handled and how.
For example: "Handled: delays: used the 2026-10-17 vintage, the latest available."
When a fallback is used, the status line names it and why, for example "Fallback model used (random walk baseline): main model was ruled out (convergence: min bulk ESS is 8, past the failure threshold 10)."
The same text goes to `provenance.json` in the `gate` artefact, to the job summary and to the ledger.
A round is skipped only if even random walk fails hub validation.

A suggested general line for the model metadata's `methods_long`:

> Submissions are produced automatically each week.
> If a renewal model's fit does not converge, we submit the next, simpler renewal model, and a random walk baseline if none converges.
> Each submission's pull request states which was used and why.

## Where this approach comes from

- **Fit monitoring and the matrix.** These follow `epiforecasts/BVDOutbreakSize`: `docs.yml`, `docs/fits/convergence.jl`, `docs/fits/check_convergence.jl` and `scripts/fetch_fits.sh`.
  That means the same fail and warn thresholds, two levels, and treating a fit that cannot be judged as a failure.
  It also means one matrix job per fit, content-hash fit caches with separate restore and save, a once-a-minute progress heartbeat, short-lived fit artefacts, and a report in the job summary.
- **Fork and PR mechanics.** These follow `epiforecasts/cityforecasts` (`bin/submit_to_hub.sh` and `.github/workflows/submit-to-hub.yaml`).
  That means a classic PAT, a branch per round cut from the hub's `main`, a force push, finding the open PR by head branch and fork owner, and `gh pr create --head owner:branch`.
- **Scheduled fits on hosted runners with artefacts.** `seabbs/ecdc-weekly-growth-forecasts` (`.github/workflows/forecast.yaml`) shows this works.
  It did not open PRs.
  Its submissions were made by hand.
