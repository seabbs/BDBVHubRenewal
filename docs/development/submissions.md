# Submissions

How each week's forecast reaches the BDBV Modeling Hub.
{ .page-subtitle }

!!! warning "Dry run, with stand-in models"
    The workflow runs in dry-run mode until Sam switches it on (see [Setup](#setup)).
    It builds everything and prints the pull request it would open, but pushes nothing.
    The PyRenew renewal model is not integrated yet.
    Until it is, the `main` and `fallback` fit configs are **stand-ins** that run the persistence baseline, and submissions go in as `epiforecasts-persistence`.

## How the pipeline works

The workflow is `.github/workflows/submit.yml`.
It runs on a schedule while the hub's submission window is open, and on demand.
The window is Sunday 00:00 to Tuesday 23:59:59 Kinshasa time (UTC+1), so the schedule runs from Saturday 23:00 UTC, every three hours on Sunday and Monday, and on Tuesday until 14:00 UTC.
Nobody approves the pull request in CI.
The gate decides, and the pull request body says what was submitted and why.

| job | what it does |
| --- | --- |
| `detect` | Compares the hub's newest data commit and `as_of` with the ledger. Stops unless there is new data for the open round. |
| `prepare` | Pins the hub data at one commit (`bdbv-forecasts data --ref <sha>`), checks national coherence and builds the persistence candidate, which is also the baseline for the checks. |
| `fit` | One matrix job per primary candidate (`main` and `fallback`), in parallel. Each runs `bdbv-forecasts fit`, then `nowcast --fit`, the checks and the plots. Fits are cached under a hash of the model source, the hub commit and the sampler settings. |
| `triage` | Says whether `main` cleared its checks. |
| `retry` | Only if `main` failed: refits with more conservative sampler settings. |
| `gate` | Walks the fallback chain and submits the first candidate that clears convergence, the posterior predictive checks, the forecast checks and hub validation. |
| `submit` | Publishes the plot, builds the pull request body, and opens or updates the pull request from `seabbs-bot`'s fork. A dry run prints the body instead. |
| `record` | Live runs only. Writes the outcome to `state.json` on the `submission-ledger` branch and archives the round's files there. |
| `outcome` | Fails the run if the round was skipped or a job broke, so GitHub sends a failure notice. |

The workflow calls this repository only through the `bdbv-forecasts` command line.

| command | used by |
| --- | --- |
| `bdbv-forecasts data --ref <sha>` | `prepare` |
| `bdbv-forecasts coherence --vintage <as_of> --out <json>` | `prepare` |
| `bdbv-forecasts fit --model <m> --config <c> --vintage <as_of> --sampler-json <json> --out <dir>` | `fit`, `retry` |
| `bdbv-forecasts nowcast --model <m> --config <c> --fit <dir> --vintage <as_of> --out <dir>` | `fit`, `retry`, `prepare` (persistence, without `--config`) |
| `bdbv-forecasts plot-submission --submission <csv> --vintage <as_of> --out <png>` | every candidate |
| `bdbv-forecasts plot-ppc --ppc <json> --out <png>` | candidates with a posterior |
| `bdbv-forecasts metadata --model <m> --out <dir>` | `gate` |
| `bdbv-forecasts validate` and `scripts/validate-hub.R` | `gate`, through `ci/steps/validate_hub.sh` |
| `bdbv-forecasts --version` | the pull request body |

`fit` writes the fit directory: `fit.json`, `diagnostics.json`, `ppc.json`, `progress.log` and whatever the forecast needs.
A model without a sampler, such as persistence, writes diagnostics with `"sampler": "none"` and `"status": "not-applicable"`, and a `ppc.json` with `"available": false`.
The gate reads both as not applicable.
The formats are specified in `ci/INTEGRATION.md`.

A fit config is registered in [`bdbv_forecasts.fitting`][bdbv_forecasts.fitting].

### Plugging in the model

The model integration plugs PyRenew in by registering `main` and `fallback` with `register_config(..., replace=True)`.
Each [`FitConfig`][bdbv_forecasts.fitting.FitConfig] has two functions.

- `fit(task, sampler, fit_dir)` fits the model to the [`ForecastTask`][bdbv_forecasts.models.base.ForecastTask], with the candidate's `sampler` table from `ci/policy.toml` (`num_chains`, `num_warmup`, `num_samples`, `target_accept`, `max_tree_depth`, `seed`).
  It saves what the forecast needs (posterior draws) in `fit_dir`, appends progress lines to `fit_dir/progress.log`, and returns the `diagnostics.json` record (one row per sampled scalar with R-hat and bulk and tail ESS, and one row per chain with draws and divergences; see `ci/INTEGRATION.md`).
  It must also write `fit_dir/ppc.json` from the in-sample posterior predictive draws, with [`summarise_draws`][bdbv_forecasts.ppc.summarise_draws] and [`write_ppc`][bdbv_forecasts.ppc.write_ppc].
  `run_fit` writes the record, `fit.json` and the run time.
- `forecast(task, fit_dir)` returns a [`Forecast`][bdbv_forecasts.fitting.Forecast]: province samples numbered 1 to 100, each one joint posterior draw across weeks and provinces, and optionally an `unassigned` national component by week and sample.

`nowcast` then writes the hub file with [`to_hub_table`][bdbv_forecasts.hub.to_hub_table], which makes the national value the exact sum of the province samples plus the unassigned component, and writes `national.json` beside it with the modelled provinces and that component.
Set the repository variable `HUB_MODEL` to `renewal` once its metadata text is in `bdbv_forecasts.cli.MODEL_METADATA`.

## The fallback chain

The chain is set in `ci/policy.toml`.
The first candidate, in order, that clears every check is submitted.

1. `main`: the main model.
2. `retry-1`: the main model with a longer warm-up, a new seed and target acceptance 0.95.
3. `retry-2`: the main model with target acceptance 0.99 and tree depth 12.
4. `fallback`: the simplified variant (data-driven delays and missingness, no spatial coupling, weekly Rt). It is fitted alongside `main`, so a failed main model does not leave too little of the window.
5. `persistence`: the persistence baseline.

The retries run only if `main` fails.
A round is skipped only if even persistence fails its checks or hub validation.
Every candidate is filed under the same hub model id, so a fallback stands in for the main model.
The pull request body names the step that produced the submission and why the steps before it were ruled out.

## Checks and thresholds

A **fail** rules the candidate out and the chain moves on.
A **warn** lets it through and is stated in the pull request body.
Thresholds are in `ci/policy.toml`.

### Convergence

These are BVDOutbreakSize's thresholds (`docs/fits/convergence.jl`).
A fit that cannot be judged (no or unreadable `diagnostics.json`) fails.
Each threshold can be overridden by a repository variable `BDBV_CONVERGENCE_{FAIL,WARN}_{RHAT,ESS_BULK,ESS_TAIL,DIVERGENT}`.

| check | fail | warn |
| --- | --- | --- |
| max R-hat | > 1.1 | > 1.05 |
| min bulk and tail ESS | < 25 | < 100 |
| divergent fraction of draws | > 5% | > 1% |

### Posterior predictive checks

`fit` writes `ppc.json` beside `diagnostics.json`: one row per province, stream (known onsets, the hub's imputed onsets and totals) and in-sample onset week.
Each row has the observed count, the predictive median, the 50% and 90% intervals, the randomised PIT and whether the week is complete.
The gate reports, for each stream, the 50% and 90% interval coverage overall, by province and over the last 4 complete weeks.
It also summarises the PIT: the Kolmogorov-Smirnov distance from uniform, the share in the outer deciles (0.2 when calibrated; above 0.3 reads as intervals too narrow, below 0.1 too wide) and the mean PIT (away from 0.5 reads as bias).
The candidate's `ppc.png` artefact plots the predictive intervals against the observed counts, one panel per province.
Thresholds apply to the `total` stream and can be overridden by repository variables `BDBV_PPC_{COVERAGE_WARN,COVERAGE_FAIL,LEVEL,RECENT_WEEKS,PIT_KS_WARN,PIT_MIN_ROWS}`.

| check | fail | warn |
| --- | --- | --- |
| 90% interval coverage of the observed counts, overall or over the last 4 complete weeks | < 0.50 | < 0.70 |
| Kolmogorov-Smirnov distance of the PIT from uniform (20 or more rows) | | > 0.2 |
| sampled fit with no readable `ppc.json` | error | |

Persistence and any other model without a posterior report n/a, which neither passes nor blocks.
The coverage numbers go in the pull request's status table, with a collapsed table by province.

### Forecast and hub checks

| check | fail | warn |
| --- | --- | --- |
| schema (columns, `sample`, target, reference date, Saturdays, `tasks.json` options) | any | |
| values (negative, NaN, infinite, unparsable) | any | |
| samples (exactly 100, each a full trajectory) | any | |
| coverage (expected locations, 4 nowcast weeks and 3 horizon weeks) | any | |
| national: sample *k* of `CD` equals the sum of sample *k* of the provinces, plus any modelled unassigned component, in every week; every modelled province present | any | |
| bounds: 95th centile against max(largest observed week, 20) | > 20× | > 5× |
| growth of the median, week on week, over the forecast (offset 5) | > 4× | > 2× |
| last complete weeks against observed (offset 5) | off by > 5× | off by > 2× |
| forecast weeks against the persistence baseline | | off by > 3× |
| shared weeks against last week's submission | | off by > 2× |
| hub validation: `hub_validate` (Python) and `hubValidations` (R) | any error | |
| data coherence: the hub's national data against the sum of its provinces (observed, imputed, total) | | any gap |

The national check reads `national.json`, which `nowcast` writes beside the hub file.
It lists the modelled provinces and any unassigned national component by sample.
The data coherence check reads the output of `bdbv-forecasts coherence`.
A gap is reported with how the model handles it (see [Data](../data.md#national-against-the-provinces)).

Hub validation runs both validators from the hub validation bundle against the hub at the data commit, as the hub's own CI would.
The submission is staged as `<model_id>/<reference_date>-<model_id>.csv` with its metadata.
The checks it runs, and where the hub's README and config disagree, are in the [hub specification](../hub-spec.md).
`HUB_VALIDATIONS=false` skips only the R step.

## Dry-run mode

`DRY_RUN` is on unless the repository variable `DRY_RUN` is exactly `false`, or a manual run sets `dry_run: false`.
A dry run does everything except push and open the pull request.
It prints the pull request body to the log and the job summary, and uploads it in the `pr` artefact.
It needs no secrets: it looks up any existing pull request with the workflow's own read-only token.

To try it, run **Actions → Submit to the BDBV hub → Run workflow**, or:

```sh
gh workflow run submit.yml -R seabbs/bdbv-hub-baseline-forecasts -f dry_run=true -f force=true
```

Outside the window, add `-f ignore_window=true` and a `reference_date`.
`force` and `ignore_window` are allowed only in a dry run.

Locally, `task ci-dry` runs the same step scripts on fixture data with a stand-in command line, through every branch of the fallback chain.
`task ci-test` runs the tests of the CI scripts, and `task ci-fit` fits and forecasts one config as a candidate does.

## What a submission pull request looks like

The pull request goes from `seabbs-bot/BDBV-Modeling-Hub`, branch `<model_id>-<reference_date>`, into the hub's `main`.
It adds `model-output/<model_id>/<reference_date>-<model_id>.csv` and, the first time, `model-metadata/<model_id>.yml`.
The hub's organisers read the body, so it is short.
The visible part is one line on what was submitted, one plain status line (all checks passed, a refit, or the fallback used and why, with any stand-in), the plot, a link to the round's page in our docs and the bot footer.
The checks, fallback chain, convergence, predictive checks, forecast checks, data coherence, a national summary and the provenance sit in collapsed sections, each shown only when it has something to say.
The full reports stay in the job summary and the run's artefacts.

The round's page, `/nowcast/rounds/<reference_date>/`, is built by the Pages workflow from the `submission-ledger` branch.
The `record` job rebuilds the docs after a live round, so the page appears a few minutes after the pull request opens.
The [Nowcast](../nowcast.md) page links to the latest submitted round.

Below is the body for the GitHub dry run [run 40556037584](https://github.com/seabbs/bdbv-hub-baseline-forecasts/actions/runs/40556037584), rebuilt with the current layout, with the plot it would publish.

<div class="pr-body-example" markdown>

--8<-- "pr-body.md"

</div>

## Setup

What Sam needs to do before the first live run (from `ci/SETUP.md`).

- [ ] As `seabbs-bot`, fork the hub once: `gh repo fork InsightNet-US/BDBV-Modeling-Hub --clone=false`.
- [ ] In the fork's Settings → Actions → General, disable Actions.
- [ ] Create a **classic** personal access token on `seabbs-bot` with the scopes `public_repo` and `workflow`, with an expiry (for example 90 days), and put a renewal reminder in the calendar.
- [ ] In this repository, Settings → Environments → New environment `hub-submission`: deployment branches **Selected branches**, `main` only; no required reviewers; environment secret `HUB_SUBMIT_TOKEN` set to the token.
- [ ] Settings → Actions → General → Workflow permissions: **Read repository contents**. If a ruleset covers all branches, exclude `submission-ledger`.
- [ ] Optionally set repository variables: `HUB_MODEL` (default `persistence`, set `renewal` once the model and its metadata exist), `HUB_MODEL_ID`, `HUB_FORK`, `EXPECTED_LOCATIONS` (for example `CD,CD-IT,CD-NK,CD-HU,CD-TO`), `HUB_VALIDATIONS`, `FIT_RUNNER`, and the `BDBV_CONVERGENCE_*` and `BDBV_PPC_*` overrides.
- [ ] Make the last edit to the `cron` lines in `submit.yml` yourself (`/commit --as-me`), so GitHub sends scheduled-run failures to you, not the bot. Set Settings → Notifications → Actions to "Notify me on failed workflows only".
- [ ] Ask the hub organisers (Justin Lessler or Claire Smith) to approve the first run of the hub's validation workflow on the bot's pull request, and say weekly bot pull requests will follow.
- [ ] Run a dry run from the Actions tab and read the body in the `submit` job summary.
- [ ] Time a real fit with a dry run on a past vintage (`force`, `ignore_window`, `reference_date`) once the model is integrated.
- [ ] Inside a window, with you watching, run once with `dry_run: false` and check the pull request on the hub.
- [ ] Set the repository variable `DRY_RUN` to `false`. Set it back to `true`, or disable the workflow, to pause.

Scheduled workflows in a public repository stop after 60 days without activity, so commit to `main` now and then.
GitHub-hosted runners cap a job at 6 hours; the fit jobs time out at 350 minutes.
`ci/SETUP.md` covers runtimes, larger runners and where each part of the approach comes from (BVDOutbreakSize, cityforecasts and ecdc-weekly-growth-forecasts).
