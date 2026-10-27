# Submission workflow

The weekly path from new hub data to a submission.
{ .page-subtitle }

!!! warning "Dry run"
    The workflow runs in dry-run mode until Sam switches it on (see [Setup](setup.md)).
    It builds everything and prints the pull request it would open, but pushes nothing.
    The `main` and `fallback` fit configs run the PyRenew renewal model, and submissions go in as `epiforecasts-hubrenewal`.
    Its model metadata is a draft until Sam has reviewed the held-out results.

## Pipeline

The workflow is `.github/workflows/submit.yml`.
It runs on a schedule while the hub's submission window is open, and on demand.
The window is Sunday 00:00 to Tuesday 23:59:59 Kinshasa time (UTC+1), so the schedule runs from Saturday 23:00 UTC, every three hours on Sunday and Monday, and on Tuesday until 14:00 UTC.
Nobody approves the pull request in CI.
The gate's checks choose the candidate, and the pull request body says what was submitted and why.

| job | what it does |
| --- | --- |
| `detect` | Compares the hub's newest data commit and `as_of` with the ledger. Stops unless there is new data for the open round. |
| `prepare` | Pins the hub data at one commit (`bdbv-hub-renewal data --ref <sha>`), checks national coherence and builds the persistence candidate, which is also the baseline for the checks. |
| `fit` | One matrix job per primary candidate (`main` and `fallback`), in parallel. Each runs `bdbv-hub-renewal fit`, then `nowcast --fit`, the checks and the plots. Fits are cached under a hash of the model source, the hub commit and the sampler settings. |
| `triage` | Says whether `main` needs its one refit. |
| `retry` | Only if `main` needs it: one refit with a longer warm-up, more chains and jittered inits. |
| `gate` | Walks the fallback chain and submits the first candidate that clears convergence, the posterior predictive checks, the forecast checks and hub validation. |
| `submit` | Publishes the plot, builds the pull request body, and opens or updates the pull request from `seabbs-bot`'s fork. A dry run prints the body instead. |
| `record` | Live runs only. Writes the outcome to `state.json` on the `submission-ledger` branch and archives the round's files there. |
| `outcome` | Fails the run if the round was skipped or a job broke, so GitHub sends a failure notice. |

The workflow calls this repository only through the `bdbv-hub-renewal` command line.

| command | used by |
| --- | --- |
| `bdbv-hub-renewal data --ref <sha>` | `prepare` |
| `bdbv-hub-renewal coherence --vintage <as_of> --out <json>` | `prepare` |
| `bdbv-hub-renewal fit --model <m> --config <c> --vintage <as_of> --sampler-json <json> --out <dir>` | `fit`, `retry` |
| `bdbv-hub-renewal nowcast --model <m> --config <c> --fit <dir> --vintage <as_of> --out <dir>` | `fit`, `retry`, `prepare` (persistence, without `--config`) |
| `bdbv-hub-renewal plot-submission --submission <csv> --vintage <as_of> --out <png>` | every candidate |
| `bdbv-hub-renewal plot-ppc --ppc <json> --out <png>` | candidates with a posterior |
| `bdbv-hub-renewal metadata --model <m> --out <dir>` | `gate` |
| `bdbv-hub-renewal validate` and `scripts/validate-hub.R` | `gate`, through `ci/steps/validate_hub.sh` |
| `bdbv-hub-renewal --version` | the pull request body |

`fit` writes the fit directory: `fit.json`, `diagnostics.json`, `ppc.json`, `progress.log` and whatever the forecast needs.
A model without a sampler, such as persistence, writes diagnostics with `"sampler": "none"` and `"status": "not-applicable"`, and a `ppc.json` with `"available": false`.
The gate reads both as not applicable.
The formats are specified in `ci/INTEGRATION.md`.

A fit config is registered in [`bdbv_hub_renewal.fitting`][bdbv_hub_renewal.fitting].

### Model interface

The model integration plugs PyRenew in by registering `main` and `fallback` with `register_config(..., replace=True)`.
Each [`FitConfig`][bdbv_hub_renewal.fitting.FitConfig] has two functions.

- `fit(task, sampler, fit_dir)` fits the model to the [`ForecastTask`][bdbv_hub_renewal.models.base.ForecastTask], with the candidate's `sampler` table from `ci/policy.toml` (`num_chains`, `num_warmup`, `num_samples`, `target_accept`, `max_tree_depth`, `seed`).
  It saves what the forecast needs (posterior draws) in `fit_dir`, appends progress lines to `fit_dir/progress.log`, and returns the `diagnostics.json` record (one row per sampled scalar with R-hat and bulk and tail ESS, and one row per chain with draws and divergences, as in `ci/INTEGRATION.md`).
  It must also write `fit_dir/ppc.json` from the in-sample posterior predictive draws, with [`summarise_draws`][bdbv_hub_renewal.ppc.summarise_draws] and [`write_ppc`][bdbv_hub_renewal.ppc.write_ppc].
  `run_fit` writes the record, `fit.json` and the run time.
- `forecast(task, fit_dir)` returns a [`Forecast`][bdbv_hub_renewal.fitting.Forecast]: province samples numbered 1 to 100, each one joint posterior draw across weeks and provinces, and optionally an `unassigned` national component by week and sample.

`nowcast` then writes the hub file with [`to_hub_table`][bdbv_hub_renewal.hub.to_hub_table], which makes the national value the exact sum of the province samples plus the unassigned component, and writes `national.json` beside it with the modelled provinces and that component.
Set the repository variable `HUB_MODEL` to `hubrenewal`, which submits as `epiforecasts-hubrenewal`, once its metadata text is in `bdbv_hub_renewal.cli.MODEL_METADATA`.

## The fallback chain

The workflow handles problems rather than failing on them, so the chain rarely moves past the main model.
A candidate is ruled out only when its fit is clearly broken, or its file would be invalid on the hub.
The chain is set in `ci/policy.toml`.

1. `main`: the main model.
2. `retry-1`: one refit of the main model with a longer warm-up, 8 chains and jittered inits (`init = "jitter"` in its sampler table). It runs only when `main` is past the retry thresholds below, or was ruled out. The refit is submitted unless it is clearly broken. Otherwise the original fit is kept, if it is not broken.
3. `fallback`: the simplified variant (data-driven delays and missingness, no spatial coupling, weekly Rt). It is fitted alongside `main`, so a broken main model does not leave too little of the window.
4. `persistence`: the persistence baseline.

There is at most one refit, and fits are cached under a content hash, so a re-run in the same window reuses them.
A round is skipped only if even persistence fails hub validation.
Every candidate is filed under the same hub model id, so a fallback stands in for the main model.
The pull request body leads with what was handled and how, on one `Handled:` line, and names the fallback and why if one was used.

## Checks and thresholds

Only an invalid submission or a clearly broken fit **fails**, which rules the candidate out.
Everything else is **handled** (and stated on the `Handled:` line) or **warns** (and is named on the status line).
Thresholds are in `ci/policy.toml`.

### Convergence

BVDOutbreakSize's thresholds (`docs/fits/convergence.jl`) are the retry and warn levels.
Past the retry level the main model is refitted once, and the refit is accepted unless it is clearly broken.
A fit with no or unreadable `diagnostics.json` cannot be judged and fails.
Each threshold can be overridden by a repository variable `BDBV_CONVERGENCE_{FAIL,RETRY,WARN}_{RHAT,ESS_BULK,ESS_TAIL,DIVERGENT}`.

| check | clearly broken (fail) | refit once | warn |
| --- | --- | --- | --- |
| max R-hat | > 1.5 | > 1.1 | > 1.05 |
| min bulk and tail ESS | < 10 | < 25 | < 100 |
| divergent fraction of draws | > 20% | > 5% | > 1% |

### Posterior predictive checks

These are reported only.
They never fail a candidate or trigger a fallback, because model choice comes from the backtest, not from one run's checks.
`fit` writes `ppc.json` beside `diagnostics.json`: one row per province, stream (known onsets, the hub's imputed onsets and totals) and in-sample onset week.
Each row has the observed count, the predictive median, the 50% and 90% intervals, the randomised PIT and whether the week is complete.
The gate reports, for each stream, the 50% and 90% interval coverage overall, by province and over the last 4 complete weeks.
It also summarises the PIT: the Kolmogorov-Smirnov distance from uniform, the share in the outer deciles (0.2 when calibrated, above 0.3 for intervals too narrow, below 0.1 for too wide) and the mean PIT (away from 0.5 reads as bias).
The candidate's `ppc.png` artefact plots the predictive intervals against the observed counts, one panel per province.
Thresholds apply to the `total` stream and can be overridden by repository variables `BDBV_PPC_{COVERAGE_WARN,COVERAGE_FAIL,LEVEL,RECENT_WEEKS,PIT_KS_WARN,PIT_MIN_ROWS}`.

| check | warn |
| --- | --- |
| 90% interval coverage of the observed counts, overall or over the last 4 complete weeks | < 0.70 (called poor below 0.50) |
| Kolmogorov-Smirnov distance of the PIT from uniform (20 or more rows) | > 0.2 |
| sampled fit with no readable `ppc.json` | always |

Persistence and any other model without a posterior report n/a.
The coverage numbers go in the pull request's collapsed checks, with a table by province.

### Forecast and hub checks

| check | fail | handled or warn |
| --- | --- | --- |
| schema (columns, `sample`, target, reference date, Saturdays, `tasks.json` options) | any | |
| values (negative, NaN, infinite, unparsable) | any | |
| samples (exactly 100, each a full trajectory) | any | |
| hub validation: `hub_validate` (Python) and `hubValidations` (R) | any error | |
| national: sample *k* of `CD` equals the sum of sample *k* of the provinces plus the unassigned component, in every week | | handled: rebuilt by construction |
| a modelled province missing from the file | | warn |
| coverage (expected locations, 4 nowcast weeks and 3 horizon weeks) | | warn |
| bounds: 95th centile against max(largest observed week, 20) | | > 5× |
| growth of the median, week on week, over the forecast (offset 5) | | > 2× |
| last complete weeks against observed (offset 5) | | off by > 2× |
| forecast weeks against the persistence baseline | | off by > 3× |
| shared weeks against last week's submission | | off by > 2× |
| data coherence: the hub's national data against the sum of its provinces (observed, imputed, total) | | handled: noted, modelled as an unassigned component |
| a data stream missing or late for the vintage (the delays) | | handled: the latest available vintage of that stream is used, and noted |

The plausibility checks keep a fail level in `policy.toml` (`bound_fail`, `growth_fail`, `observed_fail`), off by default.

The national rows are correct by construction.
`nowcast` writes them as the sum of the provinces plus the model's unassigned component, and writes `national.json` beside the hub file with the modelled provinces and that component.
Before the checks, the gate rebuilds the national rows the same way if they differ, using `national.json` or, without it, the gap between the hub's national data and its provinces for each week.
The data coherence check reads the output of `bdbv-hub-renewal coherence`.
A gap is reported with how the model handles it (see [Data](../data.md#national-coherence)).

Hub validation runs both validators from the hub validation bundle against the hub at the data commit, as the hub's own CI would.
The submission is staged as `<model_id>/<reference_date>-<model_id>.csv` with its metadata.
The checks it runs, and where the hub's README and config disagree, are in the [hub specification](hub-spec.md).
`HUB_VALIDATIONS=false` skips only the R step.

## Dry-run mode

`DRY_RUN` is on unless the repository variable `DRY_RUN` is exactly `false`, or a manual run sets `dry_run: false`.
A dry run does everything except push and open the pull request.
It prints the pull request body to the log and the job summary, and uploads it in the `pr` artefact.
It needs no secrets: it looks up any existing pull request with the workflow's own read-only token.

To try it, run **Actions → Submit to the BDBV hub → Run workflow**, or:

```sh
gh workflow run submit.yml -R seabbs/BDBVHubRenewal -f dry_run=true -f force=true
```

Outside the window, add `-f ignore_window=true` and a `reference_date`.
`force` and `ignore_window` are allowed only in a dry run.

Locally, `task ci-dry` runs the same step scripts on fixture data with a stand-in command line, through every branch of the fallback chain.
`task ci-test` runs the tests of the CI scripts, and `task ci-fit` fits and forecasts one config as a candidate does.
`task submission-preflight` reads the repository's settings and lists what a live run still needs (see [Setup](setup.md)).

## Pull request

The pull request goes from `seabbs-bot/BDBV-Modeling-Hub`, branch `<model_id>-<reference_date>`, into the hub's `main`.
It adds `model-output/<model_id>/<reference_date>-<model_id>.csv` and, the first time, `model-metadata/<model_id>.yml`.
The hub's organisers read the body, so it is short.
The visible part is one line on what was submitted, one plain status line (all checks passed, a refit, or the fallback used and why, with any stand-in), the plot, a link to the round's page in our docs and the bot footer.
The checks, fallback chain, convergence, predictive checks, forecast checks, data coherence, a national summary and the provenance sit in collapsed sections, each shown only when it has something to say.
The full reports stay in the job summary and the run's artefacts.

The round's page, `/submission/rounds/<reference_date>/`, is built by the Pages workflow from the `submission-ledger` branch.
The `record` job rebuilds the docs after a live round, so the page appears a few minutes after the pull request opens.
The [Nowcast](../nowcast.md) page links to the latest submitted round, and [Rounds](rounds/index.md) lists them all.

Below is the body for the GitHub dry run [run 40556037584](https://github.com/seabbs/BDBVHubRenewal/actions/runs/40556037584), rebuilt with the current layout, with the plot it would publish.

<div class="pr-body-example" markdown>

--8<-- "pr-body.md"

</div>
