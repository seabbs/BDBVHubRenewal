# Plugging the submission CI into the main repo

The bundle drops into `seabbs/BDBVHubRenewal` as `ci/` plus `.github/`.
`ci/` is its own small uv project with only the standard library.
The workflows call the main repo only through the `bdbv-hub-renewal` CLI, whose command is set by `FORECAST_CLI` (default `uv run bdbv-hub-renewal`), and through `scripts/validate-hub.R`.
`ci/fixtures/fake_bdbv_hub_renewal.py` implements everything below on fixture data.
It is the reference for the interface, and `ci/run_dry.sh` runs the whole workflow against it.

## What exists and what is needed

| command | status in the main repo | used by |
| --- | --- | --- |
| `bdbv-hub-renewal data --ref <hub sha>` | exists. Reads `GITHUB_TOKEN`. | `prepare` |
| `bdbv-hub-renewal nowcast --model persistence --vintage <as_of> --out <dir>` | exists | `prepare` (persistence candidate and baseline) |
| `bdbv-hub-renewal metadata --model <model> --out <dir>` | exists | `gate` |
| `scripts/validate-hub.R <hub> <csv> <yml>`, with `CHECK_WINDOW=true` | exists | `gate`, through `ci/steps/validate_hub.sh` |
| `bdbv-hub-renewal fit ...` | done (`bdbv_hub_renewal.submission.fitting`; `main` and `fallback` run the renewal model, `bdbv_hub_renewal.forecasters.renewal`) | `fit`, `retry` |
| `nowcast --config <c> --fit <dir>` | done | `fit`, `retry` |
| `bdbv-hub-renewal plot-submission ...` | done, `plots.plot_submission` | every candidate |
| `bdbv-hub-renewal plot-ppc --ppc <ppc.json> --out <png>` | done, a thin wrapper on `ppc.plot_ppc` | candidates whose `ppc.json` is available |
| `bdbv-hub-renewal --version` | done | `run.json`, the PR body |
| `bdbv-hub-renewal coherence --vintage <as_of> --out <json>` | done | `prepare` (data coherence check) |

### `bdbv-hub-renewal fit`

```sh
bdbv-hub-renewal fit --model renewal --config main|fallback --vintage <as_of> \
  --sampler-json '{"num_chains": 4, "num_warmup": 1000, "num_samples": 1000,
                   "target_accept": 0.8, "max_tree_depth": 10, "seed": 1}' \
  --out <fit dir>
```

- `--model` is the model chosen in `results/model-choice.yml` (`bdbv-hub-renewal chosen-model`).
  The hub id is set apart in `config/hub.toml` (`bdbv-hub-renewal hub-id`), `epiforecasts-seabbsrenewal`.
  `epiforecasts-renewal` is already a different hub model.
- `--config` selects the model configuration.
  `main` is the full model.
  `fallback` is the same code with the delays and missingness plugged in from data, and no spatial coupling.
  Each matrix leg passes its own config.
- `--sampler-json` is the candidate's `sampler` table from `ci/policy.toml`.
  The retries differ only in it.
- `--vintage` is a real `as_of`, or `latest`.
- Write `<fit dir>/diagnostics.json` (below) whether or not the fit converged.
  Exit non-zero only if the fit could not run.
- Write `<fit dir>/ppc.json` (below) beside it, from the posterior predictive draws of the in-sample data.
- Append progress lines to `<fit dir>/progress.log` as you go, for example `iteration=500 divergences=0` every few hundred iterations.
  The job prints the newest line once a minute, as BVDOutbreakSize does, so a stalled fit is visible.
- Everything the forecast needs from the fit stays in `<fit dir>`, for example posterior draws as NetCDF or `.npz`.
  That directory is what is cached under the content-hash key, and uploaded for 3 days.
- Set the device count before importing JAX: `numpyro.set_host_device_count(4)`.
  The setup action also sets `XLA_FLAGS` to the runner's CPU count, and sets `JAX_COMPILATION_CACHE_DIR`.

### `nowcast` with a fit

```sh
bdbv-hub-renewal nowcast --model renewal --config main --fit <fit dir> \
  --vintage <as_of> --out <candidate dir>/submission
```

- Write one hub file anywhere under `--out`.
  The existing layout, `model-output/<model_id>/<reference_date>-<model_id>.csv`, is fine.
  The PR always files it as `model-output/$HUB_MODEL_ID/<reference_date>-$HUB_MODEL_ID.csv`, so persistence can stand in for the main model.
- Include the national row `CD`, as the sum of the province trajectories by sample index, plus any national cases the model does not assign to a province.
  Use the same sample ids across locations.
  National sample k must equal the sum of province sample k plus that sample's unassigned cases, for every week.
  `to_hub_table` writes it that way, and the gate rebuilds the national rows the same way if they differ, and notes it.
- Write `national.json` at the root of `--out` (`<candidate>/submission/national.json`), which `candidate.sh` passes to `check-forecast --national-sidecar`:

  ```json
  {"schema_version": 1, "national": "CD", "modelled_provinces": ["CD-IT", "CD-NK"],
   "unassigned": [{"target_end_date": "2026-10-31", "sample": 1, "value": 0.0}]}
  ```

  A modelled province missing from the hub file warns.
  `unassigned` holds the national cases not assigned to a province, by week and sample; leave it empty when there are none.
  Without the file the gate takes the unassigned part from the gap between the national data and the provinces (`coherence.json`), and says so.
- Write `handled.json` at the root of `--out`: a JSON list of short notes on anything handled on the way, such as a data stream taken from an earlier vintage.
  They go on the PR's `Handled:` line.
- Submit at least 7 back weeks (to `reference_date − 6 weeks`) as well as the 3 forecast weeks.
  The `observed` check compares the median of the last two complete weeks (4 or more weeks before `as_of`) with the data.
  With fewer back weeks it has nothing to compare, and skips.
  `coverage` requires `nowcast_weeks = 4` back weeks and `horizon_weeks = 3`.
- Keep running the existing Python checks.
  A non-zero exit is only a warning here, because the gate's own checks decide.

### `plot-submission`

```sh
bdbv-hub-renewal plot-submission --submission <csv> --vintage <as_of> --out <png>
```

Draw what was **submitted**, the samples in `<csv>`, not a re-run of the model.
`plots.plot_nowcast` already takes them:

```python
def plot_submission(submission: Path, vintage: str) -> Figure:
    samples = read_hub_file(submission)  # location, target_end_date, sample, value
    locations = ["CD", *largest_provinces(samples, n=3)]
    return grid_of(
        plot_nowcast(location=loc, vintage=vintage, forecasts=samples)
        for loc in locations
    )
```

Use one PNG at about 1600×1000 px, light mode, under 1 MB.
It renders inline in the PR body from a raw URL pinned to a commit.

### `coherence`

```sh
bdbv-hub-renewal coherence --vintage <as_of> --out <inputs>/coherence.json
```

Compare the national count with the sum of the provinces in the data vintage, for each stream and week:

```json
{"schema_version": 1, "as_of": "2026-10-31", "national": "CD", "coherent": false,
 "streams": {"observed": {"weeks_with_gap": 1, "max_abs_gap": 2, "total_gap": 2},
             "imputed": {"weeks_with_gap": 0, "max_abs_gap": 0, "total_gap": 0},
             "total": {"weeks_with_gap": 1, "max_abs_gap": 2, "total_gap": 2}},
 "rows": [{"target_end_date": "2026-10-24", "stream": "total",
           "national": 50, "provinces": 48, "gap": 2}],
 "handling": "<one or two sentences on how the model handles the gap>"}
```

- `stream` is `observed` (known onsets), `imputed` (the hub's imputed onsets) or `total`.
- `rows` lists only weeks with a non-zero gap, where `gap` is national minus the sum of the provinces.
- `prepare.sh` runs it once per round and then `check-coherence`, which writes `coherence-summary.{md,json}` to `inputs/`.
  The check warns when any week has a gap (or the file is unreadable), passes otherwise, and is `n/a` when the command fails or writes nothing.
  It never blocks a candidate.
  The PR body reports it once, as a status line and a collapsed table of the gaps with the `handling` text.

### `plot-ppc`

```sh
bdbv-hub-renewal plot-ppc --ppc <candidate>/ppc.json --out <candidate>/ppc.png
```

`candidate.sh` calls it only when `ppc.json` has `available: true`.
It is `plot_ppc(read_ppc(path))` from `bdbv_hub_renewal.evaluation.ppc`, saved with `fig.savefig(out, dpi=...)`: small multiples by province of the predictive median and intervals against the observed counts, light mode, under 1 MB.
The PNG travels in the `candidate-<id>` artefact; the PR body points to it.

### `--version`

Print the package version, for example `0.2.0`.
The PR body adds the commit SHA itself.

## `diagnostics.json`

```json
{
  "schema_version": 1,
  "model": "renewal",
  "config": "main",
  "sampler": "nuts",
  "num_chains": 4,
  "num_samples": 1000,
  "max_tree_depth": 10,
  "runtime_seconds": 812.4,
  "parameters": [
    {"name": "log_rt_dev", "index": 3, "rhat": 1.004,
     "ess_bulk": 912.0, "ess_tail": 1033.0}
  ],
  "chains": [
    {"chain": 0, "draws": 1000, "divergences": 0, "step_size": 0.041,
     "tree_depth_hits": 0, "mean_accept": 0.86, "ebfmi": 0.91}
  ]
}
```

- `parameters` has one row per scalar element of every sampled parameter.
  `index` is 0 for a scalar.
  Use `null` for an R-hat or ESS that is undefined, such as a constant.
  The gate ignores those, as BVDOutbreakSize does.
  Leave out the latent trajectories and other deterministic sites, as BVDOutbreakSize's `_DIAGNOSTIC_EXCLUDE` does.
- `chains` has one row per chain.
  `divergences` and `draws` give the divergent fraction.
  The renewal model also reports `tree_depth_share`, the share of draws at the depth cap.
  The other fields are reported, not gated.
- `worst_rhat` lists the ten parameter elements with the highest R-hat, as BVDOutbreakSize does.
- A model with no sampler, such as persistence, writes `{"schema_version": 1, "sampler": "none"}`.
- A config that runs a stand-in model, such as a config running persistence in place of a model not yet built, adds `"stand_in": "<text>"`.
  It can also go in `<fit dir>/fit.json`.
  `candidate.sh` copies it into `run.json`.
  The chain table, the job summary and the PR body then mark the candidate "(stand-in)", and the PR body states the text in a bold **Stand-in:** line under "Produced by".
- A missing or unreadable file is an `error`, and gates as a failure.

With NumPyro and ArviZ:

```python
import arviz as az
import numpy as np

idata = az.from_numpyro(mcmc)
summ = az.summary(idata, var_names=sampled_sites, kind="diagnostics")
extra = mcmc.get_extra_fields(group_by_chain=True)  # needs extra_fields=(
#   "diverging", "num_steps", "adapt_state.step_size", "accept_prob", "energy")
depth = np.log2(extra["num_steps"] + 1)
diagnostics = {
    "schema_version": 1, "model": model, "config": config, "sampler": "nuts",
    "num_chains": n_chains, "num_samples": n_samples,
    "max_tree_depth": max_tree_depth, "runtime_seconds": runtime,
    "parameters": [
        {"name": name.split("[")[0], "index": index_of(name),
         "rhat": none_if_nan(row.r_hat), "ess_bulk": none_if_nan(row.ess_bulk),
         "ess_tail": none_if_nan(row.ess_tail)}
        for name, row in summ.iterrows()
    ],
    "chains": [
        {"chain": c, "draws": n_samples,
         "divergences": int(extra["diverging"][c].sum()),
         "step_size": float(extra["adapt_state.step_size"][c].mean()),
         "tree_depth_hits": int((depth[c] >= max_tree_depth).sum()),
         "mean_accept": float(extra["accept_prob"][c].mean()),
         "ebfmi": float(az.bfmi(extra["energy"][c][None, :])[0])}
        for c in range(n_chains)
    ],
}
```

## `ppc.json`

```json
{"schema_version": 1, "available": true, "model": "renewal", "config": "main",
 "intervals": [0.5, 0.9],
 "rows": [{"location": "CD-IT", "stream": "total", "target_end_date": "2026-10-03",
           "observed": 12, "median": 10.0, "lower_50": 7.0, "upper_50": 13.0,
           "lower_90": 4.0, "upper_90": 19.0, "pit": 0.43, "complete": true}]}
```

- One row per province, stream and in-sample onset week.
  `stream` is `observed` (known onsets), `imputed` (the hub's imputed onsets) or `total`.
- `median`, `lower_<pct>` and `upper_<pct>` summarise the posterior predictive draws of the count as observed in the data vintage, with one central interval per level in `intervals`.
- `pit` is the **randomised** PIT for counts, `F(y − 1) + v (F(y) − F(y − 1))` with `v ~ Uniform(0, 1)` from a seeded generator and `F` the empirical predictive distribution function.
  It is uniform on (0, 1) under a calibrated model even though counts are discrete.
- `complete` is true for weeks ending at least the completeness lag (4 weeks) before the data vintage.
- `bdbv_hub_renewal.evaluation.ppc.summarise_draws(observed, draws, as_of=..., model=..., config=...)` builds all of this from a table of observations (`location`, `stream`, `target_end_date`, `observed`) and an array of draws of shape `(n_draws, n_rows)`; `write_ppc` writes it.
- A model with no posterior writes `{"schema_version": 1, "available": false, "reason": "<text>"}` (`ppc.not_available(reason)`), which the gate reports as `n/a`.
- A sampled fit (`sampler` other than `none` in `diagnostics.json`) with a missing or unreadable `ppc.json` is an `error`, and gates as a failure, as a missing diagnostics file does.
  A fit with no sampler and no `ppc.json` is `n/a`.

`check-ppc` (run by `candidate.sh` after `summarise-fit`) writes `ppc-summary.{md,json}` to the candidate directory and the job summary.
It reports the 50% and 90% coverage of each stream overall, by province and over the last 4 complete weeks, and a PIT summary: the Kolmogorov-Smirnov distance from uniform, the share in the outer deciles (0.2 when calibrated; above 0.3 reads as intervals too narrow, below 0.1 too wide) and the mean PIT (away from 0.5 reads as bias).
The PR body adds the coverage to its status table and the per-province table in a collapsed section.

## Output layout

Everything goes under `$WORK` (`output/ci`):

```
output/ci/
  inputs/            time-series.csv, tasks.json, previous.csv, context.json,
                     coherence.json, coherence-summary.{md,json}
  fits/<id>/         diagnostics.json, ppc.json, progress.log, draws (cached)
  candidates/<id>/   submission/…csv, submission/national.json, nowcast.png,
                     run.json, fit-summary.{md,json}, ppc.json, ppc.png,
                     ppc-summary.{md,json}, coherence-summary.json,
                     checks.{md,json}, hub-validation.log
  metadata/          <model_id>.yml
  gate/              provenance.{md,json}, selected/ (copy of the chosen
                     candidate), pr-body.md, pr-title.txt, pr-outcome.txt
```

Add `output/` to `.gitignore` if it is not already there.

## Fit cache key

`ci/policy.toml` → `[cache] inputs` lists what a fit depends on: `src`, `config`, `pyproject.toml` and `uv.lock`.
The key hashes those files with the hub commit and the candidate's config and sampler settings.
If the model reads another path, add it there.
That list is the only place it is defined, which is the lesson from BVDOutbreakSize's cache-key drift.

## Taskfile

Suggested additions:

```yaml
  ci-test:
    desc: Test the submission CI scripts
    dir: ci
    cmds:
      - uv run ruff check src tests fixtures
      - uv run mypy
      - uv run pytest -q
  ci-dry:
    desc: Dry run of the submission workflow on fixture data
    cmds:
      - ci/run_dry.sh {{.CLI_ARGS}}
```

## Checks and thresholds

Thresholds are in `ci/policy.toml`, and the policy is described on the docs' Submissions page.
In short, problems are handled rather than failed on:

- **fail** (rules a candidate out): a file the hub would reject (schema, values, samples, hubValidations) or a clearly broken fit (R-hat above 1.5, ESS below 10, divergent draws above 20%), or no diagnostics from a sampled fit;
- **refit once**: the main model past BVDOutbreakSize's fail thresholds (R-hat 1.1, ESS 25, divergent draws 5%), with the refit preferred unless it is clearly broken;
- **handled** (on the PR's `Handled:` line): the national rows rebuilt as the sum of the provinces plus the unassigned component; a gap between the national data and the provinces; a data stream taken from an earlier vintage (`handled.json` beside the hub file);
- **warn** (reported only): posterior predictive checks, plausibility checks, missing locations or weeks, and BVDOutbreakSize's warn thresholds.
