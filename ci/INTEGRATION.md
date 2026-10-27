# Plugging the submission CI into the main repo

The bundle drops into `seabbs/bdbv-hub-baseline-forecasts` as `ci/` plus `.github/`.
`ci/` is its own small uv project with only the standard library.
The workflows call the main repo only through the `bdbv-forecasts` CLI, whose command is set by `FORECAST_CLI` (default `uv run bdbv-forecasts`), and through `scripts/validate-hub.R`.
`ci/fixtures/fake_bdbv_forecasts.py` implements everything below on fixture data.
It is the reference for the interface, and `ci/run_dry.sh` runs the whole workflow against it.

## What exists and what is needed

| command | status in the main repo | used by |
| --- | --- | --- |
| `bdbv-forecasts data --ref <hub sha>` | exists. Reads `GITHUB_TOKEN`. | `prepare` |
| `bdbv-forecasts nowcast --model persistence --vintage <as_of> --out <dir>` | exists | `prepare` (persistence candidate and baseline) |
| `bdbv-forecasts metadata --model <model> --out <dir>` | exists | `gate` |
| `scripts/validate-hub.R <hub> <csv> <yml>`, with `CHECK_WINDOW=true` | exists | `gate`, through `ci/steps/validate_hub.sh` |
| `bdbv-forecasts fit ...` | **new** | `fit`, `retry` |
| `nowcast --config <c> --fit <dir>` | **new options** | `fit`, `retry` |
| `bdbv-forecasts plot-submission ...` | **new**, a thin wrapper on `plots.plot_nowcast` | every candidate |
| `bdbv-forecasts --version` | **new** | `run.json`, the PR body |

### `bdbv-forecasts fit`

```sh
bdbv-forecasts fit --model renewal --config main|fallback --vintage <as_of> \
  --sampler-json '{"num_chains": 4, "num_warmup": 1000, "num_samples": 1000,
                   "target_accept": 0.8, "max_tree_depth": 10, "seed": 1}' \
  --out <fit dir>
```

- `--config` selects the model configuration.
  `main` is the full model.
  `fallback` is the same code with the delays and missingness plugged in from data, no spatial coupling, and a weekly Rt random walk.
  Each matrix leg passes its own config.
- `--sampler-json` is the candidate's `sampler` table from `ci/policy.toml`. The retries differ only in it.
- `--vintage` is a real `as_of`, or `latest`.
- Write `<fit dir>/diagnostics.json` (below) whether or not the fit converged.
  Exit non-zero only if the fit could not run.
- Append progress lines to `<fit dir>/progress.log` as you go, for example `iteration=500 divergences=0` every few hundred iterations.
  The job prints the newest line once a minute, as BVDOutbreakSize does, so a stalled fit is visible.
- Everything the forecast needs from the fit stays in `<fit dir>`, for example posterior draws as NetCDF or `.npz`.
  That directory is what is cached under the content-hash key, and uploaded for 3 days.
- Set the device count before importing JAX: `numpyro.set_host_device_count(4)`.
  The setup action also sets `XLA_FLAGS` to the runner's CPU count, and sets `JAX_COMPILATION_CACHE_DIR`.

### `nowcast` with a fit

```sh
bdbv-forecasts nowcast --model renewal --config main --fit <fit dir> \
  --vintage <as_of> --out <candidate dir>/submission
```

- Write one hub file anywhere under `--out`.
  The existing layout, `model-output/<model_id>/<reference_date>-<model_id>.csv`, is fine.
  The PR always files it as `model-output/$HUB_MODEL_ID/<reference_date>-$HUB_MODEL_ID.csv`, so persistence can stand in for the main model.
- Include the national row `CD`, as the sum of the province trajectories by sample index, plus any national cases the model does not assign to a province.
  Use the same sample ids across locations.
  The `national_sum` check fails unless national sample k equals the sum of province sample k plus that sample's unassigned cases exactly, for every week (only float noise, 1e-9, is allowed).
- Write `national.json` at the root of `--out` (`<candidate>/submission/national.json`), which `candidate.sh` passes to `check-forecast --national-sidecar`:

  ```json
  {"schema_version": 1, "national": "CD", "modelled_provinces": ["CD-IT", "CD-NK"],
   "unassigned": [{"target_end_date": "2026-10-31", "sample": 1, "value": 0.0}]}
  ```

  `modelled_provinces` must all be in the hub file, or the check fails.
  `unassigned` holds the national cases not assigned to a province, by week and sample; leave it empty when there are none.
  Without the file the check uses the provinces in the hub file and no unassigned component, and says so in its summary.
- Submit at least 7 back weeks (to `reference_date − 6 weeks`) as well as the 3 forecast weeks.
  The `observed` check compares the median of the last two complete weeks (4 or more weeks before `as_of`) with the data.
  With fewer back weeks it has nothing to compare, and skips.
  `coverage` requires `nowcast_weeks = 4` back weeks and `horizon_weeks = 3`.
- Keep running the existing Python checks.
  A non-zero exit is only a warning here, because the gate's own checks decide.

### `plot-submission`

```sh
bdbv-forecasts plot-submission --submission <csv> --vintage <as_of> --out <png>
```

Draw what was **submitted**: the samples in `<csv>`, not a re-run of the model.
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
  The other fields are reported, not gated.
- A model with no sampler, such as persistence, writes `{"schema_version": 1, "sampler": "none"}`.
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

## Output layout

Everything goes under `$WORK` (`output/ci`):

```
output/ci/
  inputs/            time-series.csv, tasks.json, previous.csv, context.json
  fits/<id>/         diagnostics.json, progress.log, draws (cached)
  candidates/<id>/   submission/…csv, nowcast.png, run.json,
                     fit-summary.{md,json}, checks.{md,json}, hub-validation.log
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

Thresholds are in `ci/policy.toml`.
A **fail** rules the candidate out and the chain moves on.
A **warn** lets it through and is stated in the PR body.

| check | fail | warn |
| --- | --- | --- |
| convergence, max R-hat | > 1.1 | > 1.05 |
| convergence, min bulk and tail ESS | < 25 | < 100 |
| convergence, divergent fraction of draws | > 5% | > 1% |
| schema (columns, `sample`, target, reference date, Saturdays, `tasks.json` options) | any | |
| values (negative, NaN, infinite, unparsable) | any | |
| samples (exactly 100, each a full trajectory) | any | |
| coverage (expected locations, nowcast and horizon weeks) | any | |
| national equals the sum of provinces plus unassigned cases, per sample and week, exactly; every modelled province present | any | |
| bounds: 95th centile against max(largest observed week, 20) | > 20× | > 5× |
| growth of the median, week on week, over the forecast (offset 5) | > 4× | > 2× |
| last complete weeks against observed (offset 5) | off by > 5× | off by > 2× |
| forecast weeks against the persistence baseline | | off by > 3× |
| shared weeks against last week's submission | | off by > 2× |
| hubValidations (`validate_submission` and `validate_model_metadata`) | any error | |
