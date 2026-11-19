# Using the fine-tuned Chronos-2

How to nowcast your own reporting triangle with the Chronos-2 checkpoints we fine-tuned on simulated outbreaks.
{ .page-subtitle }

Each fine-tuned checkpoint is a GitHub release of this repository, `chronos-ft-<hash>`, where the hash is that of the fine-tuning settings.
A release holds a LoRA adapter for Chronos-2, the settings that made it (`settings.json`) and the SHA-256 of each file (`sha256sums.txt`).
How the checkpoints are made is in the paper's [supporting information](../paper/si.html#foundation-model-setup-and-fine-tuning).
Fine-tuning runs locally, not on CI.

## Install

```sh
uv add "bdbvhubrenewal @ git+https://github.com/seabbs/bdbv-hub-modelling"
uv add chronos-forecasting==2.3.2 "peft>=0.17" "torch>=2.7"
```

In a clone of this repository, `uv sync --group chronos` does the same.
Chronos-2 runs on a CPU, a CUDA GPU or an Apple GPU.
Set `GH_TOKEN` or `GITHUB_TOKEN` to a GitHub token if you list releases often, as anonymous requests are rate limited.

## Load a checkpoint

`chronos_ft.load` takes `"latest"`, the newest `chronos-ft-*` release by publish date, or a settings hash, which pins that release.
It downloads the adapter once to `~/.cache/bdbvhubrenewal/chronos-ft-<hash>`.
It checks each adapter file against the release's `sha256sums.txt`, on download and on every later use, and fails if one differs.
The check catches a damaged download, not a release whose files were replaced, as the sums come from the same release.
Pin a hash for reproducible work, and record the SHA-256 in `model.checkpoint.sha256` with your results.
A process resolves `"latest"` once, so restart it to pick up a newer release.

```python
from bdbvhubrenewal import chronos_ft

model = chronos_ft.load("e8a1f5ae338b")
model.checkpoint.hash, model.adapter
```

## Forecast a reporting triangle

`chronos_ft.forecast_triangle` takes a data frame with one row per reference time and one column per delay from 0.
Each cell is the count first reported at that delay.
The rows must be in increasing, equally spaced order, the columns the delays 0, 1, 2 and so on, and the last row's cut-off sets what is known: row $t$ of $T$ is reported up to delay $T - 1 - t$.
Cells after that are ignored, so they may be NaN.
A missing cell before it is an error; give 0 where nothing was reported.
The last delay is taken as final.

The rows reported at every delay are Chronos-2's context.
Each count at each earlier delay is a covariate, known up to the last cut-off.
This is the ragged layout of the `fm_chronos_*_ragged_ft` models ([Layouts](foundation.md#layouts)).
Our checkpoints were fine-tuned with last delays of 2 to 7 weeks or 14 to 49 days, so triangles in those ranges match their training best.

The output has one row per place, reference time and quantile level:

| Column | Meaning |
|---|---|
| `place` | `series`, or the key of each triangle when given a dict of places |
| `reference` | the row's reference time, extended past the last row for dates and numbers |
| `step` | rows after the last: 0 is the last row, below 0 the rows still being reported, above 0 rows ahead (`horizon`) |
| `quantile_level` | the level, Chronos-2's own unless `quantile_levels` is given |
| `value` | the quantile of the row's final count, never below the count reported so far |

A dict of triangles, one per place with the same rows and delays, is forecast as one group, so Chronos-2 sees every place at once.
`model=None` uses the pretrained Chronos-2, for a comparison with the fine-tuned one.

## Example

```python
--8<-- "chronos_ft.py"
```

The test suite runs this example with a stand-in for Chronos-2 (`tests/test_chronos_ft.py`).

## Limits

- The checkpoints were fine-tuned on simulated Ebola-like outbreaks for the BDBV hub. Check them on your own data before relying on them.
- Quantiles are Chronos-2's for each row on its own. They are not joint draws, so they do not give the distribution of sums over rows.
- `"latest"` changes when a new release is published. Record `model.checkpoint.hash` with your results.
