# Foundation models

Six experimental candidate models that give the hub target's releases to the Chronos-2 time series foundation model[^chronos2].
{ .page-subtitle }

The data and notation are on the [Data methods page](data.md#target-and-notation).
The models are named `fm_chronos_<source>_<layout>`, with two sources of the hub target and two layouts.
None is fitted to these data: Chronos-2 forecasts in context, from its pretrained weights or a fine-tuned checkpoint.
The two `fm_chronos_<source>_ragged_ft` models are the ragged models with Chronos-2 fine-tuned on simulated outbreaks ([Fine-tuning](#fine-tuning)).
Each is a candidate, not submitted.
The hub model `seabbs-chronos2_ft` is set up to submit `fm_chronos_daily_ragged_ft` and is off until it is opted in ([Opting in](../submission/opting-in.md)).

In brief, each model:

1. lays out the hub target of every release by place, step and release lag ([Releases](#releases));
2. puts the country and every province, each as its own hub-target series, in one block that Chronos-2 forecasts jointly ([Layouts](#layouts));
3. draws 100 trajectories from Chronos-2's quantiles ([Trajectories](#trajectories));
4. rounds the draws to counts no lower than the count already reported ([Counts](#counts)).

## Releases

The target is the hub's: confirmed cases by onset week, known and imputed onsets together.
Two sources give it with its updates.

- **Weekly** (`fm_chronos_weekly_*`): the weekly counts of each release of the target data.
  A step is an onset week.
- **Daily** (`fm_chronos_daily_*`): the daily counts of each release, known plus imputed onsets.
  A step is an onset day.

For place $p$, step $t$ and release $r$ with cut-off in step $c_r$, the release lag is $d = c_r - t$.
$Y_{p,t,d}$ is the count of step $t$ at lag $d$, from the release whose cut-off is $d$ steps after $t$.
Release $r$ gives lag $d$ only at step $c_r - d$, so each lag has at most one value per release and is missing at other steps.
$X_{p,t}$ is the count of step $t$ at the vintage, with cut-off step $c$.

## Layouts

The country and all 26 provinces are each a series in one block, each the hub target itself, so the country is not the sum of the provinces.
Chronos-2 forecasts every target series in the block jointly, from one shared last step of context, $t^*$, which each layout sets differently.
Health zones are split from the provinces as for the other fitted models ([Health zones](baselines.md#health-zones)).

- **Complete** (`*_complete`): the targets $X_{p,t}$ for $t \le t^*$.
  $t^*$ is the earliest over places of the week `randomwalk` starts its walk from, the latest week 95% complete under its delay ([Baseline models](baselines.md#forecast)).
  It comes from the delay alone, so it needs no walk spread.
- **Ragged** (`*_ragged`): the same targets, with one lag series per place and lag $d < D$, indexed by onset step as in a regression nowcast.
  Lag series $d$ holds $Y_{p,t,d}$, missing where it is not yet reported, and Chronos-2 masks the missing values.
  $t^* = c - D$, the last step whose count is at lag $D$ or more.
  $D$ is the truth delay the forecasts are scored on: the release backtest's `truth_delay` (`config/backtest.toml`), or the German benchmark's (`config/germany.toml`), in days or in weeks rounded up.
  Where `truth_delay` is `"auto"`, $D$ is resolved from the tidy data as the backtest resolves it, and, with no tidy data, estimated the same way from the releases up to the origin.
  The lag series run past $t^*$ as known future values, which is how recent counts reach the nowcast.

Chronos-2 standardises each series and takes its arcsinh, so counts go in as they are.

## Trajectories

Chronos-2 gives 21 quantiles, from 1% to 99%, of each series at each step ahead.
Each of the 100 trajectories draws its own uniform level for every series and step, and reads the quantile function there: linear between the quantiles and flat beyond them.
Draws are independent across steps and series.

## Counts

A draw is rounded to the nearest count and floored at zero.
A step's draw is no lower than its count at the vintage, and steps up to $t^*$ are their count at the vintage.
The daily source is summed to weeks, as the hub's target is weekly, and each week is no lower than its reported count.

## Patches and context

Chronos-2 reads a series in patches of 16 steps.
The context is every step from the first onset in the releases to $t^*$.

## Scoring

The models forecast final counts and model no delay.
So the release backtest scores them on the complete target only, each week against the first release the truth delay after it ends ([Evaluation](evaluation.md#real-releases)).

## Fine-tuning

The `_ft` models load one LoRA checkpoint of Chronos-2, fine-tuned on outbreaks simulated with epinowcast's generative model (`scripts/finetune_simulate.R`).
The simulator, `enw_simulate`, is on epinowcast's `feature/829-simulate-forecast` branch, pinned at commit `e084b080`, and is not yet in a release.
The simulations form the grid `r3`, a crossed design over ten factors.
These are the process (exponential growth, or renewal with a mean generation time of 3, 7 or 15 days), the generation time's shape, the trajectory of $R_t$ (eight shapes, from steady growth to several waves), the delay's family, its form (parametric, non-parametric or both) and how it changes over time (static, drift, step, periodic or random walk), reporting on every day or on one to three weekdays, a reference day-of-week effect, the observation model and the strata.
Each block of 120 simulations holds every level of each factor in equal shares, shuffled per factor.
The observation model is NB2, NB1 or Poisson.
NB1 dispersion is scaled by the expected count and the delay's $\sum_d p_d^2$ (epinowcast#937).
The strata are one place, 2 to 26 places, 2 to 8 ages, or 6 to 16 places by 3 to 7 ages.
Outbreak sizes have a median of about 3,700 cases, with 90% between 70 and 84,000.
Each simulation gives ten observation draws, each a daily reporting triangle, and its weekly releases are summed from it.

A training block cuts one draw's releases at one cut-off and lays them out as the ragged layout does, with the target past $t^*$ the outbreak's final counts (`forecasters.finetune.training_block`).
$D$ is drawn from 2 to 7 weeks, or 14 to 49 days, so the model is not tied to one $D$.
A block holds the totals over places and over ages as well as the strata, so every aggregate is the exact sum of the series below it.
Four or more flat places are grouped into regions, with region totals.
A block's targets are all levels, leaves and the national total, leaves only, or the national total only, in equal shares.
Sixteen blocks are sampled from each simulation's draws, cut-offs and sources.

Training runs in one pass while the outbreaks are simulated, at a constant learning rate of $10^{-4}$ (`scripts/finetune_stream.py`, run by `scripts/finetune_driver.sh` with `config/finetune/r3.env`).
Blocks are packed into batches by size within a memory budget, which also sets the largest block.
A checkpoint is saved after 1,000, 2,000, 4,000 and each further doubling of outbreaks.
Each is scored on held-out blocks from separate simulations, by level, and published.
Training stops from 8,000 outbreaks once a doubling cuts held-out loss by less than 1%.

A checkpoint is named by its settings hash, of the blocks, the code and the training settings.
The first release, `chronos-ft-2c2b22a03107`, was trained from earlier code, so its hash cannot be reproduced from this repository.
Later releases are trained from the code here, so retraining from it gives the same hash.
Each checkpoint is published as the repository's release `chronos-ft-<hash>`, with a `sha256sums.txt` of its files.
The models load the newest such release by publish date, unless `config/finetune.toml` pins a hash.
The checkpoint's hash and adapter SHA-256 key the stored fits, so a new release refits the `_ft` models, and each stored fit records the checkpoint it used.
The adapter is downloaded once and checked against `sha256sums.txt` and the settings hash.
The fit store resolves the release once per run, with the workflow's GitHub token, and passes its hash to every fit in `BDBV_CHRONOS_FT`.
If GitHub cannot be reached, building a model fails rather than fall back to a checkpoint downloaded before.
A fit whose adapter cannot be downloaded is stored as failed, and the fit store keeps it until its inputs change.
Such fits are retried by running the Fit store workflow with `retry_failed` (`gh workflow run fit-store.yml -R seabbs/BDBVHubRenewal -f retry_failed=true -f models="fm_chronos_weekly_ragged_ft fm_chronos_daily_ragged_ft"`).

## Limitations

- The zero-shot models use Chronos-2 as pretrained. It has not seen these data or outbreaks like them.
- The fine-tuned models have seen only simulated outbreaks, whose delays and dynamics may not match these data.
- Draws are independent across steps and places, so every weekly value of the daily models, a sum of seven daily draws, is too narrow, as are sums over weeks or places.

[^chronos2]: Ansari A F et al. (2025) Chronos-2: From univariate to universal forecasting. arXiv:2510.15821. Code: <https://github.com/amazon-science/chronos-forecasting>.
