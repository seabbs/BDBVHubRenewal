# Foundation models

Six experimental candidate models that give the hub target's releases to the Chronos-2 time series foundation model[^chronos2].
{ .page-subtitle }

The data and notation are on the [Data methods page](data.md#target-and-notation).
The models are named `fm_chronos_<source>_<layout>`, with two sources of the hub target and two layouts.
None is fitted to these data: Chronos-2 forecasts in context, from its pretrained weights or a fine-tuned checkpoint.
The two `fm_chronos_<source>_ragged_ft` models are the ragged models with Chronos-2 fine-tuned on simulated outbreaks ([Fine-tuning](#fine-tuning)).
None is submitted.

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
- **Ragged** (`*_ragged`): the same targets, with one lag series per place and lag $d < D$, indexed by onset step as in a regression nowcast.
  Lag series $d$ holds $Y_{p,t,d}$, missing where it is not yet reported, and Chronos-2 masks the missing values.
  $t^* = c - D$, the last step whose count is at lag $D$ or more, with $D = 4$ weeks or $D = 28$ days.
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
So the release backtest scores them on the complete target only, the weeks the latest release reports in full ([Evaluation](evaluation.md#real-releases)).

## Fine-tuning

The `_ft` models load one LoRA checkpoint of Chronos-2, trained on 2,000 outbreaks simulated with epinowcast's generative model (`scripts/finetune_simulate.R`).
Each outbreak is a renewal process with a drawn log $R_t$ path, a reporting delay that drifts and differs by place, day-of-week effects and negative binomial noise, in one to 26 places.
Each gives a daily reporting triangle, and its weekly releases are summed from it.
A training block cuts an outbreak's releases at one cut-off and lays them out as the ragged layout does, with the target past $t^*$ the outbreak's final counts (`forecasters.finetune.training_block`).
$D$ is drawn from 2 to 7 weeks, or 14 to 49 days, so the model is not tied to one $D$.
Each outbreak gives up to four weekly and four daily blocks, with the country and a random subset of its places.
A block holds at most 26 provinces when weekly and five when daily, so daily blocks fit GPU memory.
Training is 1,000 steps of LoRA at learning rate $10^{-4}$, with 5% of blocks held out (`scripts/finetune_chronos.py`).

A checkpoint is named by its settings hash, of the blocks, the code and the training settings.
The models load the checkpoint whose hash is `FINETUNED_HASH`, downloaded once from the repository's release `chronos-ft-<hash>`.
A fit fails if that release is missing or holds a checkpoint with another hash.
The fit store keeps a failed fit until its inputs change, so a fit that failed before its release was published is retried by running the Fit store workflow with `retry_failed` (`gh workflow run fit-store.yml -R seabbs/BDBVHubRenewal -f retry_failed=true -f models="fm_chronos_weekly_ragged_ft fm_chronos_daily_ragged_ft"`).

## Limitations

- The zero-shot models use Chronos-2 as pretrained. It has not seen these data or outbreaks like them.
- The fine-tuned models have seen only simulated outbreaks, whose delays and dynamics may not match these data.
- The daily fine-tuned model was trained on blocks of at most five provinces but forecasts all 26 in one block.
- Draws are independent across steps and places, so every weekly value of the daily models, a sum of seven daily draws, is too narrow, as are sums over weeks or places.

[^chronos2]: Ansari A F et al. (2025) Chronos-2: From univariate to universal forecasting. arXiv:2510.15821. Code: <https://github.com/amazon-science/chronos-forecasting>.
