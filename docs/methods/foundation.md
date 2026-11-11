# Foundation models

Four experimental candidate models that give the hub target's releases to the pretrained Chronos-2 time series foundation model[^chronos2].
{ .page-subtitle }

The data and notation are on the [Data methods page](data.md#target-and-notation).
The models are named `fm_chronos_<source>_<layout>`, with two sources of the hub target and two layouts.
None is fitted to these data: Chronos-2 forecasts from its pretrained weights, in context.
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
It is missing where no release has that lag, so each lag has at most one value per release.
$X_{p,t}$ is the count of step $t$ at the vintage, with cut-off step $c$.

## Layouts

The country and all 26 provinces are each a series in one block, each the hub target itself, so the country is not the sum of the provinces.
Chronos-2's group attention reads the block together and forecasts every target series from one shared last step of context, $t^*$, which each layout sets differently.
Health zones are split from the provinces as for the other fitted models ([Health zones](baselines.md#health-zones)).
This is the one place a forecast is split rather than forecast.

- **Complete** (`*_complete`): the targets $X_{p,t}$ for $t \le t^*$.
  $t^*$ is the week `randomwalk` starts its walk from ([Baseline models](baselines.md#forecast)), the earliest over the places it walks.
- **Ragged** (`*_ragged`): the same targets, with one lag series per place and lag $d < D$, indexed by onset step as in a regression nowcast.
  Lag series $d$ holds $Y_{p,t,d}$, missing where it is not yet reported, and Chronos-2 masks the missing values.
  $t^* = c - D$, the last step with every lag in, with $D = 4$ weeks or $D = 28$ days.
  The lag series run past $t^*$ as values known into the future.
  They have to: Chronos-2 outputs only after the block's shared end $t^*$, and it takes no known future values of a target series.
  So a recent week's count so far reaches its nowcast only as its cell at lag $c - t$ in a lag series; without it the recent weeks would have no nowcast.

A covariate layout, with the lag series as covariates of the target series, is the ragged layout here, so it is not a separate model.
Chronos-2 standardises each series and takes its arcsinh, so counts go in as they are.
It scales each series on its own, which maps a constant series to zero, so a lag is told apart by its own values and not by a label.

## Trajectories

Chronos-2 gives 21 quantiles, from 1% to 99%, of each series at each step ahead.
Each of the 100 trajectories draws its own uniform level for every series and step, and reads the quantile function there: linear between the quantiles and flat beyond them.
Draws are independent across steps and series, as Chronos-2 gives no joint distribution and its authors give no way to build one.

## Counts

A draw is rounded to the nearest count and floored at zero.
A step's draw is no lower than its count at the vintage, and steps up to $t^*$ are their count at the vintage.
The daily source is summed to weeks, as the hub's target is weekly, and each week is no lower than its reported count.
Small counts need nothing more: a series of zeros and ones keeps draws near zero, and the floor removes negative draws.

## Patches and context

Chronos-2 reads a series in patches of 16 steps.
The weekly source has about 34 weeks of context at the latest vintages, three patches.
The daily source has about 240 days, 16 patches.
The weekly ragged block is 27 targets and 108 lag series; the daily one is 27 targets and 756 lag series.
Chronos-2 runs on a CUDA or Apple GPU when there is one, and on the CPU otherwise.

## Reported counts

The release backtest maps a model's final counts to what later releases show through a delay ([Evaluation](evaluation.md)).
These models use the daily chain ladder delay of `randomwalknowcast` ([Baseline models](baselines.md#delay)), fitted at the same vintage.

## Limitations

- The hub target has few releases, so each lag has a handful of values per place, most of them in the last weeks.
- Chronos-2 is used as pretrained. It has not seen these data or outbreaks like them.
- Trajectories are independent across steps and places, so sums over weeks or places are too narrow.

[^chronos2]: Ansari A F et al. (2025) Chronos-2: From univariate to universal forecasting. arXiv:2510.15821. Code: <https://github.com/amazon-science/chronos-forecasting>.
