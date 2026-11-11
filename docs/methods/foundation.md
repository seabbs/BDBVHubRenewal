# Foundation models

Six experimental candidate models that give the hub target's releases to the pretrained Chronos-2 time series foundation model[^chronos2].
{ .page-subtitle }

The data and notation are on the [Data methods page](data.md#target-and-notation).
The models are named `fm_chronos_<source>_<layout>`, with two sources of the hub target and three layouts.
None is fitted to these data: Chronos-2 forecasts from its pretrained weights, in context.
None is submitted.

In brief, each model:

1. lays out the hub target of every release by place, step and release lag ([Releases](#releases));
2. puts the country, every province and their health zones, each as its own series, in one group that Chronos-2 forecasts jointly ([Layouts](#layouts));
3. draws 100 trajectories from Chronos-2's quantiles ([Trajectories](#trajectories));
4. rounds the draws to counts no lower than the count already reported ([Counts](#counts)).

## Releases

The target is the hub's: confirmed cases by onset week, known and imputed onsets together.
Two sources give it with its updates.

- **Weekly** (`fm_chronos_weekly_*`): the weekly counts of each release of the target data.
  A step is an onset week.
- **Daily** (`fm_chronos_daily_*`): the daily counts of each release, known plus imputed onsets.
  A step is an onset day, and the forecast is summed to weeks at the end.

For place $p$, step $t$ and release $r$ with cut-off in step $c_r$, the release lag is $d = c_r - t$.
$Y_{p,t,d}$ is the count of step $t$ at lag $d$, from the release whose cut-off is $d$ steps after $t$.
It is missing where no release has that lag, so each lag has at most one value per release.
A count at lag $D$ or more is taken as final: $D = 4$ weeks for the weekly source and $D = 28$ days for the daily source.
At the vintage, with cut-off step $c$, steps up to $t^* = c - D$ are final, and $X_{p,t}$ is their count.

## Layouts

All places are in one group of series, so Chronos-2's group attention reads them together: the country, the provinces and the health zones of those provinces with zone data.
The country is its own series and is not the sum of the provinces.
Zone counts start with the zone releases, and are zero before them.
The context is the steps up to $t^*$, and every step after it is forecast.

- **Complete** (`*_complete`): the targets are $X_{p,t}$ for $t \le t^*$, one series per place.
- **Covariate** (`*_cov`): the same targets, with each lag $d < D$ of each place as a covariate, $Y_{p,t,d}$ over all steps.
  After $t^*$ the covariate holds the counts already reported, missing elsewhere, and Chronos-2 masks the missing values.
- **Ragged** (`*_ragged`): each place has one series per lag, as in a regression nowcast where the delay indexes the series.
  The targets are $X_{p,t}$ and $Y_{p,t,d}$ for every $d < D$, to $t^*$.
  Chronos-2 forecasts every series in a group forward from the same last step, so it cannot fill the missing tail of a series that ends earlier.
  So each lag series also enters as a covariate holding its own reported values after $t^*$, and Chronos-2 forecasts every lag series and the final count from there.

Chronos-2 standardises each series and takes its arcsinh, so counts go in as they are.
It scales each series on its own, which maps a constant series to zero, so a lag is told apart by its own values and not by a label.

## Trajectories

Chronos-2 gives 21 quantiles, from 1% to 99%, of each series at each step ahead.
Each of the 100 trajectories draws its own uniform level for every series and step, and reads the quantile function there: linear between the quantiles and flat beyond them.
Draws are independent across steps and series, as Chronos-2 gives no joint distribution and its authors give no way to build one.

## Counts

A draw is rounded to the nearest count and floored at zero.
A step's draw is no lower than its count at the vintage, and steps up to $t^*$ are their count at the vintage.
Daily draws are summed to weeks, and each week is no lower than the week's reported count.
Small counts need nothing more: a series of zeros and ones keeps draws near zero, and the floor removes negative draws.

## Patches and context

Chronos-2 reads a series in patches of 16 steps.
The weekly source has about 34 weeks of context at the latest vintages, three patches.
The daily source has about 240 days, 16 patches.

## Reported counts

The release backtest maps a model's final counts to what later releases show through a delay ([Evaluation](evaluation.md)).
These models use the daily chain ladder delay of `randomwalknowcast` ([Baseline models](baselines.md#delay)), fitted at the same vintage.

## Limitations

- The hub target has few releases, so each lag has a handful of values per place, most of them in the last weeks.
- Chronos-2 is used as pretrained. It has not seen these data or outbreaks like them.
- Trajectories are independent across steps and places, so sums over weeks or places are too narrow.
- The zones' own forecasts are not used yet: zone forecasts are split from the provinces as for the other fitted models ([Health zones](baselines.md#health-zones)).

[^chronos2]: Ansari A F et al. (2025) Chronos-2: From univariate to universal forecasting. arXiv:2510.15821. Code: <https://github.com/amazon-science/chronos-forecasting>.
