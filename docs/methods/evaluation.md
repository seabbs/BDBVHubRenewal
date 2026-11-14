# Evaluation

Scoring, and the evaluations.
{ .page-subtitle }

## What is scored

Forecasts from real releases are scored on the count each later release reports, and on each week's count once its reporting is close to complete ([Real releases](#real-releases)).
Forecasts from simulated vintages are scored against the simulated final counts.

## Scores

Scores follow [scoringutils](https://epiforecasts.io/scoringutils/) and [BVDOutbreakSize](https://github.com/epiforecasts/BVDOutbreakSize).
With sample quantiles $q_\tau$ at 23 levels, $\tau \in \{0.01, 0.025, 0.05, 0.10, \dots, 0.95, 0.975, 0.99\}$,

$$
\text{WIS}(F, y) = \frac{1}{K + 1/2} \sum_{\tau} \left(\mathbb{1}\{y < q_\tau\} - \tau\right)\left(q_\tau - y\right), \qquad K = 11.
$$

It splits into dispersion, overprediction and underprediction.
WIS on the log scale is the WIS of the transformed target, $\text{WIS}(\log(1 + x), \log(1 + y))$, never the log of WIS.
WIS on the natural scale is dominated by the places with the most cases.
On the log scale it approximates a score of the relative error, close to a score of the growth rate, so every place counts on a similar footing[^bosse].

$$
\text{Bias} = 1 - \big(P(X \le y) + P(X \le y - 1)\big),
\qquad
\text{rWIS}_m = \frac{\overline{\text{WIS}}_m}{\overline{\text{WIS}}_b}.
$$

Positive bias is overprediction.
Relative WIS is a ratio of means against a baseline $b$ over the forecasts both made, and below 1 beats it.
Relative WIS on the log scale is the same ratio of mean WIS on the log scale.
The baseline is `randomwalk`.
The interactive summaries and the checks chart have a Relative to picker that makes any model the baseline, and the overall score tables a baseline filter.
In the summaries a new baseline changes which forecasts are matched, so $n$ and the means change too.
Coverage is whether $y$ falls in the central 50% and 90% sample intervals.

### Energy score

The energy score scores one forecast of a vector $\mathbf{y}$.
For an origin, a release and a scale, $\mathbf{y}$ holds every place at that scale in every scored week, and national holds the weeks only.
It uses only the place-week cells that every compared model forecasts.
Sample $i$ of a model is its draws with sample index $i$, $\mathbf{X}_i$.
We use the fair form over $m$ samples,

$$
\text{ES}(F, \mathbf{y}) = \frac{1}{m} \sum_{i} \lVert \mathbf{X}_i - \mathbf{y} \rVert - \frac{1}{2m(m - 1)} \sum_{i \ne j} \lVert \mathbf{X}_i - \mathbf{X}_j \rVert.
$$

The energy score on the log scale is the energy score of $\log(1 + \mathbf{y})$ against $\log(1 + \mathbf{X}_i)$, never the log of the score.
Relative energy scores are ratios of means against `randomwalk` over the origins and releases both have, and $n$ counts those.
The score rewards a joint pattern across places only where draws are joint.
The draws of `renewal` and `renewalbvd` are joint across zones and provinces, one posterior draw per sample index.
`renewalsp` fits each province on its own, so its provinces are independent.
Its zones split each province draw, so they are joint within a province.
`pprandomwalknowcast` takes one posterior draw of its pooled parameters per sample index, so its places share that draw.
`randomwalk` and `randomwalknowcast` draw each place independently.
Hub teams' samples are joint across weeks within a place only, as the hub's sample format sets, so pairing their indices across places scores them as independent across places.

## Backtest design

Each vintage is a forecast task.
The fitted models (the renewal models and `pprandomwalknowcast`) forecast real vintages only, from their stored fits ([fit store](../development/cli.md#fit-store)).
The one exception is the [zone backtest](#health-zones), which fits `pprandomwalknowcast` at each origin.
The other models forecast every task.

| Run | Vintages | Truth |
|---|---|---|
| Development backtest | Real, except the latest real vintage | As for real releases |
| Recovery check | Simulated | Simulated final counts |
| Simulated scenarios | The four weekly simulated vintages up to the latest real vintage, from outbreaks calibrated to it with Rt after the data scaled by 0.75, 1 or 1.25 | Complete weeks of the simulated final counts, and the counts as reported two weeks after each vintage |
| Real releases | Every real release with a later release | The count each later release shows, and each week's count in the first release cut off the truth delay after it |
| Comparison with hub forecasts | Real where one serves the round | The count each later release shows, and the latest release's count for its complete weeks |

In the simulated scenarios and leave-one-province-out, a week is complete when it ended at least 3 weeks before the truth vintage.

## Real releases

Each real release with a later release is an origin, except the first, 2026-08-18 (`skip_origins` in `config/backtest.toml`).
That release has no delay file, so the renewal models fell back on their prior delay there.
It is also 50 days before the next release, and its weeks three or more weeks old hold 88% to 97% of the counts the 2026-11-08 release gives them, against 93% to 102% in the releases between.
It stays in the data of later origins.
The random walk models forecast from the releases up to the origin only.
The renewal models' stored forecasts from the origin are scored with them.
So are hub teams' forecasts for a round, at the latest release serving it.
A model that cannot forecast from an origin, such as a random walk model with no pair of releases within the delay window, $D = 42$ days, is listed with no forecasts there.
Two targets are scored.

**Reported.** The count each later release $o'$ shows for every forecast week it reports.
`randomwalk` is scored on its samples as they are, with no delay.
Every other model forecasts the final count $X$ of week $w$, which is mapped to what $o'$ shows by the model's own delay.

1. Day $t$ is $a = c - t$ days old at a cut-off $c$, and $\pi(a)$ is the share of its final count the model expects reported by then.
   For `randomwalknowcast`, $\pi(a) = 1 / \Gamma(a)$ under the daily delay it fits at the origin ([Baseline models](baselines.md#delay)).
   For `pprandomwalknowcast` it is the posterior median of each place's $\pi_p(a)$ from its own fit ([Baseline models](baselines.md#partially-pooled-variant)).
   For the renewal models it is the posterior median of each province's cumulative onset-to-report delay of the latest onset week, as the fit stores it.
   Ages past the end of the delay take its last value, and days after the cut-off have $\pi = 0$.
2. A week's share $\pi_w$ is the mean of $\pi$ over its seven days.
3. Each place is mapped by the model's delay at that place, a zone with none by its province's, and a place with neither is scored as forecast.
   The renewal models' shared delay is the unweighted centre of the province delays, not a national delay, so the country is the sum of the mapped provinces.
4. With $R_o$ the count reported at the origin, the count outstanding there, $X - R_o$, is thinned binomially, so each sample becomes a draw of the count reported at $o'$,

    $$
    R_o + \text{Binomial}(X - R_o, q), \qquad q = \frac{\pi_w(c_{o'}) - \pi_w(c_o)}{1 - \pi_w(c_o)},
    $$

with the sign carried when $X < R_o$, $q$ kept in $[0, 1]$, and $q = 1$ for a week fully reported at the origin.
A model with no modelled delay, such as a hub team, is scored on complete weeks only.

**Complete weeks.** Each week's count in the first release whose cut-off is at least the truth delay after the week ends.
The truth delay is `truth_delay` in `config/backtest.toml`.
By default it is the age by which 98% of the national count is reported (`truth_level`).
Each earlier release's national count of a week is set against the latest release's, at the week's age in that release rounded up to whole weeks, and the counts are pooled over releases at each age.
The first age whose pooled share reaches 98% is the truth delay.
The 2026-08-18 release is left out, as its old weeks were revised rather than late.
Weeks with no release that far after them are not scored yet.
The Backtest page gives the truth delay in use and how many forecasts it scores.
Health zones are scored on the weeks the latest release reports in full.
There a week is complete when it is at least 4 weeks back from the week holding the latest release's cut-off and grew by less than 2% since the release before.

Scores are WIS, 50% and 90% coverage and bias, by model, origin, weeks back from the week holding the origin's cut-off, and place.
Relative WIS is against `randomwalk` on the forecasts both made, for national and the provinces pooled, over all origins and by origin, and by origin and place.
Every table counts its forecasts, $n$, so a model with fewer forecasts was scored on fewer weeks or places.
The run stops if an origin has no release snapshots or two models run here give identical samples.

## Reading the figures

Fans show 100 sample trajectories, the 50% and 90% bands and the median.
In point-and-line figures, points are medians, thick lines 50% intervals and thin lines 90% intervals.
The data at the origin are dashed, shown when the origin is not the latest vintage, and the data as later reported are solid.
The dotted vertical line is the reference date.
Weeks still being reported are shaded, with hollow points.
The vertical axis of a fan ends at the top of its 90% band.
Horizon is weeks from the reference week, so negative horizons are nowcasts, 0 is the reference week and positive horizons are forecasts.

## Leave-one-province-out

One province's counts are left out of the likelihood while it stays in the model ([Renewal models](model.md#partially-pooled)).
Its onsets are scored against the latest real vintage on the complete weeks.
It runs on every real vintage for the three renewal models.

## Health zones

`randomwalk`, `randomwalknowcast` and `pprandomwalknowcast` are scored at zone level, for every zone with cases at the vintage ([Baseline models](baselines.md#health-zones)).
A week with no onsets in a zone has a count of 0.
Each model's zone samples are made to sum to its province samples, as in the submitted file, and are scored in that form.
Each real vintage before the latest is forecast, and origins with no pair of releases within the delay window are skipped, as for provinces.
Zones are scored on the reported target as for [real releases](#real-releases), with the zone delays fitted at the origin, so a zone count is never scored against a count still being reported.
Relative WIS is against `randomwalk` in the same form.
`pprandomwalknowcast` is fitted here on the zones and on the provinces.
On the Nowcast page only `randomwalk` and `randomwalknowcast` are shown at zone level.
The fitted models are shown at the country and provinces, from their stored forecasts.

## Scores by place type and plausibility { #checks }

The scores on the reported target, at province and zone level, are also averaged by the type of place and by a plausibility flag of each forecast, both decided from the counts reported at the origin.
Relative WIS in these averages is against `randomwalk`.
Week $w$ is the latest week ending on or before the origin, and $n(t)$ the count reported then for the week ending $t$, with 0 for a week with no row.
A place with no case in the 9 weeks to $w$, one with no week above 5 cases, and the rest form the three place types.
The flag is set from each origin's forecast of week $w + 1$, with median $q_{50}$ and 95% quantile $q_{95}$.
This is the round's reference week for an origin from Saturday to Tuesday, and the week before it for an origin from Wednesday to Friday.
The level is $L = \max\{(n(w - 3) + n(w - 2)) / 2,\ n(w - 1)\}$ and the recent maximum is $M = \max(1, \max_{w - 4 \le t \le w} n(t))$.
A forecast collapses if $L \ge 3$ and $q_{50} \le L / 3$.
Otherwise it blows up if $q_{50} > 3M$ and $q_{50} - M \ge 5$, or $q_{95} > 10M$ and $q_{95} - M \ge 20$.
A forecast with neither has no flag.
Every scored week of a forecast takes the forecast's place type and flag, and the scores are averaged over scored weeks.

## Comparison with hub forecasts

For each hub reference date, our unfitted models forecast from the latest real vintage serving the round.
Fitted models use their stored backtest forecasts for the same round.
Where we submitted to a round, our submitted file is a series of its own beside these main-build forecasts.
Every model is scored on the complete weeks it forecast, from the vintage serving the round, since hub teams have no modelled delay we can see.
WIS is relative to `randomwalk` on the weeks both forecast, on the [comparison page](../comparison.md).
A round is pending until a later real release exists.

## German nowcast benchmark

The German COVID-19 hospitalisation nowcasting hub[^wolffram] collected nowcasts of the 7-day hospitalisation incidence by Meldedatum, the date a case was reported positive, for Germany, its 16 states and six age groups.
`randomwalk` and `randomwalknowcast` run on its data unchanged, as follows.

1. Each day's data version is a release whose cut-off is that day, and the Meldedatum is the onset date. A release's count of a day is the count reported by then in the hub's reporting triangle, with the changes after 80 days put at 81 days, as epinowcast's `germany_covid19_hosp` does. A day's last count is carried to every later release.
2. Germany is the country. The states are its provinces in one task and Germany's age groups its provinces in another. Germany's nowcast is the sum of the states.
3. Hub weeks end on Saturdays, so every date of a nowcast moves forward by the 0 to 6 days to the next Saturday. The 7-day sums ending on the nowcast date and 7, 14, 21 and 28 days before it are then hub weeks, and the nowcasts move back.

The nowcast dates are the paper's, every day from 2021-11-22 to 2022-04-29.
They are split at their midpoint.
The 80 dates before 2022-02-10 are for developing and checking our setup.
The 79 dates from 2022-02-10 on are held out for the paper and are not scored here.
The hub collected nowcasts as quantiles at the levels 2.5%, 10%, 25%, 50%, 75%, 90% and 97.5%.
Our models' samples are reduced to their quantiles at these levels, as for a hub submission.
Our models and the hub teams are then scored the same way, as the paper did.
The scores are the WIS of the seven quantiles, the absolute error of the median and the coverage of the 50% and 95% intervals.
Each 7-day sum's truth is its count in the data 40 days after it ends, the paper's 40-day truth.
The backtest scores its complete weeks the same way, on the first release the truth delay after each week ([Real releases](#real-releases)).
Relative scores divide a model's mean score by `randomwalknowcast`'s on the nowcasts both made.
The paper's relative scores divide by the hub's own baseline, `KIT-frozen_baseline`.
Both baselines are set in `config/germany.toml`.
`bdbvhubrenewal germany-paper` writes every forecast, with its truth, as the paper's long table.

The teams' nowcasts are those the paper gathered, prospective and retrospective, at the horizons our models give.
Two fixes are the paper's.
RKI's quantiles above one million are errors, replaced by its -1 day nowcast.
Epiforecasts' missing 10% quantile is filled from the normal through its 2.5% and 25% quantiles.
Under 1% of the teams' forecasts have quantiles that cross, and their values are sorted within the forecast.
The paper also filled some missing ILM and LMU nowcasts from later model versions, and those are not scored here.

[^wolffram]: Wolffram D, Abbott S, an der Heiden M, Funk S, Günther F, Hailer D, et al. (2023). Collaborative nowcasting of COVID-19 hospitalization incidences in Germany. PLOS Computational Biology 19(8): e1011394. [doi:10.1371/journal.pcbi.1011394](https://doi.org/10.1371/journal.pcbi.1011394).

[^bosse]: Bosse NI, Abbott S, Cori A, van Leeuwen E, Bracher J, Funk S (2023). Scoring epidemiological forecasts on transformed scales. PLOS Computational Biology 19(8): e1011393. [doi:10.1371/journal.pcbi.1011393](https://doi.org/10.1371/journal.pcbi.1011393).
