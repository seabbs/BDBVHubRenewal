# Evaluation

Scoring, and the evaluations.
{ .page-subtitle }

## Scoring

Every model is scored on the count each later real release reports, with its final-count forecast turned into a forecast of that count ([Real releases](#real-releases)).
The release scores never compare a final count with a count still being reported.
Complete weeks of the latest release are a secondary check ([Real releases](#real-releases)).
Simulated forecasts are scored against the simulated final counts.

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
`renewal` and `renewalbvd` draws are joint across provinces, one posterior draw per sample index.
`renewalsp` fits each province on its own, so its draws are independent across provinces.
Zones split each province draw, so zones are joint within a province for every renewal model.
`randomwalk` and `randomwalknowcast` draw each place independently.
Hub teams' samples are joint across weeks within a place only, as the hub's sample format sets, so pairing their indices across places scores them as independent across places.

## Backtest design

Each vintage is a forecast task, and every registered model forecasts every task.

| Run | Vintages | Truth |
|---|---|---|
| Development backtest | Real up to the latest model choice date, except the latest real vintage | As for real releases |
| Recovery check | Simulated | Simulated final counts |
| Simulated scenarios | The four weekly simulated vintages up to the latest real vintage, from outbreaks calibrated to it with Rt after the data scaled by 0.75, 1 or 1.25 | Complete weeks of the simulated final counts, and the counts as reported two weeks after each vintage |
| Rolling held-out evaluation | Real vintages after each model choice | As for real releases |
| Real releases | Every real release up to the latest model choice date with a later release | The count each later release shows, and the latest release's count for its complete weeks |
| Comparison with hub forecasts | Real where one serves the round | As for real releases |

## Real releases

Each real release up to the latest model choice date with a later release is an origin.
The random walk models forecast from the releases up to the origin only.
The renewal models' stored forecasts from the origin are scored with them.
So are hub teams' forecasts for a round, at the latest release serving it.
A model that cannot forecast from an origin, such as a random walk model with no pair of releases within the delay window, $D = 42$ days, is listed with no forecasts there.
Two targets are scored.

**Reported.** The count each later release $o'$ shows for every forecast week it reports.
`randomwalk` is scored on its samples as they are, with no delay.
Every other model forecasts the final count $X$ of week $w$ and is mapped to what $o'$ shows by its own modelled delay.
Day $t$ is $a = c - t$ days old at a cut-off $c$, and $\pi(a)$ is the share of its final count the model expects reported by then.
For `randomwalknowcast` and `pprandomwalknowcast`, $\pi(a) = 1 / G(a)$ under the daily delay each fits at the origin ([Baseline models](baselines.md#delay)).
For the renewal models it is the posterior median of the onset-to-report delay each fit stores, the pooled one for every place or each province's own.
Each place is mapped by the model's delay at that place, and a zone with none by its province's.
Where a model has no delay for the country, as `renewalsp`, the country is the sum of its mapped provinces.
Days after the cut-off have $\pi = 0$, and a week's share $\pi_w$ is the mean over its seven days.
With $R_o$ the count reported at the origin, the count outstanding there, $X - R_o$, is thinned binomially, so each sample becomes a draw of the count reported at $o'$,

$$
R_o + \text{Binomial}(X - R_o, q), \qquad q = \frac{\pi_w(c_{o'}) - \pi_w(c_o)}{1 - \pi_w(c_o)},
$$

with the sign carried when $X < R_o$, and a week fully reported at the origin has $q = 1$.
A model with no modelled delay, such as a hub team, is scored on complete weeks only.

**Complete weeks.** The count the latest release shows for the weeks it reports in full.
A week is complete when it is at least 4 weeks back from the week holding the latest release's cut-off and grew by less than 2% since the release before.

Scores are WIS, 50% and 90% coverage and bias, by model, origin, weeks back from the origin's week and place.
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

## Rolling held-out evaluation

Each model choice is logged with its date, and the development set rolls forward with them.
Development runs use every vintage up to the latest choice date.
Each real vintage after a choice is forecast by the model chosen then, the latest choice dated before it.
It is also forecast by `randomwalk` and `randomwalknowcast`, and by every fitted model from its stored fit.
The run supplies the fitted models' forecasts to the [Nowcast](../nowcast.md) and [Comparison](../comparison.md) pages, and has no page of its own.

## Leave-one-province-out

One province's likelihood is masked while it stays in the model, so coupling still acts on it.
Its onsets are scored on every week against the latest real vintage.
It runs on development vintages for the models that support the mask, `renewal` and `renewalbvd`.

## Health zones

`randomwalk` and `randomwalknowcast` run on each zone's own release snapshots, for every zone with cases at the vintage.
Zones use the same models as provinces.
A week with no onsets in a zone has a count of 0.
Each zone takes its delay and spreads by the rules of the [Baseline models](baselines.md#own-or-pooled-estimates) page.
Each model's zone samples are made to sum to its province samples, as in the submitted file, and are scored in that form.
Each real vintage before the latest is forecast, and origins with no pair of releases within the delay window are skipped, as for provinces.
Zones are scored on the reported target as for [real releases](#real-releases), with the zone delays fitted at the origin, so a zone count is never scored against a count still being reported.
Relative WIS is against `randomwalk` in the same form.
On the Nowcast page the fitted models' zone samples at the latest release are those of their stored fits, where they have one.

## Checks

The scores on the reported target are also averaged by the type of place and by a plausibility flag of each forecast, both decided from the counts reported at the origin.
Week $w$ is the latest week ending on or before the origin, and $c(t)$ the count reported then for the week ending $t$.
A place is silent if no case is reported in the `RECENT_WEEKS` weeks to $w$, the random walk's walk window (`walk_weeks`), small if no week has more than 5 cases, and larger otherwise.
The flag is set from the forecast of the reference week $w + 1$, with median $q_{50}$ and 95% quantile $q_{95}$.
The level is $\ell = \max\{(c(w - 3) + c(w - 2)) / 2,\ c(w - 1)\}$ and the recent maximum is $r = \max(1, \max_{w - 4 \le t \le w} c(t))$.
A forecast collapses if $\ell \ge 3$ and $q_{50} \le \ell / 3$.
It blows up if $q_{50} > 3r$ and $q_{50} - r \ge 5$, or $q_{95} > 10r$ and $q_{95} - r \ge 20$.
The thresholds pick out forecasts far from every recent count.
A forecast with neither has no flag.
Every scored week of a forecast takes the forecast's place type and flag, and the scores are averaged over scored weeks.

## Comparison with hub forecasts

For each hub reference date, our unfitted models forecast from the latest real vintage serving the round.
Fitted models use their stored backtest or held-out forecasts for the same round.
Where we submitted to a round, our submitted file is a series of its own beside these main-build forecasts.
Every model is scored on the complete weeks it forecast, as for [real releases](#real-releases), from the vintage serving the round, since hub teams have no modelled delay we can see.
WIS is relative to `randomwalk` on the weeks both forecast, on the [comparison page](../comparison.md).
A round is pending until a later real release exists.

[^bosse]: Bosse NI, Abbott S, Cori A, van Leeuwen E, Bracher J, Funk S (2023). Scoring epidemiological forecasts on transformed scales. PLOS Computational Biology 19(8): e1011393. [doi:10.1371/journal.pcbi.1011393](https://doi.org/10.1371/journal.pcbi.1011393).
