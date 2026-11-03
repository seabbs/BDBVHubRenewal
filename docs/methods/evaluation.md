# Evaluation

Scoring, and the evaluations.
{ .page-subtitle }

## Scoring

Every model is scored on the count each later real release reports, with its final-count forecast turned into a forecast of that count ([Real releases](#real-releases)).
A final count is never scored against a count still being reported.
The development backtest scores complete weeks against each later real vintage as a secondary check ([Backtest](../backtest.md#scores-by-truth-vintage)).
Simulated forecasts are scored against the simulated final counts.

| Score set | Forecast | Weeks scored |
|---|---|---|
| Complete weeks | Final count | Weeks ended at least $K = 3$ weeks before the truth vintage. These can still grow in later vintages. |
| Vintage-matched | Count as reported by the truth vintage | Every week, so recent weeks compare like with like. |

**Vintage-matched.** The renewal variants draw reported counts from the same posterior draws as the final counts.

## Scores

Scores follow [scoringutils](https://epiforecasts.io/scoringutils/) and [BVDOutbreakSize](https://github.com/epiforecasts/BVDOutbreakSize).
With sample quantiles $q_\tau$ at 23 levels, $\tau \in \{0.01, 0.025, 0.05, 0.10, \dots, 0.95, 0.975, 0.99\}$,

$$
\text{WIS}(F, y) = \frac{1}{K + 1/2} \sum_{\tau} \left(\mathbb{1}\{y < q_\tau\} - \tau\right)\left(q_\tau - y\right), \qquad K = 11.
$$

It splits into dispersion, overprediction and underprediction.
Log WIS applies it to $\log(1 + y)$ and $\log(1 + x)$, so it measures relative error and small provinces count.

$$
\text{Bias} = 1 - \big(P(X \le y) + P(X \le y - 1)\big),
\qquad
\text{rWIS}_m = \frac{\overline{\text{WIS}}_m}{\overline{\text{WIS}}_b}.
$$

Positive bias is overprediction.
Relative WIS is a ratio of means against a baseline $b$ over the forecasts both made, and below 1 beats it.
The baseline is `randomwalk` in the backtests and `randomwalknowcast` in the comparison with hub forecasts.
Coverage is whether $y$ falls in the central 50% and 90% sample intervals.

## Backtest design

Each vintage is a forecast task, and every registered model forecasts every task.

| Run | Vintages | Truth |
|---|---|---|
| Development backtest | Real up to the latest model choice date | Latest real vintage, and every later real vintage for real forecasts |
| Recovery check | Simulated | Simulated final counts |
| Rolling held-out evaluation | Real vintages after each model choice | Every later real vintage |
| Real releases | Every real release with a later release | The count each later release shows, and the latest release's count for its complete weeks |
| Comparison with hub forecasts | Real where one serves the round | Latest real vintage, complete weeks |

## Real releases

Each real release with a later release is an origin.
The random walk models forecast from the releases up to the origin only.
The renewal models' stored forecasts from the origin, and hub teams' forecasts for the round the origin serves, are scored with them.
An origin with no pair of releases whose gap is within the delay window, $D = 42$ days, has no delay factor and no forecast.
It is listed with no forecasts and nothing replaces it.
Two targets are scored.

**Reported.** The count each later release $o'$ shows for every forecast week it reports.
Every model forecasts the final count $X$ of week $w$.
Day $t$ is $a = c - t$ days old at a cut-off $c$, and the share of its final count reported then is $\pi(a) = 1 / G(a)$, with $G(a)$ the growth from age $a$ to $D$ under the daily delay of the origin ([Baseline models](baselines.md#delay)).
Days after the cut-off have $\pi = 0$, and a week's share $\pi_w$ is the mean over its seven days.
With $R_o$ the count reported at the origin, the count outstanding there, $X - R_o$, is thinned binomially, so each sample becomes a draw of the count reported at $o'$,

$$
R_o + \text{Binomial}(X - R_o, q), \qquad q = \frac{\pi_w(c_{o'}) - \pi_w(c_o)}{1 - \pi_w(c_o)},
$$

with the sign carried when $X < R_o$, and a week fully reported at the origin has $q = 1$.
Only the delay fitted at the origin is used.
Its mean is the growth over the gap to $o'$ for `randomwalknowcast`, and the floor maps to itself.

**Complete weeks.** The count the latest release shows for the weeks it reports in full.
A week is complete when it is at least 4 weeks old at the latest release and grew by less than 2% since the release before.

The floor is the count reported at the origin, scored on both targets as a forecast with no spread.
Scores are WIS, 50% and 90% coverage and bias, by model, origin, weeks back from the origin's week and place.
Relative WIS is against `randomwalk` on the forecasts both made, by origin, for national and the provinces pooled, and by place.
Every table counts its forecasts, $n$.
The run stops if a model has no snapshots or two models run here give identical samples.

The weekly cross-check runs the snapshot chain ladder on hub weeks from the latest two releases with cut-offs a week apart, with a gap of one week and the national delay for every place.
It checks the daily ladder for consistency on weekly data.
It is drawn over the `randomwalknowcast` intervals at the later release.

## Rolling held-out evaluation

Each model choice is logged with its date, and the development set rolls forward with them.
Development runs use every vintage up to the latest choice date.
Each real vintage after a choice is forecast by the model chosen then and by `randomwalk`, with the code at the time of the run.
Each forecast is scored when the next vintage arrives and re-scored as newer ones do.

## Leave-one-province-out

One province's likelihood is masked while it stays in the model, so coupling still acts on it.
Its onsets are scored against the latest real vintage.
It runs on development vintages, for models that support the mask, so not `renewalsp` or `randomwalk`.

## Health zones

`randomwalk` and `randomwalknowcast` run on each zone's own release snapshots, for every zone with cases at the vintage.
Zones use the same models as provinces.
Each zone takes its delay and spreads by the rules of the [Baseline models](baselines.md#own-or-pooled-estimates) page.
Each model is scored as it runs, and again with its zone samples made to sum to its province samples, as in the submitted file.
Each real vintage before the latest is forecast, and origins with no pair of releases within the delay window are skipped, as for provinces.
Zones are scored on every week against the zone counts as reported at the latest real vintage.
Relative WIS is against `randomwalk` run at zone level.
`bdbvhubrenewal zone-backtest` runs it and writes `results/zone-scores.csv`.

## Comparison with hub forecasts

For each hub reference date, our unfitted models forecast from the latest real vintage serving the round.
Fitted models use their stored backtest or held-out forecasts for the same round.
Every model is scored on the weeks it forecast, against the latest real vintage, and the [comparison page](../comparison.md) reports complete weeks.
WIS is relative to `randomwalknowcast` on the weeks both forecast.
A round is pending until a later real vintage exists.
