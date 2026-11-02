# Evaluation

Scoring, and the evaluations.
{ .page-subtitle }

## Scoring

Each forecast week is scored against a later real vintage, as reported at that vintage.
Recent weeks are still incomplete in the truth, so forecasts from real vintages are re-scored against each newer real vintage ([Backtest](../backtest.md#scores-by-truth-vintage)).
Simulated forecasts are scored against the simulated final counts.

| Score set | Forecast | Weeks scored |
|---|---|---|
| Complete weeks | Final count | Weeks ended at least $K = 3$ weeks before the truth vintage. |
| All weeks | Final count | Every week in the truth vintage. |
| Vintage-matched | Count as reported by the truth vintage | Every week, so recent weeks compare like with like. |

The [real releases](#real-releases) evaluation scores the random walk models on their own targets.

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
Relative WIS is a ratio of means against `randomwalk` $b$ over the forecasts both made, and below 1 beats it.
Coverage is whether $y$ falls in the central 50% and 90% sample intervals.

## Backtest design

Each vintage is a forecast task, and every registered model forecasts every task.

| Run | Vintages | Truth |
|---|---|---|
| Development backtest | Real up to the latest model choice date | Latest real vintage, and every later real vintage for real forecasts |
| Recovery check | Simulated | Simulated final counts |
| Rolling held-out evaluation | Real vintages after each model choice | Every later real vintage |
| Real releases | Every real release with a later release | The count the next release shows, and the latest release's count for its complete weeks |
| Comparison with hub forecasts | Real where one serves the round | Latest real vintage |

## Real releases

Each real release with a later release is an origin.
The models forecast from the releases up to the origin only.
An origin with no pair of releases whose step is within the delay window, $D = 42$ days, has no delay factor and no forecast.
It is listed with no forecasts and nothing replaces it.
Two targets are scored.

**Next release.** The count the next release $o'$ shows for each of the four latest hub weeks.
`randomwalknowcast` predicts it by chaining the origin's days over the gap $c_{o'} - c_o$ between the cut-offs only, taking for each part of the gap the factor of the nearest earlier pair and never interpolating ages.
Its Skellam spread is the origin's.
The prediction is scored only when the chained steps differ from the gap by at most 1 day, as for the errors behind the spread.
`randomwalk` has no chain between releases, so its forecast samples serve for both targets.

**Complete weeks.** The count the latest release shows for the weeks it reports in full.
A week is complete when $\pi \ge 0.95$ under the delay estimate of the latest release, with $\pi$ the share of the week's final count reported ([Baseline models](baselines.md)).
Forecast weeks are scored only on this target.

The floor is the count reported at the origin, scored on the same targets as a forecast with no spread.
Scores are WIS, 50% and 90% coverage and bias, by model, origin, weeks back from the origin's week and place.
Every table counts its forecasts, $n$.
Each forecast row records the model, the start week of the walk, the number of errors behind the Skellam spread and the source of the delay, the spread and the walk.
The run stops if a model has no snapshots or two models give identical samples.

The weekly cross-check runs the port's snapshot chain ladder on hub weeks from the 2026-10-17 and 2026-10-24 releases, with a step of one week and national factors for every place.
It is drawn over the `randomwalknowcast` intervals at 2026-10-24 and is not a model in the package.

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
Each zone takes its own delay, Skellam spread and walk spread where it has at least 10 cases a week, else its province's, else national's.
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
