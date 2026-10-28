# Evaluation

Scoring, and the four evaluations.
{ .page-subtitle }

## Scoring

Each forecast week is scored against a later real vintage, as reported at that vintage.
Recent weeks are still incomplete in the truth, so forecasts from real vintages are re-scored against each newer real vintage ([Backtest](../backtest.md#scores-by-truth-vintage)).
Simulated forecasts are scored against the simulated final counts.

| Score set | Forecast | Weeks scored |
|---|---|---|
| Complete weeks | Final count | Weeks ended at least $K$ weeks before the truth vintage. Primary, since every model has them. |
| All weeks | Final count | Every week in the truth vintage. |
| Vintage-matched | Count as reported by the truth vintage | Every week, so recent weeks compare like with like. |

**Complete weeks.** With the onset uniform over the week, onset-to-report delay $d$ and publication lag $\ell = 3$ days, the share of a week's final count published $t$ days after it ends is

$$
s(t) = \frac{1}{7}\sum_{u=0}^{6} P(d + \ell - u \le t),
$$

with $P$ the empirical delay distribution of fully observed onset weeks.
$K$ is the fewest whole weeks with $s(7K) \ge 0.95$.
The 2026-10-09 to 2026-10-24 delay data all give $K = 3$.

**Vintage-matched.** The renewal variants draw reported counts from the same posterior draws as the final counts.
Persistence thins each sample $x$ of week $w$ to $\max\{Y_{p,v}(w),\ \text{Binomial}(x, s(T - w))\}$.

**Lag prior.** An origin with no earlier pair of delay files takes the publication lag from a prior shaped by later files.
Its scores are flagged, and the Backtest page also shows complete-week scores without them.

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
Relative WIS is a ratio of means against persistence $b$ over the forecasts both made, and below 1 beats it.
Coverage is whether $y$ falls in the central 50% and 90% sample intervals.

## Backtest design

Each vintage is a forecast task, and every registered model forecasts every task.

| Run | Vintages | Truth |
|---|---|---|
| Development backtest | Real and reconstructed up to the latest model choice date | Latest real vintage, and every later real vintage for real forecasts |
| Recovery check | Simulated | Simulated final counts |
| Rolling held-out evaluation | Real vintages after the first model choice date | Every later real vintage |
| Comparison with hub forecasts | Real where one serves the round, else reconstructed | Latest real vintage |

## Rolling held-out evaluation

Each model choice is logged with its date.
Every real vintage after the first choice date is forecast by the model chosen at the time and by persistence, with the code at the time of the run.
Each forecast is scored when the next vintage arrives and re-scored as newer ones do.

## Leave-one-province-out

One province's likelihood is masked while it stays in the model, so coupling still acts on it.
Its onsets are scored against the latest real vintage.
It runs on development vintages, for models that support the mask, so not `renewalsp` or persistence.

## Comparison with hub forecasts

For each hub reference date, our models forecast from the latest real vintage serving the round, or else a vintage reconstructed for the Saturday before.
All sample forecasts of the current target are scored against the latest real vintage, on the locations and weeks every model covered.
A round is pending until a later real vintage exists.
