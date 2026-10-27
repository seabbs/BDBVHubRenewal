# Evaluation

How forecasts are scored, and the four ways they are evaluated.
{ .page-subtitle }

## Scoring

Every week of a forecast is scored against a later real vintage, as reported at that vintage.
No week is left out for being recent.
Each score records the vintage it was scored against (`truth_vintage`), and the results pages label scores "as reported" at that date.
Recent weeks are still incomplete in the truth vintage, so their scores move as later vintages arrive.
So forecasts from real vintages are re-scored against each newer real vintage (`score_by_truth_vintage`), and the [Backtest page](../backtest.md#scores-as-the-truth-is-revised) shows how much the truth and the scores have changed.
Simulated forecasts are scored against the simulated final counts.
Scores follow [scoringutils](https://epiforecasts.io/scoringutils/) and [BVDOutbreakSize](https://github.com/epiforecasts/BVDOutbreakSize).

**Weighted interval score.**
With sample quantiles $q_\tau$ at the 23 levels $\tau \in \{0.01, 0.025, 0.05, 0.10, \dots, 0.95, 0.975, 0.99\}$ (11 central intervals plus the median),

$$
\text{WIS}(F, y) = \frac{1}{K + 1/2} \sum_{\tau} \left(\mathbb{1}\{y < q_\tau\} - \tau\right)\left(q_\tau - y\right), \qquad K = 11.
$$

This equals the interval form $\frac{1}{K + 1/2}\left(\frac{1}{2}|y - m| + \sum_k \frac{\alpha_k}{2} \text{IS}_{\alpha_k}\right)$.
It splits into dispersion, overprediction and underprediction.
**Log WIS** applies the same score to $\log(1 + y)$ and $\log(1 + x)$, so it measures relative error and small provinces count.

**Coverage.** Whether $y$ falls in the central 50% and 90% sample intervals.

**Bias.** For counts, $B = 1 - \left(P(X \le y) + P(X \le y - 1)\right)$, in $[-1, 1]$.
Positive values mean overprediction.

**Relative WIS.** For model $m$ and baseline $b$ over the forecasts both made,

$$
\text{rWIS}_m = \frac{\overline{\text{WIS}}_m}{\overline{\text{WIS}}_b}.
$$

Values below 1 beat persistence.
This is a ratio of means, as in BVDOutbreakSize, not a mean of ratios.

## Backtest design

Each vintage is a forecast task.
Every registered model forecasts every task, so a new model appears in every evaluation without other changes.

| Run | Vintages | Truth |
|---|---|---|
| Development backtest | Real and reconstructed up to the latest model choice date, reported separately | Latest real vintage; real forecasts also against every later real vintage |
| Recovery check | Simulated | Simulated final counts |
| Rolling held-out evaluation | Real vintages after the first model choice date | Every later real vintage |
| Comparison with hub forecasts | Real where one serves the round, else reconstructed | Latest real vintage |

Reconstructed vintages after the latest model choice date are never built.

## Rolling held-out evaluation

`results/model-choice.yml` logs the model chosen for submission and every change, each with a date.
Every real vintage after the first choice date is held out.
A held-out vintage is forecast by the model chosen at the time, the latest choice dated before the vintage, and by persistence.
Each forecast is scored against every later real vintage.
So a forecast gets its first score when the next vintage is published, and is re-scored as newer ones arrive.
The development backtest only uses vintages up to the latest choice date.

`task backtest-heldout` (`bdbv-hub-renewal backtest --evaluate-holdout`) runs it.
It refuses without the record, uses real vintages only, and logs the run with the hash of the record in `results/heldout-run.yml`.
Held-out forecasts are re-run from their vintage with the model's code at the time of the run.

## Leave-one-province-out

One province's likelihood is masked while it stays in the model, so spatial coupling still acts on it.
Its onsets are then predicted and scored against the latest real vintage.
It runs on development vintages only, over the provinces with cases, and compares variants such as coupling on or off and pooling strength.
The harness is [`run_lopo`][bdbv_hub_renewal.backtest.run_lopo].
It passes the province in `ForecastTask.mask` to models that set `supports_mask = True` and skips the rest, so persistence is not run.
The renewal model will honour the mask, and its results will be reported on the Backtest page.

## Comparison with hub forecasts

`bdbv-hub-renewal compare-hub --fetch` downloads the forecasts other teams submitted to the hub (`model-output/`) and their metadata.
For each hub reference date, our registered models forecast from the matching vintage.
That is the latest real vintage that serves the round if there is one.
Otherwise it is a vintage reconstructed for the Saturday before the reference date, labelled `reconstructed`.
Every forecast for a round is scored against the latest real vintage, on the locations and weeks every model covered.
A round is pending until a real vintage later than its matching vintage exists.
Only sample forecasts of the current round's target are scored.
Files from earlier rounds target other quantities and are listed only.
