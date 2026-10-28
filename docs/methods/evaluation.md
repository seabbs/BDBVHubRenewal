# Evaluation

Scoring, and the four evaluations.
{ .page-subtitle }

## Scoring

Every week of a forecast is scored against a later real vintage, as reported at that vintage.
Only weeks in the truth vintage are scored.
Each score records the vintage it was scored against (`truth_vintage`), and the results pages label scores "as reported" at that date.
Recent weeks are still incomplete in the truth vintage, so their scores move as later vintages arrive.
So forecasts from real vintages are re-scored against each newer real vintage (`score_by_truth_vintage`), and the [Backtest page](../backtest.md#scores-by-truth-vintage) shows how much the truth and the scores have changed.

Scores come in three sets.

**Complete weeks.** Forecasts of the final count, scored only on weeks that ended at least $K$ weeks before the truth vintage.
With the onset day spread over the week, an onset-to-report delay $d$ and a publication lag $\ell$, the share of a week's final count published $t$ days after the week ends is

$$
s(t) = \frac{1}{7}\sum_{u=0}^{6} P(d + \ell - u \le t).
$$

$P$ is the empirical delay distribution of onset weeks old enough to be fully observed, and $\ell$ is 3 days, the longest publication lag.
$K$ is the fewest whole weeks with $s(7K) \ge 0.95$ (`bdbvhubrenewal.data.completeness.complete_weeks`).
The 2026-10-09, 2026-10-17 and 2026-10-24 delay data all give $K = 3$, with $s(21) = 0.96$ in the latest.
It is set as `complete_after_weeks` in `config/backtest.toml`.
These are the primary scores for the model choice, since every model has them.

**All weeks as reported.** The same forecasts on every week, including those still incomplete in the truth.

**Vintage-matched.** A task lists the later truth vintages (`ForecastTask.report_by`).
For each, a model may predict the counts as they will be reported by that vintage, known onsets plus imputed onsets, in rows with `as_of` set to its date.
These are scored against that vintage on every week, so recent weeks are compared like with like.
The renewal variants draw them from the fitted model (`forecast_trajectories` with `as_of`), with the same posterior draws as the final counts.
Persistence thins each sample of week $w$ to $\text{Binomial}(x, s(T - w))$, from the delays published with its vintage, and takes at least the count already published.
A model that does not predict reported counts has no vintage-matched scores.

**Origins with the lag prior.** The renewal models estimate the report to publication lag from pairs of delay files on or before the vintage.
An origin with no such pair takes the lag from a prior whose shape was set from later delay files, so it sees some later data.
Its scores carry `lag_prior` (`uses_lag_prior`), and the Backtest page also shows complete-week scores without these origins.
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

`task backtest-heldout` (`bdbvhubrenewal backtest --evaluate-holdout`) runs it.
It refuses without the record, uses real vintages only, and logs the run with the hash of the record in `results/heldout-run.yml`.
Held-out forecasts are re-run from their vintage with the model's code at the time of the run.

## Leave-one-province-out

One province's likelihood is masked while it stays in the model, so spatial coupling still acts on it.
Its onsets are then predicted and scored against the latest real vintage.
It runs on development vintages only, over the provinces with cases, and compares variants such as coupling on or off and pooling strength.
The harness is [`run_lopo`][bdbvhubrenewal.evaluation.backtest.run_lopo].
It passes the province in `ForecastTask.mask` to models that set `supports_mask = True` and skips the rest, so persistence is not run.
`renewal` honours the mask.
The single patch fits the national total, so it cannot.
The results are on the Backtest page.

## Comparison with hub forecasts

`bdbvhubrenewal compare-hub --fetch` downloads the forecasts other teams submitted to the hub (`model-output/`) and their metadata.
For each hub reference date, our registered models forecast from the matching vintage.
That is the latest real vintage that serves the round if there is one.
Otherwise it is a vintage reconstructed for the Saturday before the reference date, labelled `reconstructed`.
Every forecast for a round is scored against the latest real vintage, on the locations and weeks every model covered.
A round is pending until a real vintage later than its matching vintage exists.
Only sample forecasts of the current round's target are scored.
Files from earlier rounds target other quantities and are listed only.
