# Methods

The maths of the baseline, the scores, the synthetic vintages and the planned model.
{ .page-subtitle }

## Target and notation

The target is $y_{p}(w)$, the final count of confirmed cases in province $p$ with symptom onset in epi week $w$ (Sunday to Saturday, labelled by its Saturday).
It includes onsets the hub imputes for cases reported without one.
A vintage $v$ is the data published on date $T_v$.
It gives, for each $p$ and $w$, known onsets $K_{p,v}(w)$, imputed onsets $I_{p,v}(w)$ and their total $Y_{p,v}(w) = K_{p,v}(w) + I_{p,v}(w)$.
A forecast made from vintage $v$ serves the round with reference date $r_v$, the first Saturday whose submission window (Sunday to Tuesday, Kinshasa time) closes on or after $T_v$.
Horizon $h$ is the number of weeks from $r_v$ to $w$.
Files cover $h = -4, \dots, 3$.
The national value is the sum over provinces, sample by sample.

## Persistence baseline

Let $c$ be the last week judged complete.
Week $w$ is complete if its midpoint is at least $L$ days before $T_v$, where $L$ is the 90th centile of the delays published with the vintage (10 days if none).
For $w \le c$ the forecast is the published count $Y_{p,v}(w)$.
For $w = c + h$ with $h \ge 1$,

$$
\log(1 + \lambda_{p}(c + h)) = \log(1 + Y_{p,v}(c)) + \sum_{j=1}^{h} \epsilon_{p,j},
\qquad \epsilon_{p,j} \sim \mathcal{N}(0, \sigma_p^2),
$$

$$
\tilde y_p(c + h) \sim \text{Poisson}(\lambda_p(c + h)).
$$

$\sigma_p$ is the standard deviation of $\Delta \log(1 + Y_{p,v})$ over the last 8 complete weeks, floored at 0.1.
The median stays flat and the spread grows with the horizon.

## Scoring

Forecasts are scored against the latest real vintage, only for weeks at least 2 weeks before its epi week, and only for forecasts made from earlier vintages.
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

## Testing with synthetic vintages

There are only four real vintages.
Two kinds of synthetic vintage extend testing.
Every vintage carries a `kind` (`real`, `reconstructed` or `simulated`), and every score and plot shows it.

### Reconstructed vintages

These rebuild, for any past Saturday $A$, what the hub would roughly have shown, from the latest real vintage.
Cases reach a vintage some days after report, so the cut-off is $A' = A - 3$ days.

- **Known onsets.** Each delay row is a case with onset week $W$ and delay $d$. Its onset day is uniform in the week, so it counts as reported by $A'$ with probability $\frac{1}{7}\sum_{u=0}^{6} \mathbb{1}\{W - u + d \le A'\}$.
- **Imputed onsets.** Their report dates are not public. The hub imputed each onset by subtracting a delay drawn from the pooled delays. So a case imputed to week $W$ counts as reported by $A'$ with probability $\frac{1}{7}\sum_{u} \hat G(A' - W + u)$, with $\hat G$ the pooled delay cdf.
- **Delay data.** Rows whose report (taking the onset mid-week) falls by $A'$.

The 3-day lag gives the smallest net difference from the real 2026-08-18, 2026-10-09 and 2026-10-17 vintages over their last four weeks (see [Data](data.md#reconstructed-vintages)).
Reconstructed vintages miss real revisions, onsets added after first report and changes in the hub's imputation.
So they are cleaner than real vintages, and scores on them are somewhat optimistic.

### Simulated vintages

A simulator generates province outbreaks from the planned model's own structure with known parameters:

- a renewal process with a national Rt that turns from 1.6 to 0.85 around day 120, fixed province offsets, gravity-kernel importation ($\varepsilon = 0.01$) and susceptible depletion;
- ascertainment rising from 0.5 to 0.75;
- a missing-onset share rising from 5% to 25% and falling to 12%;
- a Gamma delay whose mean falls from 12 to 6 days.

It then publishes vintages as the hub does: known onsets by onset week, missing onsets given an onset by a delay drawn from the known delays reported so far, their total, and the delay line data.
The truth is every confirmed case by its true onset week.
Recovery checks run on these vintages.

### Which vintages are used where

| Run | Vintages | Truth |
|---|---|---|
| Development backtest | Real and reconstructed, reported separately; held-out reference dates excluded | Latest real vintage |
| Recovery check | Simulated | Simulated final counts |
| Held-out evaluation | Real only | Latest real vintage |

Reconstructed vintages whose reference date is on or after the first held-out date are never built.

## Planned model

The model is judged by its nowcast and forecast scores against persistence and simpler variants.
It should be a small extension of [PyRenew](https://github.com/CDCgov/PyRenew), pinned to a commit.
Its core components are required.
Their time variation uses pooling and smoothness priors, so each reduces to near constant when the data show no change and stays stable when data are sparse.
The backtest chooses the settings marked as such.

| Component | Status | PyRenew piece | Chosen by backtest |
|---|---|---|---|
| Renewal with pooled time-varying $R_p(t)$ | Core | `SubpopulationInfections`, `RandomWalk` | Smoothness prior, form of pooling |
| Susceptible depletion | Core, meant to keep forecasts sensible | Extend the latent process | Effective population prior; on or off |
| Spatial importation | Core | New latent class | Strength $\varepsilon$, kernel |
| Time-varying ascertainment $\alpha_p(t)$ | Required | `AscertainmentModel` with a temporal process | Smoothness prior |
| Time-varying delay $f_{p,s}$ | Required | New, with censoring and truncation | Family, pooling |
| Time-varying missing onsets $\pi_p(t)$ | Required | New | Smoothness prior |
| Known-onset observation, right truncated | Required | `SubpopulationCounts` with `right_truncation_rv` | Overdispersion prior |
| Hub imputation observation | Required | New, wraps the allocation below | None |

### Transmission

Symptom onsets $\lambda_p(t)$ in province $p$ on day $t$ follow a renewal process on onsets, with generation interval $g$ (Gamma, mean about 15 days, SD about 9):

$$
\lambda_p(t) = R_p(t)\, \frac{S_p(t)}{N_p} \sum_{s \ge 1} g(s) \Big[(1 - \varepsilon)\, \lambda_p(t - s) + \varepsilon \sum_{q \ne p} K_{qp}\, \lambda_q(t - s)\Big].
$$

- $\log R_p(t) = \log R(t) + \delta_p(t)$: a national weekly random walk plus province random walks with a sum-to-zero constraint and a partially pooled scale, as in PyRenew's `SubpopulationInfections`.
- $K$ is a fixed gravity kernel, $K_{qp} \propto N_p e^{-D_{qp}/\ell}$, with $\varepsilon \sim \text{Beta}(1, 100)$.
- Seeding: each province starts in its first week with cases, or by importation only.

A NumPy reference is in [`latent.renewal`][bdbv_forecasts.latent.renewal].

### Susceptible depletion

$S_p(t) = N_p - \sum_{u < t} \lambda_p(u) / \phi$, where $\phi$ is the symptomatic share of infections, so depletion counts all infections, not only confirmed cases.
With full province populations (1 to 14 million) and about 10,000 cases, depletion is negligible.
It only protects against wild forecasts if the effective population is much smaller than the resident population.
So $N_p = \kappa_p N_p^{\text{res}}$ with a prior on $\kappa_p$ (for example log-normal centred on 1% with wide spread), pooled across provinces.

Depletion interacts with ascertainment.
Confirmed cases are $\alpha_p(t)\lambda_p(t)$, so the depleted fraction is $\sum C_p / (\alpha_p \phi \kappa_p N_p^{\text{res}})$.
Only the product $\alpha_p \phi \kappa_p$ is informed by the curvature of growth, and weakly.
The prior on $\kappa_p$ carries the weight.
The backtest compares variants with and without depletion.

### Ascertainment

Confirmed onsets are $C_p(t) = \alpha_p(t)\, \lambda_p(t)$, with

$$
\operatorname{logit} \alpha_p(t) = \alpha_0 + a(t) + a_p(t),
$$

a national weekly random walk $a(t)$ and province deviations $a_p(t)$ with a pooled scale.
Without testing or serology data, the level $\alpha_0$ is not identified from case counts: it trades off against the level of $\lambda$ and against $N_p$.
It needs an informative prior.
Its changes over time are only weakly identified, against changes in $R_p(t)$.
The smoothness prior keeps them near zero unless the data demand otherwise.

### Delays

Reports by report day are $\rho_p(t) = \sum_s C_p(s) f_{p,s}(t - s)$.
The delay $f_{p,s}$ varies with onset day $s$ and province, with a flexible shape: a discretised log-normal, Gamma or Weibull, or a nonparametric pmf with a smoothness penalty.
Its parameters $\theta_{p,s}$ follow a weekly random walk with province deviations pooled towards the national value.
Each delay in the line data is double interval censored at daily resolution.
Only the onset week is published, so right truncation at $T_v$ averages over the onset day:

$$
\mathcal{L}_v = \prod_{\text{cases}} \frac{p_{\theta}(d)}{\frac{1}{7}\sum_{u=0}^{6} F_{\theta}(T_v - W + u)}.
$$

The delay data inform $f$ well in the large provinces and in weeks with enough cases.
Small provinces and the latest weeks borrow from the pooled value.
[`EmpiricalDelayModel`][bdbv_forecasts.delays.EmpiricalDelayModel] is the simple estimator in place now: a rolling window of onset weeks shrunk towards the pooled pmf.

### Missing onsets

Let $\pi_p(t)$ be the share of cases reported on day $t$ without an onset.
The hub publishes, at every vintage, known and imputed counts by province and onset week, so $\pi_p(t)$ is informed directly:

$$
I_{p,v}(w) \mid Y_{p,v}(w) \sim \text{BetaBinomial}\left(Y_{p,v}(w),\, \bar\pi_{p,v}(w),\, \nu\right),
$$

with $\bar\pi_{p,v}(w)$ the expected imputed share implied by the model below, $\operatorname{logit}\pi_p(t)$ a smooth weekly random walk with province deviations pooled across provinces.
The change in imputed totals between vintages approximates missing cases by report week and adds information on timing.
$\pi_p(t)$ is well identified, unlike ascertainment.

### Known onsets

Cases with onset $s$ reported by $T_v$ with an onset recorded:

$$
\mathbb{E}[K_{p,v}(s)] = \sum_{t \le T_v} C_p(s)\, f_{p,s}(t - s)\, (1 - \pi_p(t)),
$$

summed to weeks, with a negative binomial likelihood.
This handles right truncation through $f$.
The weekly reporting triangle across vintages checks it.

### Modelling the hub's imputation

The hub's imputed counts are a deterministic function of latent quantities, plus the noise of its draws.
It gives each case reported without an onset an onset by subtracting a delay drawn from "the full delay distribution as available from the data on the site", from a defined vintage ([hub issue 61](https://github.com/InsightNet-US/BDBV-Modeling-Hub/issues/61)).
We model that procedure, including its bias, rather than correcting it.

1. **Missing cases by report day** are latent: $M_p(t) = \pi_p(t)\, \rho_p(t)$, from our time-varying delay.
2. **The hub's delay** at vintage $v$ is $\hat g_v$, the empirical pmf of all published delays of that vintage, pooled over provinces and onset weeks. It is not corrected for truncation.
3. **Expected imputed onsets** on day $s$:
$$
\mathbb{E}[I_{p,v}(s)] = \sum_{t \le T_v} M_p(t)\, \hat g_v(t - s),
$$
summed to onset weeks, with a negative binomial likelihood. This is the expected back-sampling allocation (for example `backsampling_allocate` from EpidemicDates, or a few lines of JAX).

The forecast of the target $y_p(w)$ is the eventual known plus imputed onsets, with future imputation using the latest $\hat g_v$.
So the forecast carries the imputation's bias at turning points, which is what the hub scores.
A NumPy reference is in [`observation`][bdbv_forecasts.observation].

**Uncertain.**
The hub has not published its code.
We assume it pools over provinces and onset weeks, draws from the vintage being published, and drops nothing.
How it treats negative and very long delays, and whether it re-imputes earlier cases at each vintage, are not stated.
The non-monotone revisions in imputed counts suggest it re-imputes each time.

### What the data identify

| Quantity | Informed by | Identified? |
|---|---|---|
| $\pi_p(t)$ | Imputed and known counts per week and vintage | Well |
| $f_{p,s}$ | Delay line data; the reporting triangle | Well in large provinces and older weeks; pooled elsewhere |
| $\hat g_v$ | Published delays | Known |
| $R_p(t)$ times $\alpha_p(t)$ | Case counts | Well, as their product's growth |
| $\alpha_p(t)$ level | Nothing in the hub data | Needs a prior |
| $\alpha_p(t)$ change | Trades off against $R_p(t)$ | Weakly; smoothness prior |
| $\kappa_p N_p$ | Curvature of growth | Weakly; prior |
| $\varepsilon$ | Timing of spread between provinces | Weakly; Beta prior |

### Sparse settings

Provinces with a handful of cases borrow strength through the pooled scales of $\delta_p$, $a_p$, the logit $\pi$ deviations and the delay deviations.
Recent weeks lean on the random walk priors and on the pooled delay.
Provinces with no cases are not forecast.

### What the backtest decides

Each candidate is a registered model variant.
The development backtest compares them by relative WIS and log WIS against persistence, by horizon and province, on real and reconstructed vintages.
It chooses the strength of spatial coupling, the form of pooling, the smoothness priors, the delay family, the depletion settings and the variant to submit.
The choice is written to `results/model-choice.yml` and committed before the [held-out evaluation](backtest.md#held-out-evaluation) is run once.
