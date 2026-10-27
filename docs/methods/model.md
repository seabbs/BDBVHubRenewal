# Model

The renewal model and the persistence baseline, with the maths of each.
{ .page-subtitle }

The data and notation are on the [Data methods page](data.md#target-and-notation).
The renewal model submits to the hub as `epiforecasts-hubrenewal`.
`epiforecasts-renewal` is a different hub model, from [BVDOutbreakSize](https://github.com/epiforecasts/BVDOutbreakSize).

## Structure

![The province renewal model as a generative chain, from the reproduction number to the hub's data streams.](../assets/diagrams/model_dag.svg)

The figure shows the main variant, `hubrenewal`.
Grey boxes are parameters, blue boxes latent quantities and orange boxes the data with their likelihoods.
The hub's pooled delay $\hat f$ is taken from the data of each vintage, not estimated.
The source is `docs/diagrams/model_dag.tex`, built by `scripts/build_diagrams.sh`.

## Components

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

## Transmission

Symptom onsets $\lambda_p(t)$ in province $p$ on day $t$ follow a renewal process on onsets, with generation interval $g$ (Gamma, mean about 15 days, SD about 9):

$$
\lambda_p(t) = R_p(t)\, \frac{S_p(t)}{N_p} \sum_{s \ge 1} g(s) \Big[(1 - \varepsilon)\, \lambda_p(t - s) + \varepsilon \sum_{q \ne p} K_{qp}\, \lambda_q(t - s)\Big].
$$

- $\log R_p(t) = \log R(t) + \delta_p(t)$: a national weekly random walk plus province random walks with a sum-to-zero constraint and a partially pooled scale, as in PyRenew's `SubpopulationInfections`.
- $K$ is a fixed gravity kernel, $K_{qp} \propto N_p e^{-D_{qp}/\ell}$, with $\varepsilon \sim \text{Beta}(1, 100)$.
- Seeding: each province starts in its first week with cases, or by importation only.

A NumPy reference is in [`latent.renewal`][bdbv_hub_renewal.latent.renewal].

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

## Delays

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
[`EmpiricalDelayModel`][bdbv_hub_renewal.delays.EmpiricalDelayModel] is the simple estimator in place now: a rolling window of onset weeks shrunk towards the pooled pmf.

## Missingness and imputation

### Missing onsets

Let $\pi_p(t)$ be the share of cases reported on day $t$ without an onset.
The hub publishes, at every vintage, known and imputed counts by province and onset week, so $\pi_p(t)$ is informed directly:

$$
I_{p,v}(w) \mid Y_{p,v}(w) \sim \text{BetaBinomial}\left(Y_{p,v}(w),\, \bar\pi_{p,v}(w),\, \nu\right),
$$

with $\bar\pi_{p,v}(w)$ the expected imputed share implied by the model below, $\operatorname{logit}\pi_p(t)$ a smooth weekly random walk with province deviations pooled across provinces.
The change in imputed totals between vintages approximates missing cases by report week and adds information on timing.
$\pi_p(t)$ is well identified, unlike ascertainment.

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
A NumPy reference is in [`observation`][bdbv_hub_renewal.observation].

**Uncertain.**
The hub has not published its code.
We assume it pools over provinces and onset weeks, draws from the vintage being published, and drops nothing.
How it treats negative and very long delays, and whether it re-imputes earlier cases at each vintage, are not stated.
The non-monotone revisions in imputed counts suggest it re-imputes each time.

## Observation

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

### Known onsets

Cases with onset $s$ reported by $T_v$ with an onset recorded:

$$
\mathbb{E}[K_{p,v}(s)] = \sum_{t \le T_v} C_p(s)\, f_{p,s}(t - s)\, (1 - \pi_p(t)),
$$

summed to weeks, with a negative binomial likelihood.
This handles right truncation through $f$.
The weekly reporting triangle across vintages checks it.

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

## Variants

### What the backtest decides

Each candidate is a registered model variant.
The development backtest compares them by relative WIS and log WIS against persistence, by horizon and province, on real and reconstructed vintages.
It chooses the strength of spatial coupling, the form of pooling, the smoothness priors, the delay family, the depletion settings and the variant to submit.
The choice is logged with its date in `results/model-choice.yml`.
Every later vintage is then held out (see [Evaluation](evaluation.md#rolling-held-out-evaluation)).

### Renewal variants

The renewal variants run in the backtest, the held-out evaluation and the plots, next to persistence.

| Variant | What it is |
|---|---|
| `hubrenewal` | The model in the figure: provinces with pooled $R_p(t)$, gravity importation and depletion. |
| `hubrenewalsp` | Single patch: one national renewal with the same delays and imputation, split across provinces as below. |
| `hubrenewalbvd` | The spatial model informed by the latest posterior estimates of the main [BVDOutbreakSize](https://github.com/epiforecasts/BVDOutbreakSize) model, taken from the newest release published before the data cut-off. |
| `fallback` | The CI's fallback: delays and $\pi$ fixed from the data, no coupling, weekly $R(t)$. It is a fit configuration, not a backtest variant. |

The model code is in `bdbv_hub_renewal.pyrenew` and the variants in [`models.renewal`][bdbv_hub_renewal.models.renewal].

### Single patch

`hubrenewalsp` fits one renewal process to the national counts.
Each national trajectory $\tilde y(w)$ is split across the provinces $p$ of the task by a multinomial draw:

$$
\big(\tilde y_p(w)\big)_p \sim \text{Multinomial}\big(\tilde y(w),\, \theta\big),
\qquad
\theta_p = \frac{c_p + a}{\sum_q (c_q + a)},
$$

with $c_p$ the province's onsets over the last 4 data weeks and $a = 1/2$, the Jeffreys pseudo-count.
So a province with no recent onsets keeps a small share, and the provinces sum to the national count in every sample and week.
The single patch has no province likelihood, so it does not take part in the leave-one-province-out check.

### Persistence baseline

`persistence` is the baseline every other model is compared against.
Let $c$ be the last week judged complete.
Week $w$ is complete if its midpoint (its Wednesday) is at least $L$ days before $T_v$.
$L$ is the 90th centile of the delays published with the vintage, or 10 days if none were published.
For the current vintages $L$ is 14 days, so $c$ is two weeks before the vintage's epi week, about horizon $-3$.

Weeks up to $c$ are copied from the vintage, with no uncertainty:

$$
\tilde y_p(w) = Y_{p,v}(w), \qquad w \le c.
$$

Later weeks carry $Y_{p,v}(c)$ forward as a random walk on the log scale, with Poisson noise.
For $w = c + h$ with $h \ge 1$,

$$
\log(1 + \lambda_{p}(c + h)) = \max\Big(0,\ \log(1 + Y_{p,v}(c)) + \sum_{j=1}^{h} \epsilon_{p,j}\Big),
\qquad \epsilon_{p,j} \sim \mathcal{N}(0, \sigma_p^2),
$$

$$
\tilde y_p(c + h) \sim \text{Poisson}(\lambda_p(c + h)).
$$

$\sigma_p$ is the sample standard deviation of the 7 weekly changes in $\log(1 + Y_{p,v})$ over the last 8 complete weeks, floored at 0.1.
Each sample is one joint trajectory across weeks.
The national value is the sum of the province samples, sample by sample.

Two things follow.
The median of $\lambda_p$ stays at $Y_{p,v}(c)$, so it has no growth.
And weeks up to $c$ are point masses at counts that are still revised upwards after the vintage.
So persistence underpredicts while the epidemic grows and recent weeks fill in (see [Backtest](../backtest.md#persistence-underpredicts)).
