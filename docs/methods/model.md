# Model

The renewal model and the persistence baseline, with the maths of each.
{ .page-subtitle }

The data and notation are on the [Data methods page](data.md#target-and-notation).
How the model is built in code and sampled is on the [Implementation](implementation.md) page.
Submissions go to the hub as `epiforecasts-seabbsrenewal`, whichever model produced them.
`epiforecasts-renewal` is a different hub model, from [BVDOutbreakSize](https://github.com/epiforecasts/BVDOutbreakSize).

## Structure

![The province renewal model as a generative chain, from the reproduction number to the hub's data streams.](../assets/diagrams/model_dag.svg)

The figure shows the main variant, `renewal`.
Grey boxes are parameters, blue boxes latent quantities and orange boxes the data with their likelihoods.
The hub's pooled delay $\hat f$ is taken from the data of each vintage, not estimated.
The dashed box is the BVDOutbreakSize term, in `renewalbvd` only.
Every time-varying part has a smoothness prior, so it stays near constant when the data show no change and stays stable when data are sparse.

## Transmission

Days $s$ and $t$ lie on one daily axis, with epi weeks from Sunday to Saturday.
Provinces $p$ with cases in the vintage have a full renewal process.
Provinces $q$ with none have the importation-only component below.

### Reproduction number

$$
\log R_p(t) = \log R(t) + \delta_p(t).
$$

The pooled trend $\log R(t)$ is the unweighted mean of $\log R_p(t)$ over the provinces with cases.
It is not the national Rt, which would weight provinces by their infections.
It is a weekly random walk, sampled as its weekly values (knots) $k_j$:

$$
k_0 \sim \mathcal{N}(0, 0.5^2), \qquad k_j - k_{j-1} \sim \mathcal{N}(0, \sigma_R^2), \qquad \sigma_R \sim \mathcal{N}^+(0, 0.2^2).
$$

Sampling the knots rather than a level and its steps removes the direction in which the level and every step move together.
The prior is the same.

The province deviations sum to zero.
They are $\delta_p(t) = \sum_j Q_{pj}\, \eta_j(t)$, with $Q$ an orthonormal basis of the sum-to-zero subspace and $P - 1$ weekly AR(1) processes $\eta_j$, which revert to their starting offsets:

$$
\eta_j(t) = \phi\, \eta_j(t - 1) + \sigma_\delta z_j(t), \qquad \phi = 2^{-1/h}, \qquad h \sim \text{LogNormal}(\log 6, 0.6^2)\ \text{weeks}, \qquad \sigma_\delta \sim \mathcal{N}^+(0, 0.1^2).
$$

The shared $\sigma_\delta$ and half-life $h$ pool the provinces.
BVDOutbreakSize uses the same form, with a 42-day half-life.
The starting offsets are $v = Q u$, with $u \sim \mathcal{N}(0, \sigma_v^2)$ sampled centred and $\sigma_v \sim \mathcal{N}^+(0, 0.3^2)$.
All walks are flat over the 6 seeding weeks before the data start, and are fitted on the data weeks only.
Their steps past the data are drawn in the forecast, from the fitted step SDs.
As an option, Rt instead holds at its last fitted value, as in EpiNow2.
Another option joins the weekly knots linearly rather than holding them for the week.

### Renewal with importation and depletion

The renewal runs on infections per resident, dated by onset:

$$
y_p(t) = R_p(t) \sum_q W_{pq} \sum_{\tau \ge 1} x_q(t - \tau)\, g(\tau),
\qquad
x_p(t) = S_p(t)\big\{1 - e^{-y_p(t)/\nu}\big\},
\qquad
S_p(t + 1) = S_p(t)\, e^{-y_p(t)/\nu}.
$$

- $g$ is a Gamma generation interval with mean 15.3 and SD 9.3 days, BVDOutbreakSize's prior means, double interval censored on lags 1 to 42 (about its 98.5% quantile). After censoring, dropping lag 0 and truncation its mean is 14.8 and SD 8.4 days. Each infection day is uniform within its day and the pmf is $G(d + 1) - 2G(d) + G(d - 1)$, with $G$ the Gamma's integrated cdf, as in the R package primarycensored. Lag 0 is dropped and the rest renormalised.
- $W = (1 - \varepsilon) I + \varepsilon\, \mathrm{diag}(1/N) K^\top \mathrm{diag}(N)$ mixes transmission between provinces, with $\varepsilon \sim \text{Beta}(1, 100)$.
- $K$ is a row-normalised gravity kernel, $K_{qp} \propto N_p e^{-D_{qp}/300\,\text{km}}$, with $D$ the great-circle distance between population-weighted province centres and $N_p$ the province population.
- As an option, the kernel's decay and destination weights are estimated, $\log K_{qp} = \log N_p - D_{qp}/(300\,\text{km} \cdot e^{\kappa_K}) + w_p + c_q$, with $\kappa_K \sim \mathcal{N}(0, 1)$, sum-to-zero weights $w$ with scale $\mathcal{N}^+(0, 0.5^2)$, and $c_q$ normalising each row. With $\kappa_K = 0$ and $w = 0$ it is the fixed gravity kernel.
- As an option, each origin has its own share $\varepsilon_q$, partially pooled on the logit scale.
- $\nu \sim \text{LogNormal}(\log 0.05, 0.75^2)$ is the share of each province at risk.
  $S_p$ starts at $\nu$, so cumulative infections never exceed $\nu N_p$.

The exponential form agrees with linear depletion, $x = S y$, to first order, and cannot overshoot the pool.

Seeding is set on the scale of confirmed onsets.
Each province's expected confirmed onsets per day are $c_p \sim \text{LogNormal}(0, 1.5^2)$, which become $I_{0,p} = c_p / (\alpha_0 N_p)$ infections per resident.
Provinces with cases in the first data week are seeded at the start.
Their seed is projected back over the seeding weeks at the growth rate implied by $R_p$ and $g$, clipped to $[-0.05, 0.3]$ per day, so a steep early decline cannot project a seed larger than the pool.
Later provinces start at $10^{-6}$ of that level and take $I_{0,p}$ as a constant daily addition to $y_p(t)$ over their first case week.
Importation adds to it.

The simulator ([simulated vintages](data.md#simulated-vintages)) uses this renewal too.

## Observation

### Ascertainment

Expected confirmed onsets are

$$
\lambda_p(s) = \alpha(s)\, N_p\, x_p(s),
\qquad
\operatorname{logit} \alpha(s) = \operatorname{logit} \alpha_0 + a(s),
$$

with $\alpha_0 = 0.6$ fixed and $a$ a national weekly random walk from zero with step SD $\mathcal{N}^+(0, 0.03^2)$.
The level is fixed because the data only identify ascertainment together with $\nu$ and the seeds.

### Delays

The onset-to-report delay $f_s(d)$, on days $d$ from 0 to 60, comes from a discrete hazard:

$$
\operatorname{logit} h_s(d) = \beta(d) + \gamma(s),
\qquad
f_s(d) = h_s(d) \prod_{j < d}\{1 - h_s(j)\},
\qquad h_s(60) = 1.
$$

- $\beta$ is a random walk over delay with $\beta(0) \sim \mathcal{N}(-2, 1)$ and step SD $\mathcal{N}^+(0, 0.4^2)$.
  This makes the shape smooth but not parametric.
  The fit to the 2026-10-24 vintage puts the step SD at 0.20 (90% CrI 0.16 to 0.25), in the tail of the earlier $\mathcal{N}^+(0, 0.1^2)$.
- $\gamma$ is a weekly random walk over onset day from zero, with step SD $\mathcal{N}^+(0, 0.15^2)$.
  It shifts all hazards together.
- As an option, off by default, the logit hazard also has a sum-to-zero effect of the report day's day of week, $\delta_{(s + d) \bmod 7}$, with scale $\mathcal{N}^+(0, 0.5^2)$.

The delay is pooled over provinces.

### Missing onsets

Cases reported on day $t$ without an onset are a share $\pi_p(t)$ of reports, with

$$
\operatorname{logit} \pi_p(t) = b(t) + \eta_p(t),
$$

built as $\log R_p$ is: a pooled weekly random walk $b$ plus sum-to-zero province deviations $\eta_p$.
The starting national value is $\mathcal{N}(-2, 1)$ with province spread $\mathcal{N}^+(0, 0.5^2)$.
The step SDs are $\mathcal{N}^+(0, 0.3^2)$ for $b$ and $\mathcal{N}^+(0, 0.1^2)$ for $\eta$, sampled non-centred.
$\pi_p$ is informed by the ratio of imputed to known onsets, through the imputation below.

### Known onsets

A case reported on day $t$ is in the vintage published at $T$ with probability $c(t) = F_\ell(T - t)$, with $F_\ell$ the cdf of the lag $\ell$ from report to publication.
The lag pmf is estimated from the data on 0 to 180 days.
Rows in a delay file beyond those of the previous file were first published between the two dates.
Each such row's lag is spread over its onset day within the week and its publication day between the files, over feasible lags only.
Between the real files 8-9% of new rows were more than 60 days late and none more than 180.
Only pairs of files dated on or before the forecast date are used, so backtests do not see later files.
Between 2026-10-17 and 2026-10-24, 23% of new rows had report dates more than 41 days old, so late entries matter for the nowcast weeks.
A vintage with no earlier pair (2026-10-09 and before) uses a fixed prior pmf instead: 85% uniform on 0 to 3 days and 15% uniform on 4 to 60 days.
That prior is shaped by the later files (29% of rows new between 10-09 and 10-17 were more than 6 days late), so the 2026-10-09 backtest uses a little later information.
Without a tail, the 2026-10-09 fit treated weeks 4 to 5 weeks old as complete and its national 90% intervals fell below the counts reported by 10-24 for three of the four nowcast weeks.
The cases with onset $s$ in the vintage with an onset recorded are expected to be

$$
\mathrm{known}_{p,T}(s) = \lambda_p(s) \sum_{d} f_s(d)\, \{1 - \pi_p(s + d)\}\, c(s + d).
$$

This is right truncation by vintage.

### Hub imputation

The hub's imputed counts are a deterministic function of latent quantities, plus the noise of its draws.
It gives each case reported without an onset an onset by subtracting a delay drawn from "the full delay distribution as available from the data on the site", from a defined vintage ([hub issue 61](https://github.com/InsightNet-US/BDBV-Modeling-Hub/issues/61)).
We model that procedure, including its bias, rather than correcting it.

$$
M_p(t) = \pi_p(t) \sum_s \lambda_p(s)\, f_s(t - s),
\qquad
\mathrm{imputed}_{p,T}(s) = \sum_{t = s}^{T} \hat f_T(t - s)\, c(t)\, M_p(t).
$$

$M_p(t)$ are the cases reported on day $t$ without an onset.
$\hat f_T$ is the hub's delay at vintage $T$.
It is the empirical pmf of all published delays of that vintage, pooled over provinces and onset weeks, with negative delays and delays over 60 days dropped.
It is data, not a parameter, and it is not corrected for truncation.

The forecast target is the eventual known plus imputed onsets, with future imputation using the latest $\hat f_T$.
So the forecast carries the imputation's bias at turning points, which is what the hub scores.

**Uncertain.**
The hub has not published its code.
We assume it pools over provinces and onset weeks, draws from the vintage being published, and drops nothing.
How it treats negative and very long delays, and whether it re-imputes earlier cases at each vintage, are not stated.
The non-monotone revisions in imputed counts suggest it re-imputes each time.

### Likelihood

- Weekly sums of $\mathrm{known}_{p,T}$ and $\mathrm{imputed}_{p,T}$ are negative binomial, each stream with its own dispersion, $1/\sqrt{\phi} \sim \text{HalfNormal}(0.25)$, as in EpiNow2.
- The delays of known-onset cases in the hub's delay file are tabulated by onset week and delay, pooled over provinces.
  Delays are clipped to 0 to 60 days.
  Given each week's total they are Dirichlet-multinomial, with mean probabilities from $\lambda$, $f$, $\pi$ and $c$, $1/\sqrt{\psi} \sim \text{HalfNormal}(0.1)$ and $\psi$ capped at $10^4$.
  A multinomial treats the 8,000 or so pooled delays as independent, which makes the delay overconfident.
  The multinomial is available as an option.
  This is the conditional part of the reporting triangle, so the known stream is not counted twice.
- Provinces with no cases have expected onsets $\mu_q(s) = \kappa \sum_p K_{pq}\, \lambda_p(s)$, with one scale $\kappa \sim \text{LogNormal}(\log 10^{-3}, 2^2)$.
  Their weekly zeros are Poisson.
- Leaving a province out switches off its terms in the first two.
  It stays in the latent process, so it still transmits and is still forecast.

### Forecasts

The model axis runs to the end of the forecast horizon, so the random walks carry $R_p$, $\alpha$, $f$ and $\pi$ forward.
Each of the 100 trajectories uses one posterior draw for every province and week.
It is the final known plus final imputed counts.
For known onsets, this is the count already reported by the cut-off plus a negative binomial draw of the expected cases reported after it, as in epinowcast.
So a trajectory is never below the known onsets already reported.
The hub re-imputes missing onsets at each vintage, so a week's imputed count can fall as well as rise.
The final imputed count is a negative binomial draw centred on the current imputed count plus the count expected from cases reported later.
In recovery fits, keeping the current imputed count fixed instead under-covered the final counts.
A province left out uses a draw of its vintage count instead of the data.
Provinces with no cases are zero up to the cut-off.
After it they draw Poisson importations, $\mu_q = \kappa \sum_p K_{pq} \lambda_p$, which spread locally at the national Rt, $\sum_p R_p I_p / \sum_p I_p$, weighted by infections.
For `renewal` and `renewalbvd`, the national trajectory is the sum of the province trajectories plus an unassigned component.
The unassigned component holds any provinces the model does not report, and the national cases in the data that no province has.
In weeks with data, that gap is the vintage's $g(w) = \max\{0, Y_{CD}(w) - \sum_p Y_p(w)\}$ in every trajectory.
In later weeks it is $g_k(w) = \rho \sum_p \tilde y_{p,k}(w)$, with $\rho$ the gap over the provinces' total in the last 4 data weeks and $\tilde y_{p,k}$ trajectory $k$ of province $p$.

### Identifiability

| Quantity | Informed by | Identified? |
|---|---|---|
| $R_p(t)$ | Growth of onsets in each province | Well where cases are many; pooled where few |
| $\pi_p(t)$ | Imputed against known onsets | Well in recent weeks; prior-driven in early weeks with few cases |
| $f_s(d)$ | The delay file and the shape of right truncation | Well |
| $\hat f_T$ | Published delays | Known |
| $\alpha(s)$ level | Only with $\nu$ and the seeds | Fixed at $\alpha_0$ |
| $\alpha(s)$ change | Trades off against $R_p(t)$ | Weakly; its small step SD keeps it near constant |
| $\nu$ | Depletion only | Mostly prior; it bounds forecasts rather than explaining the data |
| $\varepsilon$ | Timing of spread between provinces, against the seeds | Weakly; Beta prior |
| $\kappa$ | Zeros in provinces with no cases | Bounded above only; it does not affect provinces with cases |

### Sparse data

Provinces with a handful of cases borrow strength through the pooled walks of $\delta_p$ and $\eta_p$, and share the delay and ascertainment.
Recent weeks lean on the random walk priors.
Provinces with no cases get the importation-only component, so their forecasts allow for introductions.

## Variants

### Model choice

The variant to submit is chosen on the development backtest, by WIS and log WIS relative to persistence, by horizon and province.
Every later vintage is held out (see [Evaluation](evaluation.md#rolling-held-out-evaluation)).
Each week the submission takes the first of `renewal`, one refit of `renewal` with smaller steps, `renewalsp` and `persistence` that passes its checks.

### Renewal variants

| Variant | What differs from the main model |
|---|---|
| `renewal` | Nothing. It is the model in the figure. |
| `renewalsp` | Each location fitted on its own, with no pooling, coupling or depletion. |
| `renewalbvd` | A tempered pseudo-likelihood on BVDOutbreakSize's latest Rt. |

### Single patch

`renewalsp` fits each location $\ell$, the nation included, to its own known and imputed onsets.
A location with at least 10 cases has

$$
x_\ell(t) = R_\ell(t) \sum_{\tau \ge 1} x_\ell(t - \tau)\, g(\tau),
\qquad
\log R_\ell(t) = k_{\ell, j(t)},
\qquad
\operatorname{logit} \pi_\ell(t) = b_\ell(t),
$$

with $k_\ell$ the weekly knot walk and $j(t)$ the week of day $t$.
This is the main model with one province, so $\delta = \eta = 0$, $\varepsilon = 0$ and $x = y$ with no depletion.
Ascertainment $\alpha_\ell$, the delay $f_{\ell, s}$, the walk $b_\ell$ and the likelihood are as in the main model, for location $\ell$ alone.
A location with fewer cases, $y_w$ in each of its $n_w$ data weeks, has

$$
\lambda_\ell \sim \text{Gamma}\Big(\tfrac{1}{2} + \sum_w y_w,\ n_w\Big),
\qquad
\tilde y_\ell(w) \sim \text{Poisson}(\lambda_\ell),
$$

with one $\lambda_\ell$ per trajectory.
The national forecast is the national fit, so it need not equal the sum of the provinces.

### BVDOutbreakSize-informed

`renewalbvd` adds the Rt estimates of the newest [BVDOutbreakSize](https://github.com/epiforecasts/BVDOutbreakSize) release published by the forecast's cut-off whose data end on or before the vintage.
With $a_b$ the release's data cut-off, the estimates are aligned to onset day $s_b = a_b + 6$, as BVDOutbreakSize dates Rt by infection.
For the nation $N$ and each BVDOutbreakSize patch $k$, the model's Rt is weighted by infections over the week of $s_b$,

$$
R_G(s_b) = \frac{\sum_{p \in G} O_p\, R_p(s_b)}{\sum_{p \in G} O_p},
\qquad
O_p = N_p \sum_{t \in \text{week}(s_b)} x_p(t),
\qquad
G \in \{N, k\}.
$$

With tempering $\alpha = 0.5$, the likelihood gains

$$
\mu_N \sim \mathcal{N}\big(\log R_N(s_b),\, \sigma_N^2/\alpha\big),
\qquad
m_k - m_N \sim \mathcal{N}\big(\log R_k(s_b) - \log R_N(s_b),\, \tau_k^2/\alpha\big),
\qquad
\tau_k^2 = \max(\sigma_k^2 - \sigma_N'^2,\, 0.05^2).
$$

$\mu_N$ and $\sigma_N$ are the mean and SD of the release's national log Rt draws.
$m_k$ and $\sigma_k$ are each patch's log median and SD, and $m_N$ and $\sigma_N'$ the national row of the same table.
The term adds no parameters, and $\alpha = 0$ gives `renewal`.
It is tempered because both models fit the same confirmed cases.
Releases before 23 October give the national term only, and the earliest give a growth rate $r$, converted by $R = (1 + r\theta)^k$ with BVDOutbreakSize's Gamma generation interval.

### Persistence baseline

`persistence` is the baseline every other model is compared against.
With $c$ the last complete week, weeks up to $c$ are the vintage counts and later weeks a log-scale random walk from $Y_{p,v}(c)$ with Poisson noise.
For $h \ge 1$,

$$
\tilde y_p(w) = Y_{p,v}(w), \quad w \le c,
\qquad
\log\{1 + \lambda_p(c + h)\} = \max\Big(0,\ \log\{1 + Y_{p,v}(c)\} + \sum_{j=1}^{h} \epsilon_{p,j}\Big),
\qquad
\epsilon_{p,j} \sim \mathcal{N}(0, \sigma_p^2),
$$

$$
\tilde y_p(c + h) \sim \text{Poisson}\big(\lambda_p(c + h)\big).
$$

$\sigma_p$ is the SD of the 7 weekly changes in $\log(1 + Y_{p,v})$ over the last 8 complete weeks, floored at 0.1.
Week $w$ is complete if its Wednesday is at least $L$ days before $T_v$, with $L$ the 90th centile of the delays published with the vintage, or 10 days if none were published.
Published delays are right truncated, so the comparator `persistence_fu` takes $L$ from onset weeks that ended at least 63 days before the vintage.
Each sample is one joint trajectory across weeks, and the national value is the sum of the provinces.
The median has no growth and weeks up to $c$ are still revised upwards, so persistence underpredicts during growth (see [Backtest](../backtest.md#persistence-bias)).
