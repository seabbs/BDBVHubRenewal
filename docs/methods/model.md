# Renewal models

The renewal models, with the maths of each.
{ .page-subtitle }

The data and notation are on the [Data methods page](data.md#target-and-notation).
How the models are built in code and sampled is on the [Implementation](implementation.md) page.

## Structure

![The province renewal model as a generative chain, from the reproduction number to the hub's data streams.](../assets/diagrams/model_dag.svg)

The figure shows `renewal`.
Grey boxes are parameters, blue latent quantities and orange data with their likelihoods.
The dashed box is in `renewalbvd` only.
Every time-varying part is a random walk or AR(1), so it stays near constant unless the data show change.

## Transmission

Provinces $p$ with cases have a renewal process, and provinces $q$ without cases an importation-only component.

### Reproduction number

$$
\log R_p(t) = \log R(t) + \delta_p(t).
$$

The pooled trend $\log R(t)$ is the unweighted mean of $\log R_p(t)$, not the national Rt.
It is a weekly random walk on knots $k_j$,

$$
k_0 \sim \mathcal{N}(0, 0.5^2), \qquad k_j - k_{j-1} \sim \mathcal{N}(0, \sigma_R^2), \qquad \sigma_R \sim \mathcal{N}^+(0, 0.2^2).
$$

The province deviations sum to zero.
They are $\delta_p(t) = \sum_j Q_{pj}\, \eta_j(t)$, with $Q$ an orthonormal basis of the sum-to-zero subspace and $P - 1$ weekly AR(1) processes $\eta_j$, which revert to their starting offsets:

$$
\eta_j(t) = \phi\, \eta_j(t - 1) + \sigma_\delta z_j(t), \qquad \phi = 2^{-1/h}, \qquad h \sim \text{LogNormal}(\log 6, 0.6^2)\ \text{weeks}, \qquad \sigma_\delta \sim \mathcal{N}^+(0, 0.1^2).
$$

The shared $\sigma_\delta$ and $h$ pool the provinces, as in BVDOutbreakSize with a 42-day half-life.
The starting offsets are $v = Q u$, with $u \sim \mathcal{N}(0, \sigma_v^2)$ and $\sigma_v \sim \mathcal{N}^+(0, 0.3^2)$.
All walks are flat over the 6 seeding weeks and continue past the data with their fitted step SDs.
Options hold Rt at its last fitted value past the data, or join the knots linearly.

### Renewal with importation and depletion

The renewal runs on infections per resident, dated by onset:

$$
y_p(t) = R_p(t) \sum_q W_{pq} \sum_{\tau \ge 1} x_q(t - \tau)\, g(\tau),
\qquad
x_p(t) = S_p(t)\big\{1 - e^{-y_p(t)/\nu}\big\},
\qquad
S_p(t + 1) = S_p(t)\, e^{-y_p(t)/\nu}.
$$

- $g$ is BVDOutbreakSize's Gamma generation interval (mean 15.3, SD 9.3 days), double interval censored as $G(d + 1) - 2G(d) + G(d - 1)$ with $G$ the integrated cdf, on lags 1 to 42 and renormalised (mean 14.8, SD 8.4 days).
- $W = (1 - \varepsilon) I + \varepsilon\, \mathrm{diag}(1/N) K^\top \mathrm{diag}(N)$ mixes transmission between provinces, with $\varepsilon \sim \text{Beta}(1, 100)$.
- $K$ is a row-normalised gravity kernel, $K_{qp} \propto N_p e^{-D_{qp}/300\,\text{km}}$, with $D$ the great-circle distance between population-weighted province centres and $N_p$ the province population.
- An option estimates the kernel, $\log K_{qp} = \log N_p - D_{qp}/(300\,\text{km} \cdot e^{\kappa_K}) + w_p + c_q$, with $\kappa_K \sim \mathcal{N}(0, 1)$, sum-to-zero $w$ with scale $\mathcal{N}^+(0, 0.5^2)$ and $c_q$ normalising rows.
- An option gives each origin its own $\varepsilon_q$, partially pooled on the logit scale.
- $\nu \sim \text{LogNormal}(\log 0.05, 0.75^2)$ is the share at risk, and $S_p$ starts at $\nu$, so infections never exceed $\nu N_p$.

To first order this is linear depletion, $x = S y$, and it cannot overshoot the pool.

Seeds are expected confirmed onsets per day, $c_p \sim \text{LogNormal}(0, 1.5^2)$, or $I_{0,p} = c_p / (\alpha_0 N_p)$ infections per resident.
Provinces with cases in the first data week are seeded at the start, projected back at the growth rate implied by $R_p$ and $g$, clipped to $[-0.05, 0.3]$ per day.
Later provinces start at $10^{-6}$ of that level and add $I_{0,p}$ to $y_p(t)$ each day of their first case week.

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

- $\beta$ is a random walk over delay, $\beta(0) \sim \mathcal{N}(-2, 1)$ with step SD $\mathcal{N}^+(0, 0.4^2)$, so the shape is smooth but not parametric.
- $\gamma$ is a weekly random walk over onset day from zero, with step SD $\mathcal{N}^+(0, 0.15^2)$, shifting all hazards together.
- An option adds a sum-to-zero report day-of-week effect $\delta_{(s + d) \bmod 7}$ with scale $\mathcal{N}^+(0, 0.5^2)$.

The delay is pooled over provinces.

### Missing onsets

Cases reported on day $t$ without an onset are a share $\pi_p(t)$ of reports, with

$$
\operatorname{logit} \pi_p(t) = b(t) + \eta_p(t),
$$

a pooled weekly random walk $b$ plus sum-to-zero province deviations $\eta_p$, as for $\log R_p$.
The start is $\mathcal{N}(-2, 1)$ with province spread $\mathcal{N}^+(0, 0.5^2)$, and the step SDs are $\mathcal{N}^+(0, 0.3^2)$ for $b$ and $\mathcal{N}^+(0, 0.1^2)$ for $\eta$.

### Known onsets

A case reported on day $t$ is in the vintage published at $T$ with probability $c(t) = F_\ell(T - t)$, with $F_\ell$ the cdf of the report-to-publication lag.
The lag is uniform on 0 to 3 days.
Expected known onsets, right truncated by vintage, are

$$
\mathrm{known}_{p,T}(s) = \lambda_p(s) \sum_{d} f_s(d)\, \{1 - \pi_p(s + d)\}\, c(s + d).
$$

### Hub imputation

The hub gives each case reported without an onset one by subtracting a delay drawn from its published delays ([hub issue 61](https://github.com/InsightNet-US/BDBV-Modeling-Hub/issues/61)).
We model that procedure, bias included:

$$
M_p(t) = \pi_p(t) \sum_s \lambda_p(s)\, f_s(t - s),
\qquad
\mathrm{imputed}_{p,T}(s) = \sum_{t = s}^{T} \hat f_T(t - s)\, c(t)\, M_p(t).
$$

$M_p(t)$ are the cases reported on day $t$ without an onset.
$\hat f_T$ is the empirical pmf of all delays published at $T$, pooled and on 0 to 60 days, and is data.
Future imputation uses the latest $\hat f_T$, so forecasts carry the imputation's bias, which is what the hub scores.
The hub's code is not public, so pooling and the handling of negative and long delays are assumed.

### Likelihood

- Weekly $\mathrm{known}_{p,T}$ and $\mathrm{imputed}_{p,T}$ are negative binomial, each with $1/\sqrt{\phi} \sim \text{HalfNormal}(0.25)$.
- Known-onset delays, by onset week and delay on 0 to 60 days, are Dirichlet-multinomial given each week's total, with probabilities from $\lambda$, $f$, $\pi$ and $c$ and $1/\sqrt{\psi} \sim \text{HalfNormal}(0.1)$. A multinomial, an option, would treat the 8,000 pooled delays as independent.
- Provinces with no cases have Poisson weekly zeros with mean $\mu_q(s) = \kappa \sum_p K_{pq}\, \lambda_p(s)$, $\kappa \sim \text{LogNormal}(\log 10^{-3}, 2^2)$.
- Leaving a province out removes its first two terms but keeps it in the latent process.

### Forecasts

The random walks carry $R_p$, $\alpha$, $f$ and $\pi$ to the forecast horizon, and each of 100 trajectories is one posterior draw.
A trajectory is the final known plus final imputed counts.
Final known counts are those reported by the cut-off plus a negative binomial draw of those reported later, as in epinowcast.
Final imputed counts are a negative binomial draw centred on the current imputed count plus those expected later, since the hub re-imputes each vintage.
Provinces with no cases draw Poisson importations after the cut-off, which spread at the infection-weighted national Rt.
The national trajectory is the sum of the provinces plus an unassigned component, $\max\{0, Y_{CD}(w) - \sum_p Y_p(w)\}$ in data weeks and $\rho \sum_p \tilde y_{p,k}(w)$ after, with $\rho$ its share over the last 4 data weeks.

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

## Health zones

Health-zone forecasts split the province forecast of whichever model is submitted.
So zone samples sum to their province's samples, sample by sample and week by week, and the national value stays the sum of the provinces.
Every zone of a forecast province is included, zones with no cases too.

For province $p$ with zones $z$, let $o_z(w)$ be zone $z$'s published count for week $w$ and $O(w) = \sum_z o_z(w)$.
Each province sample $k$ draws its zone shares once, for all weeks:

$$
\theta^{(k)} \sim \text{Dirichlet}(\alpha),
\qquad
\alpha_z = r_z + \kappa\, \frac{l_z + a}{\sum_{z'} (l_{z'} + a)}.
$$

$r_z$ is the zone's onsets over the 2 weeks before the vintage and $l_z$ those over the 16 weeks before it.
So recent shares are shrunk towards longer-run shares by $\kappa = 100$ pseudo-cases.
The floor $a = 1/2$ gives a zone with no cases a small share, so its draws are near zero but not always zero.
Each zone keeps its published count and the rest of the province sample $Y^{(k)}(w)$ is split:

$$
y^{(k)}_z(w) = o_z(w) + m_z,
\qquad
m \sim \text{Multinomial}\big(Y^{(k)}(w) - O(w),\ \theta^{(k)}\big).
$$

A sample below the published total is drawn from the published zone counts without replacement instead.
Province values are rounded first, as in the hub file.
The settings were the best of a small grid on the real vintages, and the differences across the grid were small (see [Evaluation](evaluation.md#health-zones)).

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
| `renewalhold` | $R_p(t)$ held at its last fitted value past the data. |
| `renewalpo` | One importation share $\varepsilon_q$ per origin, partially pooled. |

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
