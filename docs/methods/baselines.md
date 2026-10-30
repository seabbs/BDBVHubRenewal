# Baseline models

The persistence baselines, with the maths of each.
{ .page-subtitle }

The data and notation are on the [Data methods page](data.md#target-and-notation).
The renewal models are on the [Renewal models](model.md) page.
Health-zone forecasts split the province samples of either baseline as described under [Health zones](model.md#health-zones).

## Persistence

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

The weekly steps $\epsilon_{p,j}$ are independent, so the log level is a random walk from the last complete week.
$\sigma_p$ is the standard deviation of the 7 weekly changes in $\log(1 + Y_{p,v})$ over the last 8 complete weeks, floored at 0.1.
A province with fewer than 10 cases a week over those weeks takes $\sigma_p$ from the weekly changes of the provinces above that, pooled.
`persistencenow` uses the same walk and the same rule for $\sigma_p$.

Completeness comes from the hub's releases.
The totals of each onset week by release form a reporting triangle.
Each pair of releases a week apart up to $T_v$ gives, pooled over provinces, the growth $f_k$ of the totals of weeks $k$ weeks old to the next release.
The share of a week's final total published at $k$ weeks old is $1 / \prod_{j \ge k} f_j$, for $j$ up to 12 weeks, with $f_j = 1$ where no pair gives it.
Week $w$ is complete once that share is at least 0.95.
Without two releases a week apart, week $w$ is complete if its Wednesday is at least $L$ days before $T_v$, with $L$ the 95th centile of the delays published with the vintage, or 10 days if none were published.
Each sample is one joint trajectory across weeks, and the national value is the sum of the provinces.

## Persistence on a nowcast

`persistencenow` is a baseline model that builds on the ideas of the baselinenowcast paper[^bnc-paper] and package[^bnc-package].
It nowcasts the recent weeks with a chain ladder on the hub's totals by release, then runs the `persistence` random walk forward from each nowcast draw.

**Release triangle.**
The hub's totals $Y_{p,r}(w)$, known plus imputed, for onset week $w$ at each release $r$ form a cumulative reporting triangle, with onset week as reference and release as report date.
Each release is placed at the Saturday ending its week, so week $w$ is $k = (r - w)/7$ weeks old at it.
Late entries, re-imputation and onsets filled in for imputed cases all change the totals, so they are in the triangle.
With releases on 08-18, 10-09, 10-17 and 10-24, each week is seen at only a few ages, so the triangle is a band with no complete row.
baselinenowcast's `estimate_delay` needs a complete row, so the delay is estimated with the chain ladder in development-factor form, the same estimator on the cells the band has.
For each pair of releases a week apart up to $T_v$, a week $k$ weeks old at the first adds its total there to $D_k$ and at the second to $N_k$, and

$$
f_k = \frac{N_k}{D_k}, \qquad G_k = \prod_{j=k}^{12} f_j,
$$

with $f_k = 1$ where no pair gives it and weeks over 12 weeks old taken as final.
A province with at least 50 cases a week over the last 8 weeks uses its own $f_k$ where it has them, and the others use the factors pooled over provinces.
This part is our own code; the rest follows baselinenowcast.

**Nowcast.**
Week $w$, $k$ weeks old at $T_v$, is expected to reach $Y_{p,v}(w) G_k$, and the cases still to come are

$$
U_{p,k'}(w) \sim \text{NegBin}\Big(\text{mean } Y_{p,v}(w) (G_k - 1),\ \text{size } \phi_k\Big),
\qquad
C_{p,k'}(w) = Y_{p,v}(w) + U_{p,k'}(w),
$$

for each sample $k'$, as baselinenowcast's `sample_nb`.
The sizes $\phi_k$ are learnt from the releases, as `estimate_uncertainty_retro` does.
From each earlier release $r$ with a pair of releases a week apart up to it, the factors at $r$ predict the growth of each week's total from $r$ to $T_v$, which is compared with the growth seen, in each province.
A negative binomial is fitted to those points for each $k$ with at least 3 of them, and to all of them together for the rest (`fit_nb_size`, as baselinenowcast's `fit_nb`).
Without such a release the sizes are infinite, so the draws are Poisson.
At 10-24 only the release of 10-17 gives a retrospective nowcast, so each $\phi_k$ rests on one release of the 8 provinces, and the estimates will settle as weekly releases accumulate.
Without two releases a week apart, `persistencenow` is `persistence`.

**Forecast.**
Each sample $k'$ starts from its own nowcast of the latest week $c$, the one ending on or after $T_v$, $\log\{1 + C_{p,k'}(c)\}$.
From there it follows the `persistence` random walk, with the same code:

$$
\log\{1 + \lambda_{p,k'}(c + h)\} = \max\Big(0,\ \log\{1 + C_{p,k'}(c)\} + \sum_{j=1}^{h} \epsilon_{p,k',j}\Big),
\qquad
\epsilon_{p,k',j} \sim \mathcal{N}(0, \sigma_p^2),
\qquad
\tilde y_{p,k'}(c + h) \sim \text{Poisson}\big(\lambda_{p,k'}(c + h)\big).
$$

$\sigma_p$ is set as for `persistence`, from the weekly changes in $\log(1 + \hat y_p)$ over the 8 weeks up to $c$, with $\hat y_p(w) = Y_{p,v}(w) G_k$ the point nowcast.
Forecasts are kept at least the known onsets.
Counts reported by a later release $T$, $m$ weeks after $T_v$, are $Y_{p,v}(w) + \text{Binomial}\big(\max(B, 0),\ q\big) + \min(B, 0)$, with $B = C_{p,k'}(w) - Y_{p,v}(w)$ and $q = (G_k / G_{k+m} - 1)/(G_k - 1)$, the share of the growth still to come that arrives by $T$.

[^bnc-paper]: Johnson KE, Tang M, Tyszka E, Nemcova B, Wolffram D, Ergas R, Reich NG, Funk S, Mellor J, Bracher J, Abbott S (2025). Baseline nowcasting methods for handling delays in epidemiological data. Wellcome Open Research. [doi:10.12688/wellcomeopenres.25027.2](https://doi.org/10.12688/wellcomeopenres.25027.2).
[^bnc-package]: Johnson KE, Tyszka E, Bracher J, Funk S, Abbott S (2025). baselinenowcast: Methods for baseline nowcasting right-truncated epidemiological data. <https://github.com/epinowcast/baselinenowcast/>.
