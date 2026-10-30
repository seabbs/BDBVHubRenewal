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

`persistencenow` nowcasts recent weeks with a chain ladder on the hub's releases, as in baselinenowcast[^bnc-paper][^bnc-package], then runs the `persistence` random walk from each nowcast draw.

**Release triangle.**
$Y_{p,r}(w)$ is the total, known plus imputed, of onset week $w$ in province $p$ at release $r$.
Week $w$ is $k = \lfloor (r - w)/7 \rfloor$ weeks old at $r$, and weeks ending after $r$ are left out.
For consecutive releases $a < b$ up to $T_v$, the weeks $k_a$ weeks old at $a$ and $k_b$ at $b$ give

$$
\log \frac{\sum_w Y_{p,b}(w)}{\sum_w Y_{p,a}(w)} = \sum_{j=k_a}^{k_b - 1} \log f_j,
$$

fitted by least squares weighted by $\sum_w Y_{p,a}(w)$.
$f_j = 1$ for ages no pair covers, and weeks over 12 weeks old are final, so $G_k = \prod_{j=k}^{12} f_j$.
Provinces with at least 50 cases a week over the last 8 weeks use their own $f_j$ where their releases cover $j$, and the others use $f_j$ pooled over provinces.
With one release, `persistencenow` is `persistence`.

**Nowcast.**
For week $w$, $k$ weeks old at $T_v$, and sample $i$,

$$
C_{p,i}(w) = \max\big\{0,\ Y_{p,v}(w) + A - B\big\},
\qquad
A \sim \text{Poisson}(\mu^+ + \rho_k Y_{p,v}(w)),
\qquad
B \sim \text{Poisson}(\mu^- + \rho_k Y_{p,v}(w)),
$$

with $\mu = Y_{p,v}(w)(G_k - 1)$ and $\mu^+$, $\mu^-$ its positive and negative parts.
The change still to come is Skellam with mean $\mu$ and variance $|\mu| + 2 \rho_k Y_{p,v}(w)$, so totals can be revised down.
$\rho_k$ is fitted by maximum likelihood to nowcasts made at each earlier release with factors, compared with $T_v$, pooled over provinces.
Ages with fewer than 3 points share one $\rho$.

**Forecast.**
Each sample starts from its own nowcast of the latest week $c$, the last ending on or before $T_v$:

$$
\log\{1 + \lambda_{p,i}(c + h)\} = \max\Big(0,\ \log\{1 + C_{p,i}(c)\} + \sum_{j=1}^{h} \epsilon_{p,i,j}\Big),
\qquad
\epsilon_{p,i,j} \sim \mathcal{N}(0, \sigma_p^2),
\qquad
\tilde y_{p,i}(c + h) \sim \text{Poisson}\big(\lambda_{p,i}(c + h)\big).
$$

$\sigma_p$ is set as for `persistence`, from $\log\{1 + Y_{p,v}(w) G_k\}$ over the 8 weeks up to $c$.
Counts reported by a later release $T$ are $Y_{p,v}(w) + \text{Binomial}\big(\max(D, 0),\ q\big) + \min(D, 0)$, with $D = C_{p,i}(w) - Y_{p,v}(w)$, $q = (G_k / G_{k'} - 1)/(G_k - 1)$ and $k'$ the age of $w$ at $T$.

[^bnc-paper]: Johnson KE, Tang M, Tyszka E, Nemcova B, Wolffram D, Ergas R, Reich NG, Funk S, Mellor J, Bracher J, Abbott S (2025). Baseline nowcasting methods for handling delays in epidemiological data. Wellcome Open Research. [doi:10.12688/wellcomeopenres.25027.2](https://doi.org/10.12688/wellcomeopenres.25027.2).
[^bnc-package]: Johnson KE, Tyszka E, Bracher J, Funk S, Abbott S (2025). baselinenowcast: Methods for baseline nowcasting right-truncated epidemiological data. <https://github.com/epinowcast/baselinenowcast/>.
