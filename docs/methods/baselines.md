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

$\sigma_p$ is the SD of the 7 weekly changes in $\log(1 + Y_{p,v})$ over the last 8 complete weeks, floored at 0.1.
Week $w$ is complete if its Wednesday is at least $L$ days before $T_v$, with $L$ the 90th centile of the delays published with the vintage, or 10 days if none were published.
Each sample is one joint trajectory across weeks, and the national value is the sum of the provinces.
The median has no growth and weeks up to $c$ are still revised upwards, so persistence underpredicts during growth (see [Backtest](../backtest.md#persistence-bias)).

## Persistence on a nowcast

`persistencenow` is a baseline model that builds on the ideas of the baselinenowcast paper[^bnc-paper] and package[^bnc-package].
It nowcasts the recent weeks with a Python port of [baselinenowcast](https://github.com/epinowcast/baselinenowcast), then applies persistence to each nowcast draw.
Tests check the port's delay, point nowcast and negative binomial sizes against baselinenowcast 0.2.2 on fixed triangles, and a CI job regenerates those fixtures with the R package (`scripts/baselinenowcast_fixtures.R`).

The reporting triangle $n(W, d)$ counts the cases in the vintage's delay file by onset week $W$ and delay $d$ in days, up to 35 days.
Ituri and Nord-Kivu, with at least 50 delays a week, each have their own triangle. The other provinces borrow the triangle pooled over all provinces, as baselinenowcast borrows a delay from another triangle.
A case is published on day $W + d$, its onset mid-week plus its delay plus a 3-day publication lag, so cell $(W, d)$ is observed if $W + d \le T_v$.
On the last 6 onset weeks, a chain ladder fills the missing cells column by column,

$$
\hat n(W, d) = \frac{\sum_{W' \in A_d} n(W', d)}{\sum_{W' \in A_d} \sum_{d' < d} n(W', d')} \sum_{d' < d} \hat n(W, d'),
$$

with $A_d$ the weeks where delay $d$ is observed.
Week $w$ gets the ratio $r(w)$ of its filled cells to its observed cells.
The same is done on the triangle cut back to each of the 8 previous weeks.
Comparing what each of those nowcasts expected to be added to week $w$ with what has been added since gives a negative binomial size $\phi_h$ for each horizon $h$, fitted by maximum likelihood.
Each province's delay is applied to its published total, known plus imputed, and its cases still to come are drawn on their own:

$$
U_{p,k}(w) \sim \text{NegBin}\big(\text{mean } r_p(w)\, Y_{p,v}(w),\ \text{size } \phi_{p,h(w)}\big).
$$

The hub re-imputes missing onsets at each vintage, so totals can also fall.
Each province adds a revision $E_{p,k}(w) = A - B$ with $A, B \sim \text{Poisson}\{\kappa I_{p,v}(w)/2\}$, so its variance is $\kappa I_{p,v}(w)$:

$$
C_{p,k}(w) = \max\big\{Y_{p,v}(w) + U_{p,k}(w) + E_{p,k}(w),\ K_{p,v}(w)\big\},
\qquad
\kappa = \frac{\sum \{Y_{p,b}(w) - Y_{p,a}(w)\}^2}{\sum I_{p,a}(w)}.
$$

The sums run over consecutive vintages $a < b$ up to $T_v$ and the weeks that ended at least 63 days before $a$.
So $C_{p,k}(w)$ can fall below the published total, but not below the known onsets.

The last week with $1/\{1 + r(w)\} \ge 0.5$ under the pooled triangle takes the place of $c$, and persistence runs from $C_{p,k}(c)$ in each sample $k$, with its spread at $h$ steps scaled by $\omega_h$:

$$
\log\{1 + \lambda_{p,k}(c + h)\} = \max\Big(0,\ \log\{1 + C_{p,k}(c)\} + \sum_{j=1}^{h} \epsilon_{p,j}\Big),
\qquad
\operatorname{Var}\Big(\sum_{j=1}^{h} \epsilon_{p,j}\Big) = \omega_h^2 h \sigma_p^2,
$$

$$
\tilde y_{p,k}(c + h) = \max\big\{\text{Poisson}(\lambda_{p,k}(c + h)),\ K_{p,v}(c + h)\big\}.
$$

$\sigma_p$ is taken as for `persistence`, from $Y_{p,v}(1 + r)$ over the 8 weeks up to $c$.
$\omega_h$ calibrates the walk on past errors, as baselinenowcast calibrates its nowcast.
From each earlier vintage, each province with a nowcast level $L \ge 10$ gives $z = \{\log(1 + y) - \log(1 + L)\}/(\sigma \sqrt h)$ against its count $y$ in vintage $v$, for weeks at least 95% reported by then.
Then $\omega_h^2 = (\sum z^2 + 10)/(n_h + 10)$, which shrinks to 1 when there are few errors.
Only data available at $T_v$ are used.
Counts reported by a later date $T$ are $Y_{p,v}(w) + \text{Binomial}\big(\max(D, 0),\ q\big) + \min(D, 0)$, with $D = C_{p,k}(w) - Y_{p,v}(w)$ and $q$ the share of the remaining delay mass of the province's pmf that is published by $T$.
Late rows in the delay files are mostly onsets recorded later for cases already counted as imputed, so the publication lag has no long tail.
Without a complete onset week or 50 cases in the last 6 weeks, or with fewer than 3 past nowcasts, `persistencenow` is `persistence`.

[^bnc-paper]: Johnson KE, Tang M, Tyszka E, Nemcova B, Wolffram D, Ergas R, Reich NG, Funk S, Mellor J, Bracher J, Abbott S (2025). Baseline nowcasting methods for handling delays in epidemiological data. Wellcome Open Research. [doi:10.12688/wellcomeopenres.25027.2](https://doi.org/10.12688/wellcomeopenres.25027.2).
[^bnc-package]: Johnson KE, Tyszka E, Bracher J, Funk S, Abbott S (2025). baselinenowcast: Methods for baseline nowcasting right-truncated epidemiological data. <https://github.com/epinowcast/baselinenowcast/>.
