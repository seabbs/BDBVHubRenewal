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
Week $w$ is complete if its Wednesday is at least $L$ days before $T_v$, with $L$ the 90th centile of the delays published with the vintage, or 10 days if none were published.
Each sample is one joint trajectory across weeks, and the national value is the sum of the provinces.
The median has no growth and weeks up to $c$ are still revised upwards, so persistence underpredicts during growth (see [Backtest](../backtest.md#persistence-bias)).

## Persistence on a nowcast

`persistencenow` is a baseline model that builds on the ideas of the baselinenowcast paper[^bnc-paper] and package[^bnc-package].
It nowcasts the recent weeks with a Python port of [baselinenowcast](https://github.com/epinowcast/baselinenowcast), then runs the hub baseline forward from each nowcast draw.
Tests check the port's delay, point nowcast and negative binomial sizes against baselinenowcast 0.2.0 from CRAN on fixed triangles, and a CI job regenerates those fixtures with the R package (`scripts/baselinenowcast_fixtures.R`).

**Delay and uncertainty.**
The reporting triangle $n(W, d)$ counts the cases in the vintage's delay file by onset week $W$ and reported delay $d$ in days, from onset to report, up to 35 days.
Delays are not matched across delay files.
The hub fills in onsets for cases it had imputed, and these appear in later files with their original report delay while the totals stay the same, so they are not late reports.
A case is taken as published on day $W + d$, so cell $(W, d)$ is observed if $W + d \le T_v$.
On the last 6 onset weeks, a chain ladder gives the delay pmf $p$ with cdf $F$, as baselinenowcast's `estimate_delay` does.
The same is done on the triangle cut back to each of the 8 previous weeks, and each open row is filled from its pmf as `apply_delay` does.
Comparing what each of those nowcasts expected to be added to each onset week with what has been added since gives a negative binomial size $\phi_h$ for each horizon $h$, the weeks from that onset week to the latest, as `estimate_uncertainty_retro` does.
So the sizes are fitted to weekly sums.
Ituri and Nord-Kivu, with at least 50 delays a week, use their own $p$ and $\phi_h$.
The other provinces borrow those of the triangle pooled over provinces, as baselinenowcast borrows a delay from another triangle.

**Nowcast.**
A case with onset on day $W - u$ of week $W$, delay $d$ and publication lag $\ell$ is in the vintage if $d + \ell - u \le T_v - W$.
With onsets spread evenly over the week, the share of week $w$ reported is

$$
s_p(w) = \frac{1}{7} \sum_{u=0}^{6} \sum_{\ell} P(\ell)\, F_p(T_v - w + u - \ell),
$$

with the publication lag $\ell$ uniform on 0 to 3 days with probability 0.9 and on 4 to 28 days with probability 0.1, for the cases that reach the hub's totals weeks after report.
The delay file holds no report or publication dates, so the lag cannot be learnt from it.
Without it, the shares are too high and recent weeks are nowcast too low.

The published total $Y_{p,v}(w)$, known plus imputed, is expected to reach $Y_{p,v}(w)/s_p(w)$, and the cases still to come are

$$
U_{p,k}(w) \sim \text{NegBin}\Big(\text{mean } Y_{p,v}(w) \frac{1 - s_p(w)}{s_p(w)},\ \text{size } \phi_{p,h(w)}\Big),
$$

for each sample $k$, with $h(w)$ the weeks from $w$ to the latest week $c$, the one ending on or before $T_v$.

The hub re-imputes missing onsets at each vintage, so totals can also fall.
Each province adds a revision $E_{p,k}(w) = A - B$ with $A, B \sim \text{Poisson}\{\kappa I_{p,v}(w)/2\}$, so its variance is $\kappa I_{p,v}(w)$:

$$
C_{p,k}(w) = \max\big\{Y_{p,v}(w) + U_{p,k}(w) + E_{p,k}(w),\ K_{p,v}(w)\big\},
\qquad
\kappa = \frac{\sum \{Y_{p,b}(w) - Y_{p,a}(w)\}^2}{\sum I_{p,a}(w)}.
$$

The sums run over consecutive vintages $a < b$ up to $T_v$ and the weeks that ended at least 63 days before $a$.
So $C_{p,k}(w)$ can fall below the published total, but not below the known onsets $K_{p,v}(w)$.

**Forecast.**
Each sample $k$ starts from its own nowcast of the latest week, $\log\{1 + C_{p,k}(c)\}$.
From there it follows the baseline of the [COVID-19 Forecast Hub](https://github.com/reichlab/covidModels) and FluSight on the log scale, as [BVDOutbreakSize](https://github.com/epiforecasts/BVDOutbreakSize) does for its persistence baseline:

$$
\log\{1 + \tilde y_{p,k}(c + h)\} = \log\{1 + C_{p,k}(c)\} + \sum_{j=1}^{h} \delta_{p,k,j},
$$

with each step $\delta_{p,k,j}$ drawn with equal probability from $\pm \Delta_{p,i}$.
$\Delta_{p,i}$, $i = 1, \dots, 7$, are the weekly changes in $\log(1 + \hat y_p)$ over the 8 weeks up to $c$, with $\hat y_p(w) = Y_{p,v}(w) / s_p(w)$ the point nowcast.
With their negatives, the median stays at the start and the spread grows with the horizon.
Provinces without their own delay draw from the changes of the provinces with one, so a few small, noisy counts do not set their spread.
A province whose mean nowcast over the last 8 weeks, $\bar y_{p,k}$, has a median over samples below 10 cases a week instead walks the log of that mean as a Poisson rate, $\tilde y_{p,k}(c + h) \sim \text{Poisson}\{\bar y_{p,k} \exp(\sum_{j \le h} \delta_{p,k,j})\}$, so a few cases a week stay plausible.
With fewer than 3 changes the forecast is Poisson around the start.
Forecasts are rounded and kept at least the known onsets.
Counts reported by a later date $T$ are $Y_{p,v}(w) + \text{Binomial}\big(\max(D, 0),\ q\big) + \min(D, 0)$, with $D = C_{p,k}(w) - Y_{p,v}(w)$ and $q = \{s_p(T - w) - s_p(w)\}/\{1 - s_p(w)\}$.
Without a complete onset week or 50 cases in the last 6 weeks, or with fewer than 3 past nowcasts, `persistencenow` is `persistence`.

[^bnc-paper]: Johnson KE, Tang M, Tyszka E, Nemcova B, Wolffram D, Ergas R, Reich NG, Funk S, Mellor J, Bracher J, Abbott S (2025). Baseline nowcasting methods for handling delays in epidemiological data. Wellcome Open Research. [doi:10.12688/wellcomeopenres.25027.2](https://doi.org/10.12688/wellcomeopenres.25027.2).
[^bnc-package]: Johnson KE, Tyszka E, Bracher J, Funk S, Abbott S (2025). baselinenowcast: Methods for baseline nowcasting right-truncated epidemiological data. <https://github.com/epinowcast/baselinenowcast/>.
