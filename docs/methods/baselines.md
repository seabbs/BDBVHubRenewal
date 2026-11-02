# Baseline models

The two random walk baselines, with the maths of each.
{ .page-subtitle }

The data and notation are on the [Data methods page](data.md#target-and-notation).
The renewal models are on the [Renewal models](model.md) page.
Both baselines also run at zone level, as described under [Health zones](model.md#health-zones).

## Daily release snapshots

Both models use the daily onsets of each hub release.
$N_{p,r}(t)$ is the known plus imputed onsets in province $p$ on day $t$ at release $r$, which sum over a hub week to the weekly target.
Release $r$ has a cut-off $c_r$, the latest onset day in the release over all provinces.
Day $t \le c_r$ is observed at $r$ and is $a = c_r - t$ days old.
Cells after the cut-off are not observed.
Hub weeks run Sunday to Saturday and are labelled by the Saturday.
$Y_{p,r}(w)$ is the sum of $N_{p,r}$ over the days of week $w$.
$w$ is $k$ weeks back when it is $k$ weeks before the week holding the release cut-off.

A pair of consecutive releases $(r, r + 1)$ has step $s = c_{r+1} - c_r$ days.
$D = 42$ days is baselinenowcast's `max_delay`, and a pair is in window when $s \le D$.
Only in-window pairs give delay factors.
A pair outside the window keeps its counts in the data.
A model raises if the vintage has no snapshots or no in-window pair.

## Delay

The delay follows baselinenowcast[^bnc-paper][^bnc-package] on the daily snapshots, with one extension.
For step $s$ and age $x = 0, \dots, D - 1$,

$$
F_{p,s}(x) = \frac{\sum N_{p,r+1}(t)}{\sum N_{p,r}(t)},
$$

summed over in-window pairs of step $s$ and over days $t$ of age $x$ at $r$.
Cells a release does not observe are skipped, not filled.
$F_{p,s}(x)$ is not identified where the denominator is 0, and takes the factor of the parent of $p$ at that age.
The parent of a province is the country, whose counts are the sum of its provinces.

At the latest release $o$, $s_o$ is the step of its latest in-window pair.
Day $t$ of age $a$ is chained over the actual release gap,

$$
G_p(a) = \prod_{m=0}^{M(a)-1} F_{p,s_o}(a + m s_o),
\qquad
G_p(a) = 1 \text{ for } a \ge D,
$$

with $M(a)$ the fewest steps with $a + M s_o \ge D$.
There is no factor past $D$.
$G_p(a)$ is the growth of a day's count from age $a$ to final, and $1 / G_p(a)$ is the share of it reported.

## Nowcast

Every observed day is nowcast, with no handover and no minimum count.
The nowcast mean of day $t$ is $N_{p,o}(t)\, G_p(c_o - t)$, summed to hub weeks as $\mu_p(w)$.
With $\mu_p(w) - Y_{p,o}(w)$ the change still to come, the count at the next release is Skellam,

$$
X_{p,i}(w) = \max\big\{0,\ Y_{p,o}(w) + A - B\big\},
\qquad
A \sim \text{Poisson}\big(\text{rem}^+ + \rho_{p,k}\, \mu_p(w)\big),
\qquad
B \sim \text{Poisson}\big(\text{rem}^- + \rho_{p,k}\, \mu_p(w)\big),
$$

with $\text{rem} = \mu_p(w) - Y_{p,o}(w)$, $\text{rem}^+ = \max(\text{rem}, 0)$ and $\text{rem}^- = \max(-\text{rem}, 0)$.
Before the floor the mean is $\mu_p(w)$ and the variance is $|\text{rem}| + 2\rho_{p,k}\mu_p(w)$, so counts can be revised down.
Sample $i$ is one draw of $X_{p,i}(w)$ per week.

**Spread.**
For each earlier release $r < o$ with an in-window pair, a nowcast $\hat y_p(w)$ is made from the data up to $r$, with the delay of $r$.
It is chained over the gap $c_o - c_r$ only.
Each remaining part of the gap takes the step of the in-window pair at $r$ nearest to it, ties to the latest pair, and ages are never interpolated.
The release is kept when the steps differ from the gap by at most 1 day.
The target is $y_p(w)$, the count of $w$ at release $o$ on days up to $c_r$.
For weeks back $k$ with a day younger than $D$ at $r$, and where $Y_{p,r}(w) > 0$ or $\hat y_p(w) > 0$,

$$
e = y_p(w) - \hat y_p(w),
\qquad
\text{rem} = \hat y_p(w) - Y_{p,r}(w),
\qquad
\rho_{p,k} = \max\Big(0,\ \frac{\sum (e^2 - |\text{rem}|)}{\sum 2\hat y_p(w)^+}\Big),
$$

summed over the kept releases.
Weeks without a $\rho_{p,k}$ have $\rho = 0$.

## Own or pooled estimates

The delay $F_{p,s}$, the spread $\rho_{p,k}$ and the spread of the walk $\sigma_p$ use one rule and one threshold.
$\ell_p$ is the mean of $Y_{p,r}(w)$ over weeks 3 to 6 back, in cases a week.
A province with $\ell_p \ge 10$ uses its own estimate.
For $\rho_{p,k}$ it also needs at least 2 errors at $k$.
Otherwise it uses the pooled estimate of the country, which pools the provinces with $\ell \ge 10$.
The pooled $F$ is the country's own, since its counts are the sum of the provinces.
The pooled $\rho_{p,k}$ and $\sigma_p$ use the errors and weekly changes of those provinces together.
If the pooled estimate has fewer than 2 errors at $k$, $\rho_{p,k} = 0$.

## Walk

`randomwalk` and `randomwalknowcast` run the same walk from a start week $w_0$ with start value $x_i$ in sample $i$.
For $h \ge 1$,

$$
\log\{1 + \lambda_i(w_0 + h)\} = \max\Big(0,\ \log\{1 + x_i\} + \sum_{j=1}^{h} \epsilon_{i,j}\Big),
\qquad
\epsilon_{i,j} \sim \mathcal{N}(0, \sigma_p^2),
$$

$$
\tilde y_i(w_0 + h) \sim \text{Poisson}\big(\lambda_i(w_0 + h)\big).
$$

Steps are summed within a sample, so each sample is one trajectory.
National is the sum of the provinces, sample by sample.

$\sigma_p$ is the standard deviation of the 8 weekly changes of $\log\{1 + Y_{p,o}(w)\}$ over the 9 weeks ending at the start week of `randomwalk`.
Both models use reported counts for $\sigma_p$.

## Random walk

`randomwalk` makes no nowcast.
Its start week $w_0$ is the latest week with

$$
\pi_p(w) = \frac{1}{7} \sum_{t \in w,\ t \le c_o} \frac{1}{G_p(c_o - t)} \ge 0.95,
$$

the share of the week's final count reported, under the delay above.
$x_i = Y_{p,o}(w_0)$ for every sample.
Weeks up to $w_0$ are the reported counts.
Later weeks with data are walk draws, floored at the reported count.

## Random walk on a nowcast

`randomwalknowcast` starts each sample from its own nowcast $X_{p,i}$ of the latest week with all 7 days on or before $c_o$, so $x_i = X_{p,i}(w_0)$.
Weeks up to $w_0$ are the nowcast draws.
If the week after $w_0$ holds $c_o$ and is partial, $n$ days of it have no data.
Its value is its nowcast draw plus Poisson$(n \lambda_i / 7)$, with $\lambda_i$ the walk rate of that week.
Later weeks are walk draws.

[^bnc-paper]: Johnson KE, Tang M, Tyszka E, Nemcova B, Wolffram D, Ergas R, Reich NG, Funk S, Mellor J, Bracher J, Abbott S (2025). Baseline nowcasting methods for handling delays in epidemiological data. Wellcome Open Research. [doi:10.12688/wellcomeopenres.25027.2](https://doi.org/10.12688/wellcomeopenres.25027.2).
[^bnc-package]: Johnson KE, Tyszka E, Bracher J, Funk S, Abbott S (2025). baselinenowcast: Methods for baseline nowcasting right-truncated epidemiological data. <https://github.com/epinowcast/baselinenowcast/>.
