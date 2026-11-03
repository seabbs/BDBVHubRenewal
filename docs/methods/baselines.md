# Baseline models

`randomwalknowcast` and `randomwalk`, its comparison model.
{ .page-subtitle }

The data and notation are on the [Data methods page](data.md#target-and-notation).
The renewal models are on the [Renewal models](model.md) page.
`randomwalknowcast` is baselinenowcast[^bnc-paper][^bnc-package] with a random walk on top.
`randomwalk` is the same walk with no nowcast.
Both run at province and zone level ([Health zones](#health-zones)), and national is the sum of the provinces.

## Daily release snapshots

Both models use the daily onsets of each hub release.
$N_{p,r}(t)$ is the known plus imputed onsets in province $p$ on day $t$ at release $r$, which sum over a hub week to the weekly target.
Release $r$ has a cut-off $c_r$, the latest onset day in the release over all provinces.
Day $t \le c_r$ is observed at $r$ and is $a = c_r - t$ days old.
Cells after the cut-off are not observed.
Hub weeks run Sunday to Saturday and are labelled by the Saturday.
$Y_{p,r}(w)$ is the sum of $N_{p,r}$ over the days of week $w$.
$w$ is $k$ weeks back when it is $k$ weeks before the week holding the release cut-off.

A pair of consecutive releases $(r, r + 1)$ has gap $s = c_{r+1} - c_r$ days.
$D = 42$ days is baselinenowcast's `max_delay`, and a pair is in window when $s \le D$.
Only in-window pairs inform the delay.
A pair outside the window keeps its counts in the data.
A model raises if the vintage has no snapshots or no in-window pair.

## Delay

The delay follows baselinenowcast[^bnc-paper][^bnc-package] on the daily snapshots, with one extension.
Releases are batched reports of one daily delay.
With $g_p(j) \ge 0$ the daily log development factor at age $j$, and $g_p(j) = 0$ for $j \ge D$, a day of age $x$ at release $r$ grows to release $r + 1$ by $\exp \sum_{j=x}^{x+s-1} g_p(j)$.
For each gap $s$ and age $x = 0, \dots, D - 1$,

$$
\log \frac{\sum N_{p,r+1}(t)}{\sum N_{p,r}(t)} = \sum_{j=x}^{x+s-1} g_p(j),
$$

summed over the in-window pairs of gap $s$ and over days $t$ of age $x$ at $r$.
$g_p$ is the non-negative least squares fit of all these equations, each weighted by its denominator $\sum N_{p,r}(t)$, with no smoothing over age.
With a gap of 1 only, $\exp g_p(x)$ is baselinenowcast's chain ladder factor.
Cells a release does not observe are skipped, not filled, and a gap and age with a zero denominator give no equation.
The parent of a province is the country, whose counts are the sum of its provinces.
Only onset days within $1.5 D = 63$ days of the cut-off of the latest release used contribute to the sums, as baselinenowcast's default `n_history_delay`, and a retrospective fit uses the cut-off of its own release.
Before the sums, each day's increments between its releases go through baselinenowcast's `preprocess_negative_values`, so a downward revision is set to zero and taken from the earlier increments of that day and no ratio falls below 1.
This applies to the delay only, because the removals below still allow negative revisions.
The preprocessing runs along releases for each onset day, where baselinenowcast runs it along the delay columns of a reporting triangle.

The growth of a day's count from age $a$ to final is

$$
G_p(a) = \exp \sum_{j=a}^{D-1} g_p(j),
\qquad
G_p(a) = 1 \text{ for } a \ge D,
$$

and $1 / G_p(a)$ is the share of it reported.
There is no growth past $D$, so an equation whose ages run past $D$ puts that growth on its ages below $D$.

## Nowcast

Every observed day is nowcast, with no handover and no minimum count.
The nowcast mean of day $t$ is $N_{p,o}(t)\, G_p(c_o - t)$, summed to hub weeks as $\mu_p(w)$.
With $\text{rem} = \mu_p(w) - Y_{p,o}(w) \ge 0$ the count still to come, the final count is new reports minus removals, as in epinowcast,

$$
X_{p,i}(w) = \max\big\{0,\ Y_{p,o}(w) + A - B\big\},
\qquad
A \sim \text{NB}\big(\text{rem},\ \phi_{p,k}\big),
\qquad
B \sim \text{Poisson}\big(b_{p,k}\, Y_{p,o}(w)\big).
$$

$A$ is baselinenowcast's negative binomial on the remaining count, with mean $\text{rem}$ and size $\phi_{p,k}$, drawn with the port's `sample_nb`.
$B$ removes a share $b_{p,k}$ of the count reported, so counts can be revised down.
Before the floor the mean is $\mu_p(w) - b_{p,k} Y_{p,o}(w)$.
Sample $i$ is one draw of $X_{p,i}(w)$ per week.

**Fit.**
For each earlier release $r < o$ with an in-window pair, a nowcast $\hat y_p(w)$ is made from the data up to $r$, with the delay of $r$.
Each day grows over the gap $c_o - c_r$ by $\exp \sum_{j=a}^{a+c_o-c_r-1} g_p(j)$, with the $g_p$ of $r$.
For weeks back $k$ with a day younger than $D$ at $r$, and where $Y_{p,r}(w) > 0$ or $\hat y_p(w) > 0$, the count of $w$ on days up to $c_r$ is compared between releases $r$ and $o$.
A rise is the new reports $A_{p,r}(w)$ and a fall the removals $B_{p,r}(w)$, so both are weekly, as the forecast is.
With $\widehat{\text{rem}} = \hat y_p(w) - Y_{p,r}(w)$,

$$
\phi_{p,k} = \arg\max_\phi \sum \log \text{NB}\big(A_{p,r}(w);\ \widehat{\text{rem}},\ \phi\big),
\qquad
b_{p,k} = \frac{\sum B_{p,r}(w)}{\sum Y_{p,r}(w)},
$$

summed over the kept releases.
$\phi_{p,k}$ is fitted with the port's `fit_nb` through `fit_by_horizon`, as baselinenowcast does, on $[0.1, 1000]$.
A week with $\widehat{\text{rem}} = 0$ says nothing about $\phi$, so it is left out of that fit.
$b_{p,k}$ is the Poisson maximum likelihood estimate of the removals.
An estimate with fewer than 3 errors at $k$ has $\phi_{p,k} = 1000$, the Poisson end of the range.

## Own or pooled estimates

The delay $g_p(x)$ is own or parent by age.
A place uses its own $g_p(x)$ when its denominator $\sum N_{p,r}(t)$ at age $x$, summed over the pairs, is at least 10 cases.
Otherwise it uses its parent's at that age (zone, province, country).
The country uses its own.
The observation model $(\phi_{p,k}, b_{p,k})$ and the spread of the walk $\sigma_p$ use one rule with a threshold of 10 cases a week and 3 errors.
$\ell_p$ is the mean of $Y_{p,r}(w)$ over weeks 3 to 6 back, in cases a week.
A place with $\ell_p \ge 10$ uses its own estimate.
For $(\phi_{p,k}, b_{p,k})$ it also needs at least 3 errors at $k$.
Otherwise it uses the pooled estimate of its nearest parent that meets the same rule, and otherwise the country's.
A pooled estimate uses the errors and weekly changes of the place's children with $\ell \ge 10$ together.
A place's counts are the sum of its children's, so the place itself is pooled only when none of its children has $\ell \ge 10$.

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

the estimated share of the week's final count reported, under the delay above.
A day with $G_p = 0$ counts as fully reported.
$\pi_p(w)$ can exceed 1 where counts are revised down.
If no week has $\pi_p(w) \ge 0.95$ for a place, `randomwalk` raises (`NoCompleteWeek`) and gives no forecast, with nothing in its place.
$x_i = Y_{p,o}(w_0)$ for every sample.
Weeks up to $w_0$ are the reported counts.
Later weeks with data are walk draws, floored at the reported count.

## Random walk on a nowcast

`randomwalknowcast` starts each sample from its own nowcast $X_{p,i}$ of the latest week with all 7 days on or before $c_o$, so $x_i = X_{p,i}(w_0)$.
Weeks up to $w_0$ are the nowcast draws.
If the week after $w_0$ holds $c_o$ and is partial, $n$ days of it have no data.
Its value is its nowcast draw plus Poisson$(n \lambda_i / 7)$, with $\lambda_i$ the walk rate of that week.
Later weeks are walk draws.

## Health zones

`randomwalk` and `randomwalknowcast` run on each zone's own release snapshots with the same code as for provinces.
The hierarchy is zone, province, country, and each zone takes its delay, spread and walk spread by the rules [above](#own-or-pooled-estimates).
Zone samples sum to their province's samples, sample by sample and week by week.
Province sample $Y^{(k)}(w)$ is apportioned over the zones in proportion to the zone model's own sample $x^{(k)}_z(w)$ of the same id and week:

$$
y^{(k)}_z(w) \approx Y^{(k)}(w)\, \frac{x^{(k)}_z(w)}{\sum_{z'} x^{(k)}_{z'}(w)}.
$$

Largest remainders make the values whole numbers that sum to $Y^{(k)}(w)$.
Where every $x^{(k)}_z(w)$ is 0 the split uses the Dirichlet mean shares of the [Renewal models](model.md#split-of-province-samples) page.

## Limitations

The release history in the delay window is shorter than $D$, so no chain to the final count has been checked against a final count.
Each delay factor rests on one onset day per age per pair, and the youngest ages rest on few cases even nationally.
The onset clock assumes a constant lag from the latest onset day to extraction, and the real lag varies by a few days.
The chain ladder treats reporting speed as fixed by age and step, but speed and backlogs vary between releases.
The spread is learnt from few release transitions.
Counts include the hub's imputed onsets, which are re-drawn as cases gain onsets, so changes between releases mix reporting with imputation.
Places are sampled independently, so a shock to a whole release is not shared, and the national band is a sum of independent provincial bands.
The walk has no drift, and its spread comes from 8 weekly changes, which for small places include Poisson noise.

[^bnc-paper]: Johnson KE, Tang M, Tyszka E, Nemcova B, Wolffram D, Ergas R, Reich NG, Funk S, Mellor J, Bracher J, Abbott S (2025). Baseline nowcasting methods for handling delays in epidemiological data. Wellcome Open Research. [doi:10.12688/wellcomeopenres.25027.2](https://doi.org/10.12688/wellcomeopenres.25027.2).
[^bnc-package]: Johnson KE, Tyszka E, Bracher J, Funk S, Abbott S (2025). baselinenowcast: Methods for baseline nowcasting right-truncated epidemiological data. <https://github.com/epinowcast/baselinenowcast/>.
