# Baseline models

`randomwalknowcast`, its partially pooled variant `pprandomwalknowcast`, and `randomwalk`, their comparison model.
{ .page-subtitle }

The data and notation are on the [Data methods page](data.md#target-and-notation).
The renewal models are on the [Renewal models](model.md) page.
`randomwalknowcast` nowcasts the recent weeks with baselinenowcast[^bnc-paper][^bnc-package] and forecasts with a random walk from the nowcast.
`randomwalk` is the same walk from the latest complete week, with no nowcast.
Both run at province and zone level ([Health zones](#health-zones)), and national is the sum of the provinces.

In brief, `randomwalknowcast`:

1. follows baselinenowcast, through a JAX port of the R package with one named extension for release snapshots ([Delay](#delay));
2. has an observation model for the nowcast as in baselinenowcast, negative binomial new reports on the count still to come, fitted from retrospective nowcasts ([Nowcast](#nowcast));
3. nowcasts every week up to the cut-off and completes a partial cut-off week from a walk that starts at the latest full week at least 5% reported with at least 5 cases ([Start from the nowcast](#start-from-the-nowcast));
4. continues each nowcast trajectory with a random walk on $\log(1 + y)$ ([Forecast](#forecast));
5. takes the walk's variance net of Poisson noise ([Forecast](#forecast));
6. shares the delay, size and walk variance with a parent where a place has fewer than 10 cases a week ([Own or pooled estimates](#own-or-pooled-estimates)).

Two further rules keep places with few or no cases sensible: introductions for places with no recent case ([Forecast](#forecast)), and zone samples that sum to their province's ([Health zones](#health-zones)).

## Structure

`randomwalknowcast` nowcasts each recent week and walks forward from each nowcast draw.

```mermaid
flowchart TD
    subgraph TRI["Reporting triangle, per place p"]
        SNAP["Daily onsets C<sub>p,r</sub>(t)<br/>by onset day and release r"]
        PAIR["Release pairs (r, r + 1)<br/>gap s ≤ D = 42 days"]
    end
    DEL["Daily delay Δ<sub>p</sub>(j) ≥ 0<br/>weighted non-negative<br/>least squares<br/><i>own or parent by age</i>"]
    G["Growth to final Γ<sub>p</sub>(a)<br/>share reported 1 / Γ<sub>p</sub>(a)"]
    MU["Nowcast mean μ<sub>p</sub>(w)<br/>weekly fill<br/>shares F<sub>p,j</sub>(w) from Γ<sub>p</sub>"]
    RETRO["Earlier releases r &lt; o<br/>nowcast to release o"]
    OBS["Observation model<br/>by weeks back k<br/>size φ<sub>p,k</sub>, removal b<sub>p,k</sub>"]
    X["Nowcast draw X<sub>p,i</sub>(w)<br/>max(0, Y<sub>p,o</sub> + A − B)<br/>A ~ NB(μ<sub>p</sub> − Y<sub>p,o</sub>, φ<sub>p,k</sub>)<br/>B ~ Poisson(b<sub>p,k</sub> Y<sub>p,o</sub>)"]
    SIG["Walk spread σ<sub>p</sub><br/>changes in log(1 + Y<sub>p,o</sub>)"]
    W["Walk on log(1 + λ<sub>i</sub>)<br/>from each sample's completed<br/>cut-off week<br/>places with no recent case 0<br/>until an introduction<br/>Poisson counts"]
    NAT["National = sum of provinces<br/>sample by sample"]

    SNAP --> PAIR
    PAIR --> DEL
    DEL --> G
    G --> MU
    SNAP --> MU
    TRI --> RETRO
    RETRO --> OBS
    MU --> X
    OBS --> X
    X --> W
    SIG --> W
    W --> NAT
```

`randomwalk` uses the same delay only to find its start week.

```mermaid
flowchart TD
    subgraph TRI["Reporting triangle, per place p"]
        SNAP["Daily onsets C<sub>p,r</sub>(t)<br/>by onset day and release r"]
        PAIR["Release pairs (r, r + 1)<br/>gap s ≤ D = 42 days"]
    end
    DEL["Daily delay Δ<sub>p</sub>(j)<br/>and growth to final Γ<sub>p</sub>(a)"]
    PI["Share reported π<sub>p</sub>(w)<br/>mean of 1 / Γ<sub>p</sub>(c<sub>o</sub> − t)"]
    W0["Start week w<sub>0</sub><br/>latest with π<sub>p</sub>(w) ≥ 0.95"]
    Y["Reported Y<sub>p,o</sub>(w<sub>0</sub>)"]
    SIG["Walk spread σ<sub>p</sub><br/>changes in log(1 + Y<sub>p,o</sub>)"]
    W["Walk on log(1 + λ<sub>i</sub>)<br/>from Y<sub>p,o</sub>(w<sub>0</sub>)<br/>places with no recent case 0<br/>until an introduction<br/>Poisson counts<br/>floored at Y<sub>p,o</sub>(w)<br/>in weeks with data"]
    NAT["National = sum of provinces<br/>sample by sample"]

    SNAP --> PAIR
    PAIR --> DEL
    DEL --> PI
    PI --> W0
    W0 --> Y
    SNAP --> Y
    Y --> W
    SIG --> W
    W --> NAT
```

## Daily release snapshots

Both models use the daily onsets of each hub release ([Data](data.md#release-snapshots)).
$C_{p,r}(t)$ is the known plus imputed onsets in place $p$ on day $t$ at release $r$.
$Y_{p,r}(w)$ is their sum over the days of week $w$.
A forecast is made from the vintage of its date, whose latest release is $o$.
Release $r$ has cut-off $c_r$, its latest onset day ([Data](data.md#vintages)).
Day $t \le c_r$ is observed at $r$ and is $a = c_r - t$ days old.
Days after the cut-off are not observed.
$w$ is $k$ weeks back when it is $k$ weeks before the week holding the release cut-off.

A pair of consecutive releases $(r, r + 1)$ has gap $s = c_{r+1} - c_r$ days.
$D = 42$ days is baselinenowcast's maximum delay, and a pair is in window when $1 \le s \le D$.
Only in-window pairs inform the delay.
A pair outside the window keeps its counts in the data.
A model raises if the vintage has no snapshots or no in-window pair.

## Delay

The delay follows baselinenowcast[^bnc-paper][^bnc-package] on the daily snapshots, with one extension.
Releases are batched reports of one daily delay.
With $\Delta_p(j) \ge 0$ the daily log development factor at age $j$, and $\Delta_p(j) = 0$ for $j \ge D$, a day of age $x$ at release $r$ grows to release $r + 1$ by $\exp \sum_{j=x}^{x+s-1} \Delta_p(j)$.
For each gap $s$ and age $x = 0, \dots, D - 1$,

$$
\log \frac{\sum C_{p,r+1}(t)}{\sum C_{p,r}(t)} = \sum_{j=x}^{x+s-1} \Delta_p(j),
$$

summed over the in-window pairs of gap $s$ and over days $t$ of age $x$ at $r$ that both releases observe.
$\Delta_p$ is the non-negative least squares fit of all these equations, each weighted by its denominator $\sum C_{p,r}(t)$, with no smoothing over age.
With a gap of 1 only, $\exp \Delta_p(x)$ is baselinenowcast's chain ladder factor.
Cells a release does not observe are skipped, not filled, and a gap and age with a zero denominator give no equation.
The parent of a province is the country, whose counts are the sum of its provinces.
Only onset days less than $1.5 D = 63$ days before the cut-off of the latest release used contribute to the sums, as in baselinenowcast's default, and a retrospective fit uses the cut-off of its own release.
Before the sums, each day's increments between its releases go through baselinenowcast's handling of negative values, so a downward revision is set to zero and taken from the earlier increments of that day and no ratio falls below 1.
This applies to the delay only, because the removals below still allow negative revisions.
The preprocessing runs along releases for each onset day, where baselinenowcast runs it along the delay columns of a reporting triangle.

The growth of a day's count from age $a$ to final is

$$
\Gamma_p(a) = \exp \sum_{j=a}^{D-1} \Delta_p(j),
\qquad
\Gamma_p(a) = 1 \text{ for } a \ge D,
$$

and $1 / \Gamma_p(a)$ is the share of it reported.
There is no growth past $D$, so an equation whose ages run past $D$ puts that growth on its ages below $D$.

## Nowcast

Each hub week is a reference time of a weekly reporting triangle, with a column every 7 days after $c_o$ and a last one at $c_o + D$, where $F = 1$.
The share of week $w$ reported by column $j$ is

$$
F_{p,j}(w) = \frac{1}{n_w} \sum_{t \in w,\ t \le c_o} \frac{1}{\Gamma_p(c_o - t + 7j)},
$$

with $n_w$ the days of $w$ observed at $o$.
baselinenowcast fills the row from $x_0 = Y_{p,o}(w)$, adding at each column

$$
x_j = x_{j-1} + \frac{x_{j-1} + 1 - F_{p,j-1}(w)}{F_{p,j-1}(w)} \big(F_{p,j}(w) - F_{p,j-1}(w)\big),
$$

and the last column is the nowcast mean $\mu_p(w)$.
With $\text{rem} = \mu_p(w) - Y_{p,o}(w) \ge 0$ the count still to come, the final count is new reports minus removals, as in epinowcast,

$$
X_{p,i}(w) = \max\big\{0,\ Y_{p,o}(w) + A - B\big\},
\qquad
A \sim \text{NB}\big(\text{rem},\ \phi_{p,k}\big),
\qquad
B \sim \text{Poisson}\big(b_{p,k}\, Y_{p,o}(w)\big).
$$

$A$ is baselinenowcast's negative binomial on the remaining count, with mean $\text{rem}$ and size $\phi_{p,k}$.
$B$ removes a share $b_{p,k}$ of the count reported, so counts can be revised down.
Before the floor the mean is $\mu_p(w) - b_{p,k} Y_{p,o}(w)$.
Weeks with no day younger than $D$ at $o$ have $\text{rem} = 0$ and $b_{p,k} = 0$, so they keep their reported count.
Sample $i$ is one draw of $X_{p,i}(w)$ per week.

$\phi_{p,k}$ and $b_{p,k}$ are fitted as follows.

1. Take each earlier release $r < o$ with an in-window pair among the releases up to $r$, and a cut-off at most $3D - 1.5D = 63$ days before $c_o$, the reference times baselinenowcast leaves for its retrospective nowcasts.
2. Nowcast from the data up to $r$ with the delay of $r$, $\hat y_p(w)$.
   It is the same weekly fill from $c_r$, with the shares taken relative to the count at $c_o$, so its last column is at $c_o$.
3. For weeks back $k$ with a day younger than $D$ at $r$, and where $Y_{p,r}(w) > 0$ or $\hat y_p(w) > 0$, compare the count of $w$ on days up to $c_r$ between releases $r$ and $o$.
   A rise is the new reports $A_{p,r}(w)$ and a fall the removals $B_{p,r}(w)$.
4. With $\widehat{\text{rem}} = \hat y_p(w) - Y_{p,r}(w)$, fit

    $$
    \phi_{p,k} = \arg\max_\phi \sum \log \text{NB}\big(A_{p,r}(w);\ \widehat{\text{rem}},\ \phi\big),
    \qquad
    b_{p,k} = \frac{\sum B_{p,r}(w)}{\sum Y_{p,r}(w)},
    $$

summed over the kept releases.
$\phi_{p,k}$ is fitted by maximum likelihood on $[0.1, 1000]$, one size per $k$, as baselinenowcast does.
A week with $\widehat{\text{rem}} = 0$ says nothing about $\phi$, so it is left out of that fit.
The weeks in the fit of $\phi_{p,k}$ are the errors at $k$.
$b_{p,k}$ is the Poisson maximum likelihood estimate of the removals.
Where the estimate chosen by the [pooling rule](#own-or-pooled-estimates) has fewer than 3 errors at $k$, $\phi_{p,k} \approx 2.2$, the median of epinowcast's prior $1 / \sqrt{\phi} \sim \mathcal{N}^+(0, 1)$.

## Own or pooled estimates

The places form a hierarchy of zone, province and country.
Each estimate is chosen as follows.

1. **Delay.** At each age $x$, a place uses its own $\Delta_p(x)$ if its denominator $\sum C_{p,r}(t)$ at that age, summed over the pairs, is at least 10 cases.
   Otherwise it uses its parent's.
   The country always uses its own.
2. **Level.** The place's level $\ell_p$ is the mean of $Y_{p,o}(w)$ over weeks 3 to 6 back.
3. **Walk spread.** A place with $\ell_p \ge 10$ cases a week and an own $\sigma_p^2 > 0$ ([below](#forecast)) uses its own $\sigma_p$.
4. **Size.** A place with $\ell_p \ge 10$ and at least 3 errors at $k$ uses its own $\phi_{p,k}$.
5. **Fallback.** A place that fails step 3 or 4 uses the pooled estimate of its nearest parent that meets the same rule, and otherwise the country's.
   A pooled estimate uses together the errors and weekly changes of the parent's children with $\ell \ge 10$, or the parent's own where none has.
6. **Removals.** The removal rate $b_{p,k}$ is always the place's own.

## Forecast

Both models forecast with a random walk on $\log(1 + \lambda)$ and Poisson counts.
From a start value $x_i$ in sample $i$, for $h \ge 1$ weeks after the start,

$$
\log\{1 + \lambda_i(h)\} = \max\Big(0,\ \log\{1 + x_i\} + \sigma_p \sum_{j=1}^{h} z_{i,j}\Big),
\qquad
z_{i,j} \sim \mathcal{N}(0, 1),
\qquad
\tilde y_i(h) \sim \text{Poisson}\big(\lambda_i(h)\big).
$$

Steps are summed within a sample, so each sample is one trajectory.
`randomwalk` starts from the latest complete week ([below](#start-from-the-latest-complete-week)).
`randomwalknowcast` starts from each sample's completed week holding $c_o$ ([below](#start-from-the-nowcast)).

$\sigma_p$ comes from the weekly changes $\Delta_w = \log\{1 + Y_{p,o}(w)\} - \log\{1 + Y_{p,o}(w - 1)\}$ over the 9 weeks ending at the latest complete week, so up to 8 changes.
Each change carries Poisson noise, and the walk draws Poisson counts on top, so the noise is taken out of the spread:

$$
\sigma_p^2 = s^2_\Delta - \overline{v\{Y_{p,o}(w - 1)\} + v\{Y_{p,o}(w)\}},
\qquad
v(y) = \frac{y}{(1 + y)^2},
$$

where $s^2_\Delta$ is the sample variance of the changes and the bar a mean over the changes.
$v$ is the delta-method variance of $\log(1 + Y)$ for $Y \sim \text{Poisson}(\lambda)$, $\lambda / (1 + \lambda)^2$, with the reported count for $\lambda$.
The noise matters only for small places. At 10 cases a week the term for a change is about 0.17, and at 250 it is under 0.01.
At a few cases a week the delta method understates the noise, so some of it stays in $\sigma_p$.
A place whose $\sigma_p^2$ is 0 or below, or which has fewer than 2 changes, has no own estimate, and takes its parent's by the [own-or-pooled rule](#own-or-pooled-estimates).
If no parent nor the country has a variance above 0, the model gives no forecast for that origin.
A pooled $\sigma_p$ uses the changes and noise of all the places in the pool together.

A place with no case in the 9 weeks to $c_o$ stays at 0 until a new introduction, which starts its walk from a Poisson(1) count given at least 1.
An introduction occurs each week after the latest complete week with probability $q_p$, times the share of the week not yet reported, where $q_p$ is the share of weeks with a case after 9 weeks without one in the place and its siblings.
Without this rule a silent place would walk from a nowcast near 0, with baselinenowcast's $+1$ and a pooled spread, and so forecast a few cases every week.
Most zones and several provinces are silent at any origin, and their truth is almost always 0, so the rule keeps them at exactly 0 unless a case is likely from what their siblings show.

## Start from the latest complete week

`randomwalk` makes no nowcast.
Its start week $w_0$ is the latest week with

$$
\pi_p(w) = \frac{1}{7} \sum_{t \in w,\ t \le c_o} \frac{1}{\Gamma_p(c_o - t)} \ge 0.95,
$$

the estimated share of the week's final count reported, under the delay above.
If any place has no such week, `randomwalk` gives no forecast for the vintage.
$x_i = Y_{p,o}(w_0)$ for every sample.
Weeks up to $w_0$ are the reported counts.
Later weeks with data are walk draws, floored at the reported count.
Where no week reaches 0.95, `randomwalknowcast` takes the first week as the latest complete week.

## Start from the nowcast

`randomwalknowcast` walks on from each sample's own completed week holding $c_o$.

1. Weeks before the one holding $c_o$ are the nowcast draws $X_{p,i}(w)$.
2. If the week holding $c_o$ is full, its value in sample $i$ is its nowcast draw.
3. If it is partial, its nowcast covers its days up to $c_o$, and its later days add $\text{Poisson}(\lambda_i \cdot m / 7)$, with $m$ the missing days.
   $\lambda_i$ is the walk's rate for that week, walked by sample $i$ from its nowcast $X_{p,i}(w_0)$ of the start week $w_0$, the latest full week that is at least 5% reported and has at least 5 cases.
4. The weeks after it are the walk's draws from the sample's completed value of that week, $\log(1 + y)$ plus the walk's steps.
5. A place with no case in the 9 weeks to $c_o$, or with no start week, starts at the latest complete week instead.
   After it the place stays at 0 until an introduction, as above, floored at the counts reported.

## Health zones

`randomwalk` and `randomwalknowcast` run on each zone's own release snapshots in the same way as for provinces.
Each zone takes its delay, size and walk spread by the rules [above](#own-or-pooled-estimates).
Province sample $Y^{(i)}(w)$ is apportioned over the zones in proportion to the zone model's own sample $x^{(i)}_z(w)$ of the same id and week:

$$
y^{(i)}_z(w) \approx Y^{(i)}(w)\, \frac{x^{(i)}_z(w)}{\sum_{z'} x^{(i)}_{z'}(w)}.
$$

Largest remainders make the values whole numbers that sum to $Y^{(i)}(w)$, so zone samples sum to their province's samples ([Renewal models](model.md#health-zones)).
Where every $x^{(i)}_z(w)$ is 0 the split uses the Dirichlet mean shares of the [Renewal models](model.md#split-of-province-samples) page.
Zones are reconciled because their own walks are noisy and do not sum to their province's, which pools more data.
The split keeps each zone's share from its own sample, and the province sets the total, so the zone, province and national forecasts describe one set of trajectories.

## Partially pooled variant

`pprandomwalknowcast` is `randomwalknowcast` with partially pooled point estimates.
Its delay $\Delta_p$, walk spread $\sigma_p$ and nowcast size $\phi_{p,k}$ come from one hierarchical numpyro model, which pools each place towards its parent, in place of the [own-or-pooled rule](#own-or-pooled-estimates).
The model only estimates them.
Their posterior medians are plugged into `randomwalknowcast`'s nowcast and walk, so the forecasts do not carry their posterior uncertainty.

**The same as `randomwalknowcast`.**
The release snapshots, delay window and preprocessing, the weekly nowcast fill, the retrospective errors, the start rule, the nowcast draws, the walk, the introductions and the zone reconciliation.

**Different.**

| Estimate | `randomwalknowcast` | `pprandomwalknowcast` |
|---|---|---|
| Delay $\Delta_p$ | Least squares on the log ratios of sums, own or pooled parent's | Poisson increments, below, pooled over the hierarchy |
| Walk spread $\sigma_p$ | Variance of the changes net of Poisson noise, own or pooled parent's | Normal changes with variance $\sigma_p^2$ plus the Poisson noise, pooled |
| Size $\phi_{p,k}$ | Maximum likelihood per $k$, own or pooled parent's | Negative binomial new reports, pooled |
| Which places enter | Each place's own data, or its pool's | Only the places forecast; their parents are latent centres |

Each parameter $\theta$ of place $p$ is its parent's plus an offset for the place's tier,

$$
\theta_p = \theta_{\mathrm{pa}(p)} + \tau_{v(p)}\, z_p,
\qquad
z_p \sim \mathcal{N}(0, 1),
$$

with $v(p)$ the tier of $p$ (province or zone) and one $\tau_v$ for each tier and parameter.
$\theta$ is $\log \Delta_p(j)$ for each age $j$, $\log \sigma_p$, and $\log \phi_{p,k}$ for each week back $k$.
A place's counts are the sum of its children's, so only the places forecast (the provinces, or the zones at zone level) enter the likelihood, and each case enters once.

- **Delay.** For each gap $s$ and age $x$ with $\sum C_{p,r}(t) > 0$, the increment is Poisson,

    $$
    \sum C_{p,r+1}(t) - \sum C_{p,r}(t) \sim \text{Poisson}\Big(\sum C_{p,r}(t)\, \big(e^{\sum_{j=x}^{x+s-1} \Delta_p(j)} - 1\big)\Big),
    $$

  with the sums, window and preprocessing of the [delay](#delay).
  With one place and a gap of 1 its maximum likelihood estimate is baselinenowcast's chain ladder factor.
  It is fitted at the origin and at each earlier release that gives errors.
- **Walk spread.** The weekly changes $\Delta_w$ of $\log\{1 + Y_{p,o}(w)\}$ over the [forecast](#forecast) window are $\mathcal{N}\big(0, \sigma_p^2 + v\{Y_{p,o}(w - 1)\} + v\{Y_{p,o}(w)\}\big)$.
  A change between two weeks with no cases has no noise, so its density grows without bound as $\sigma_p$ falls to 0, and it is left out.
- **Nowcast uncertainty.** The new reports $A_{p,r}(w)$ are $\text{NB}(\widehat{\text{rem}}, \phi_{p,k})$ on the errors of [the fit](#nowcast), made with the pooled delay of each earlier release.

**Priors and sampler.**

| Parameter | Prior of the country's value | Why |
|---|---|---|
| $\log \Delta(j)$ | $\mathcal{N}(-3, 2^2)$ | Daily log growth about 0.05, 0.007 to 0.37 within one sd |
| $\log \sigma$ | $\mathcal{N}(\log 0.2, 1)$ | Weekly log changes of about 0.2, 0.07 to 0.54 within one sd |
| $\log \phi_k$ | $\mathcal{N}(0, 1.5^2)$ | Sizes about 1, 0.2 to 4.5 within one sd |
| $\tau_v$ | $\mathcal{N}^+(0, 0.5^2)$ | Median offset sd 0.34, a place within a factor of about 1.4 of its parent |

Each fit runs NUTS with 2 chains of 500 warm-up and 500 draws, in sequence, from the prior medians, at target acceptance 0.9.
A fit whose worst split $\hat R$ is above 1.05, or with more than 1% of its draws divergent, raises, and the model gives no forecast for that origin.
The posterior medians of $\phi_{p,k}$ are kept in $[0.1, 1000]$, the range of baselinenowcast's fit.
It is scored on the real releases at province and zone level.

**Open.** Each forecast sample could instead take its own posterior draw of $(\Delta, \sigma, \phi)$, which would make the forecast carry the uncertainty of the estimates.
That choice is not made yet.

## Limitations

While the release history in the delay window is shorter than $D$, no chain to the final count can be checked against a final count.
Each delay factor rests on one onset day per age per pair, and the youngest ages rest on few cases even nationally.
The cut-off is the latest onset day, which can be earlier than the true last report day.
The chain ladder treats reporting speed as fixed by age and step, but speed and backlogs vary between releases.
The observation model is learnt from the earlier releases, which may be few.
Counts include the hub's imputed onsets, which are re-drawn as cases gain onsets, so changes between releases mix reporting with imputation.
Places are sampled independently, so a shock to a whole release is not shared, and the national band is a sum of independent provincial bands.
The walk has no drift, and its spread comes from 8 weekly changes, net of a delta-method Poisson noise term that assumes no extra dispersion in the reported counts.
`pprandomwalknowcast` plugs in posterior medians, so the uncertainty in its pooled estimates does not widen its forecasts.

[^bnc-paper]: Johnson KE, Tang M, Tyszka E, Nemcova B, Wolffram D, Ergas R, Reich NG, Funk S, Mellor J, Bracher J, Abbott S (2025). Baseline nowcasting methods for handling delays in epidemiological data. Wellcome Open Research. [doi:10.12688/wellcomeopenres.25027.2](https://doi.org/10.12688/wellcomeopenres.25027.2).
[^bnc-package]: Johnson KE, Tyszka E, Bracher J, Funk S, Abbott S (2025). baselinenowcast: Methods for baseline nowcasting right-truncated epidemiological data. <https://github.com/epinowcast/baselinenowcast/>.
