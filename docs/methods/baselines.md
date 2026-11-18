# Baseline models

`randomwalknowcast`, its generative partially pooled variant `pprandomwalknowcast`, and `randomwalk`, their comparison model.
{ .page-subtitle }

`randomwalk` and `randomwalknowcast` are described in full, with their equations, in the [paper's supporting information (Baselines)](../paper/si.html#baselines).
This page summarises them and gives `pprandomwalknowcast`.
The data and notation are on the [Data methods page](data.md#target-and-notation), and the renewal models on the [Renewal models](model.md) page.

`randomwalknowcast` nowcasts the recent weeks with baselinenowcast[^bnc-paper][^bnc-package] and forecasts each nowcast draw on with a random walk.
`randomwalk` is the same walk from the latest complete week, with no nowcast.
Both run on each province and each health zone.
The country is the per-sample sum of the provinces.
Both pair their province draws within each week by a Gaussian copula ([Correlated trajectories](evaluation.md#correlated-trajectories)).

## Daily release snapshots

Both models use the daily onsets of each hub release ([Data](data.md#release-snapshots)).
Only consecutive releases at most $D = 42$ days apart inform the delay, and a pair outside that window keeps its counts in the data.

## Delay

The delay is baselinenowcast's chain ladder fitted on the daily snapshots, by non-negative least squares over release pairs of any gap up to $D$ ([SI](../paper/si.html#baselines)).

## Nowcast

baselinenowcast's weekly fill gives each recent week's nowcast mean.
The final count is the count reported plus negative binomial new reports, whose size is fitted to retrospective nowcasts from earlier releases ([SI](../paper/si.html#baselines)).
A place with fewer than 10 cases a week takes its parent's delay, size and walk spread.

## Forecast

Each sample walks on $\log(1 + \lambda)$ with Poisson counts, with a spread from the last 8 weekly changes net of Poisson noise.
Places with no case in the 9 weeks to the cut-off stay at 0 until an introduction.
`randomwalk` starts from the latest week at least 95% reported, and `randomwalknowcast` from each sample's own nowcast of the cut-off week ([SI](../paper/si.html#baselines)).

## Health zones

Both models run on each zone's own release snapshots as for provinces, and the zone samples are these draws.
No province sample is split over the zones, so zones need not sum to their province.
The hub does not require them to.

## Partially pooled variant

`pprandomwalknowcast` is one generative model of the expected final counts and the reporting delay, in the style of epinowcast[^enw].
It is fitted with NUTS to the release snapshots, and its forecast samples are posterior predictive draws.
It keeps the random walk of `randomwalknowcast`, but makes the nowcast from the same model in place of a chain ladder, and partially pools over zones, provinces and the country in place of the [own-or-pooled rule](#nowcast).

**Places.**
The places forecast are the leaves of the hierarchy: the provinces, or the zones at zone level.
A leaf with no case in the 9 weeks to $c_o$ is silent, as in `randomwalknowcast`.
Silent leaves are not fitted and draw introductions (below).
Every other leaf $p$ enters the likelihood.
A parent's counts are the sum of its children's, so the parents enter only as the centres the leaves are pooled towards, and each case enters once.

**Expected final counts.**
The expected final count of leaf $p$ in hub week $w$ is $\lambda_{p,w}$.
It is a random walk on the log scale over the 16 weeks to the week holding $c_o$,

$$
\log \lambda_{p,w} = \log \lambda_{p,w-1} + \delta_{p,w},
\qquad
\delta_{p,w} \sim \mathcal{N}(0, \sigma_p^2).
$$

Each onset day $t$ in week $w(t)$ expects $\lambda_{p,w(t)} / 7$ cases.

**Delay.**
A case is reported $d$ days after onset, with $d$ log-normal with location $\mu_p$ and scale $s_p$, cut to whole days and capped at $D = 42$.
The share of a day's final count reported by age $a$ is

$$
\pi_p(a) = \frac{F(a + 1;\ \mu_p, s_p)}{F(D + 1;\ \mu_p, s_p)}, \quad 0 \le a < D,
\qquad
\pi_p(a) = 1 \text{ for } a \ge D,
\qquad
\pi_p(-1) = 0,
$$

with $F$ the log-normal CDF.

**Observations.**
Day $t$ is $a_r(t) = c_r - t$ days old at release $r$.
Let $r'$ be the release before $r$, with $a_{r'}(t) = -1$ and $C_{p,r'}(t) = 0$ when $r$ is the first release that observes $t$.
The new reports between the two are

$$
C_{p,r}(t) - C_{p,r'}(t) \sim \text{NB}\Big(\frac{\lambda_{p,w(t)}}{7}\,\big\{\pi_p(a_r(t)) - \pi_p(a_{r'}(t))\big\},\ \phi_p\Big),
$$

for every onset day in the 16 weeks and every release up to $o$ that observes it.
This is epinowcast's reporting triangle, with each release a batch of report days.
It is right truncated by construction: a young day has seen only the start of its delay.
The counts first go through the [delay](#delay)'s handling of negative values, baselinenowcast's, so a downward revision is taken from earlier increments of the day and no increment is negative.
New reports of a day that was already $D$ days old at $r'$ have no probability under the model and are left out, as the chain ladder leaves out growth past $D$.
At 11-08 these are 412 of 10,878 national cases, 220 of them added by the 11-08 release.

**Pooling.**
Every place-level parameter is pooled over the hierarchy: the delay location $\mu_p$ and scale $\log s_p$, the walk spread $\log \sigma_p$, the dispersion $\log \phi_p$ and the walk's first level $\log \lambda_{p,1}$.
Each tier's places are normal about their parents,

$$
\theta_p \sim \mathcal{N}\big(\theta_{\mathrm{pa}(p)},\ \tau_{v(p)}^2\big),
\qquad
\tau_v \sim \mathcal{N}^+(0, c_\theta^2),
$$

with $v(p)$ the tier of $p$ (province or zone) and one $\tau_v$ for each tier and parameter.
The fit's tree holds the fitted leaves and their ancestors only, so no place without data has offsets left to the prior alone.
A silent place takes the delay of its nearest fitted ancestor.

**Priors.**

| Parameter | Prior of the country's value | Tier spread scale $c_\theta$ | Why |
|---|---|---|---|
| $\mu$ | $\mathcal{N}(1.6, 1)$ | 0.5 | Median delay about 5 days, 2 to 13 within one sd |
| $\log s$ | $\mathcal{N}(\log 0.7, 0.5^2)$ | 0.1 | Only a few large places inform it |
| $\log \sigma$ | $\mathcal{N}(\log 0.2, 1)$ | 0.5 | Weekly log changes of about 0.2, 0.07 to 0.54 within one sd |
| $\log \phi$ | $\mathcal{N}(\log 10, 1.5^2)$ | 0.25 | Only a few large places inform it |
| $\log \lambda_{p,1}$ | $\mathcal{N}(2, 3^2)$ | 1.5 | Places range from a few cases a week to hundreds |

**Forecast.**
Each sample takes one posterior draw, and every draw is used once before any is used again.
A week's final count is the sum over its days.
A day up to $c_o$ is its count reported plus an NB draw of the rest, with mean $\lambda_{p,w(t)} / 7 \cdot \{1 - \pi_p(a_o(t))\}$ and size $\phi_p$.
A later day of the week holding $c_o$ is an NB draw with mean $\lambda_{p,w(t)} / 7$.
Later weeks continue the walk from the draw's last $\lambda$ with its $\sigma_p$, and their counts are NB with mean $\lambda$ and size $7 \phi_p$, the sum of seven days.
Weeks before the 16 fitted weeks are the counts reported.
National is the sum of the provinces by sample.
In the zone backtest it also runs on the zones, and its zone draws are scored as they are ([Health zones](#health-zones)).
Elsewhere, as in the hub tables, its province samples are split across the zones as for the renewal models ([Renewal models](model.md#split-of-province-samples)).

**Silent places.**
A silent leaf has no case to fit, and a walk from a $\lambda$ near 0 gave divergent draws.
As in `renewalsp` ([Renewal models](model.md)), it takes `randomwalknowcast`'s draws: its reported counts, then exactly 0 until an introduction, then a walk ([Forecast](#forecast)).
This is the one part of the model that is not drawn from its posterior.

**Scoring.**
On the reported target, the backtest maps its final counts to what a later release shows by the posterior median $\pi_p(a)$ of each place ([Evaluation](evaluation.md#real-releases)).

**Sampler.**
NUTS runs 2 chains of 500 warm-up and 750 draws, in sequence, from the prior medians, at target acceptance 0.9.
Each walk step is sampled as $\delta^*_{p,w}$, normal with mean 0 and scale $\sigma_p^{1/2}$, and set to $\delta_{p,w} = \sqrt{\sigma_p}\, \delta^*_{p,w}$, numpyro's location-scale reparameterisation with centring $\tfrac{1}{2}$.
A large place's counts pin its walk, which suits centred steps, and a place with few cases leaves it to the prior, which suits non-centred ones.
The pooled parameters are non-centred, except the first level, whose tiers are centred, and $\log s$, whose tiers are half centred.
Sampling each tier about its parent's value, not as the country's value plus an offset, cut the median leapfrog steps from 255 to 511 down to 63.
With a tier spread scale of 0.25 for $\log s$, 213, 198 and 62 of 1500 draws were divergent at 10-17, 10-24 and 11-01; with 0.1, 14, 9 and 3, and 1 at 11-08.
So $s$ is pooled only weakly: its tier spread sits near its prior scale.
A fit whose worst split $\hat R$ is above 1.05, or with more than 1% of its draws divergent, raises, and the model gives no forecast for that origin.

**How it differs from `randomwalknowcast`.**

| | `randomwalknowcast` | `pprandomwalknowcast` |
|---|---|---|
| Delay | Chain ladder, non-parametric by age, own or parent's | Log-normal, $\mu_p$ and $s_p$ pooled |
| Nowcast | Chain-ladder fill, with NB new reports fitted to retrospective nowcasts | NB new reports of the same model that fits the delay and the walk |
| Walk | On $\log(1 + y)$ from a nowcast draw, spread from 8 changes net of Poisson noise | On $\log \lambda$ over all 16 weeks, spread pooled, fitted jointly |
| Uncertainty in the estimates | None, point estimates | Each sample is a posterior draw |
| Removals | None, falls are preprocessed out, so a draw is at least the count reported | None, the forecast is at least the count reported |
| Silent places | Introductions | `randomwalknowcast`'s draws, as in `renewalsp` |

## Limitations

While the release history in the delay window is shorter than $D$, no chain to the final count can be checked against a final count.
Each delay factor rests on one onset day per age per pair, and the youngest ages rest on few cases even nationally.
The cut-off is the latest onset day, which can be earlier than the true last report day.
The chain ladder treats reporting speed as fixed by age and step, but speed and backlogs vary between releases.
The observation model is learnt from the earlier releases, which may be few.
Counts include the hub's imputed onsets, which are re-drawn as cases gain onsets, so changes between releases mix reporting with imputation.
Places are sampled independently, so a shock to a whole release is not shared, and the national band is a sum of independent provincial bands.
So it pairs its province draws within each week by a copula fitted to its own errors ([Correlated trajectories](evaluation.md#correlated-trajectories)).
The walk has no drift, and its spread comes from 8 weekly changes, net of a delta-method Poisson noise term that assumes no extra dispersion in the reported counts.
`pprandomwalknowcast`'s delay is log-normal, so it cannot take the shape of an irregular delay as the chain ladder can.

[^bnc-paper]: Johnson KE, Tang M, Tyszka E, Nemcova B, Wolffram D, Ergas R, Reich NG, Funk S, Mellor J, Bracher J, Abbott S (2025). Baseline nowcasting methods for handling delays in epidemiological data. Wellcome Open Research. [doi:10.12688/wellcomeopenres.25027.2](https://doi.org/10.12688/wellcomeopenres.25027.2).
[^enw]: Abbott S, Lison A, Funk S, Pearson C, Gruson H, Guenther F, et al. epinowcast: A Bayesian Framework for Real-Time Infectious Disease Surveillance. R package. [doi:10.5281/zenodo.5637165](https://doi.org/10.5281/zenodo.5637165). <https://github.com/epinowcast/epinowcast/>.
[^bnc-package]: Johnson KE, Tyszka E, Bracher J, Funk S, Abbott S (2025). baselinenowcast: Methods for baseline nowcasting right-truncated epidemiological data. <https://github.com/epinowcast/baselinenowcast/>.
