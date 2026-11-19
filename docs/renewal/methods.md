# Methods

How the renewal models used for the hub's nowcasts and forecasts work, from data to submission.
{ .page-subtitle }

This page describes the methods in prose, in the order the data meet them.
The full equations of each part are on the [Renewal models](../methods/model.md) page, and the code is in [`forecasters/zonerenewal.py`](https://github.com/seabbs/bdbv-hub-modelling/blob/main/src/bdbvhubrenewal/forecasters/zonerenewal.py) and [`model/`](https://github.com/seabbs/bdbv-hub-modelling/tree/main/src/bdbvhubrenewal/model).

## Data and levels

The hub's target is the weekly count of confirmed Bundibugyo virus disease cases by symptom onset week, for the country, its 26 provinces and its health zones ([Data methods](../methods/data.md#target-and-notation)).
Each weekly release revises recent weeks, so each forecast is made from one release and scored on what later releases report.
From each release we use the daily onsets of every health zone, split into cases with a known onset and cases the hub gave an imputed onset.
We also use the release's onset-to-report delays of cases with a known onset, which the hub publishes by province only, and earlier releases, which show how counts grow after first publication.
Zone populations and centres come from WorldPop 2025 counts and the zone map behind BVDOutbreakSize's province centres[^worldpop].

We fit health zones, not provinces.
A zone is fitted if it had a case in the walk window of the random walk baselines ([Baseline models](../methods/baselines.md)), which was 55 of the 519 hub zones at the 2026-11-08 release.
Each other zone is forecast by the random walk's introduction layer: zero until an introduction, then its walk.
A province forecast is the sum of its zones' forecasts, draw by draw, and the national forecast is the sum of the provinces.
So the three levels are coherent by construction, and correlation between zones in the same draw carries through to provinces and the country.

## Transmission

Infections in each fitted zone follow a renewal equation[^fraser][^cori].
New infections on a day are the zone's reproduction number times past infections weighted by the generation interval, with a share of transmission from other zones.
The generation interval is BVDOutbreakSize's Gamma distribution (mean 15.3 days, SD 9.3 days), discretised with double interval censoring as in primarycensored[^pcens][^park].
Transmission between zones follows a gravity kernel on zone populations and centroid distances, with a 300 km length scale and a small learnt importation share.
Infections deplete a learnt share of each zone's population at risk, so growth slows as that share is used up.

The log reproduction number of a zone is the sum of three parts.
A shared trend, the mean of the zones' log Rt, follows a weekly random walk.
A province deviation and a zone deviation within its province follow weekly AR(1) processes that decay to zero, each summing to zero over its level.
So zones borrow strength from their province and provinces from the country, and a zone with few cases stays close to its province.
As in BVDOutbreakSize's zone model, only zones with at least 30 cases by the cut-off get a deviation of their own.
The others share their province's.
We call this model `renewal`.

## Observation model and reporting delays

Expected onsets are infections convolved with the incubation period, times a weekly varying ascertainment.
The incubation period is log-normal, with BVDOutbreakSize's priors on its mean and SD[^macneil].
Each onset is reported after a delay from a discrete hazard that is smooth in delay and drifts with onset day.
Each province has its own delay level and drift, shrunk towards the shared delay.
Its zones share it, because the delay data are by province.
A reported case enters a release after a short publication lag, which we learn from how often a row of one release's delay file was already in the previous release.

The hub gives each case without an onset an imputed onset by subtracting a delay drawn from its own fit to the published delays.
We model that procedure rather than treat imputed onsets as observed.
Cases without an onset are a weekly varying share of reports, with province deviations, and the hub's imputation delay spreads them back over earlier days.
So each zone has two data streams by onset week, known onsets and imputed onsets, each right truncated by the release.
Both are negative binomial.
The known-onset delays of each province are Dirichlet-multinomial by onset week.
Because the delay is learnt jointly with the counts, the model corrects recent weeks for cases not yet reported, and the corrected counts are the nowcast.

## BVDOutbreakSize-informed Rt

BVDOutbreakSize fits a larger joint model to more data streams than the hub publishes[^bvd].
We add its Rt estimates to our model as a down-weighted pseudo-observation of our own Rt, a form of approximate Markov melding[^goudie].
We take the newest BVDOutbreakSize release whose files were updated by the forecast's cut-off and whose joint fit's worst R-hat is at most 1.1 ([Data methods](../methods/data.md#bvdoutbreaksize-estimates)).
At that release's data cut-off, our infection-weighted national Rt is matched to the release's national log Rt, and each patch's Rt relative to national is matched to the patch's estimate.
Each fitted zone with a zone estimate in the release gets a term of its own in the same way.
The terms are normal, with the release's variance multiplied by $1/\zeta$ and $\zeta = 0.5$, which tempers the likelihood as in a power prior[^ibrahim].
So the external estimates move our Rt where our data are weak but do not override them.
The term adds no parameters, and $\zeta = 0$ gives `renewal` exactly.
We call the model with this term `renewalbvd`.

Every renewal model also inherits from BVDOutbreakSize the generation interval, the incubation period priors, the prior on the half-life of the province deviations and the sum-to-zero basis for the deviations.
Without a qualifying release `renewalbvd` is `renewal`.

## Single-patch comparison

To see what the joint fit adds, we also fit the same model to one place at a time.
The place is the only patch in its fit.
So it has no shared trend with other places, no deviations, no transmission from other places and no BVDOutbreakSize term.
Its delay and missing-onset share are its own, and its log Rt is a single weekly random walk.
One place's series cannot inform the share at risk, so this model has no depletion.
Its other parts and priors are those of `renewal`.
In code it is the same model builder as the zone fits, with each place as its own patch.
We fit it to each province with a case, and split each province's forecast into its zones by their recent share of cases ([Renewal models](../methods/model.md#health-zones)).
We call this model `renewalsp`.
Comparing it with `renewal` and `renewalbvd` shows what pooling, transmission between places and the external Rt add.
It is also a fallback in the submission chain that does not share the zone fits' failure modes.

## Priors

Every time-varying part is a weekly random walk or AR(1), so it stays near constant unless the data show change.
The main priors are below, and the rest are on the [Renewal models](../methods/model.md) page.

| Quantity | Prior | Source |
|---|---|---|
| Step SD of the shared log Rt walk | $\mathcal{N}^+(0, 0.2^2)$ | weakly informative |
| Start of log Rt | $\mathcal{N}(0, 0.5^2)$ | weakly informative |
| Step SD of the province and zone deviations | $\mathcal{N}^+(0, 0.1^2)$ | weakly informative |
| Half-life of the deviations | $\text{LogNormal}(\log 6, 0.6^2)$ weeks | BVDOutbreakSize |
| Spread of the starting deviations | $\mathcal{N}^+(0, 0.3^2)$ | weakly informative |
| Importation share | $\text{Beta}(1, 100)$ | mostly local transmission |
| Share at risk | $\text{LogNormal}(\log 0.05, 0.75^2)$ | weakly informative |
| Incubation period mean and SD | $\mathcal{N}(6.3, 0.54^2)$ and $\mathcal{N}(3.5, 0.8^2)$ days | BVDOutbreakSize, MacNeil et al.[^macneil] |
| Seed infections per day | $\text{LogNormal}(0, 1.5^2)$ | weakly informative |
| Negative binomial overdispersion | $1/\sqrt{\phi} \sim \text{HalfNormal}(0.25)$ | weakly informative |
| Province delay offsets | $\mathcal{N}^+(0, 0.5^2)$ spread | weakly informative |

## Inference

We fit each model with the No-U-Turn sampler[^nuts] in NumPyro[^numpyro], through PyRenew[^pyrenew].
Each zone fit runs 2 chains of 500 warm-up and 1500 draws, keeping every other draw, with target acceptance 0.9 (0.95 for `renewalbvd`) and maximum tree depth 10 (`ci/policy.toml`).
`renewalsp` runs 2 chains of 500 warm-up and 750 draws per province.
Each chain starts from its own draw from the prior.
The walks are sampled non-centred and anchored at their mean over the fit, which removes a ridge between a walk's level and its first steps.
In `renewalsp` the log Rt walk is the only Rt process, and the data pin its level closely, so we sample its weekly levels directly (centred) rather than its standardised steps (#330).
A fit runs on a 4-core CI runner in 1.4 to 3.1 hours at the origins 2026-10-17 to 2026-11-15 (#185).

Each fit reports, for every parameter, the rank-normalised R-hat and the bulk and tail effective sample sizes[^vehtari].
It also reports, for every chain, the divergent transitions, the share of draws at the maximum tree depth, the E-BFMI and the step size.
A fit with an R-hat above 1.1, a bulk or tail ESS below 25 or more than 5% divergent transitions is ruled out of the submission, and the next model in the chain is used.
An R-hat above 1.05, an ESS below 100 or more than 1% divergent transitions is reported only.
Posterior predictive coverage of the in-sample weeks is reported and never gated.
Each fit writes them to its `diagnostics.json`, which the fit store keeps.

## Forecasting and hub targets

Each of 100 forecast trajectories is one posterior draw.
Past the data, the shared trend in log Rt reverts towards zero with a half-life of 2 weeks, so growth does not compound over the horizon, and the deviations keep their AR(1).
A trajectory is the final count of known onsets, those reported by the cut-off plus a negative binomial draw of those reported later, plus the final imputed onsets.
The hub's target is `wk inc confirmed cases` as 100 samples for 4 weeks before the reference date to 3 weeks after it, for every zone, province and the country ([Hub specification](../submission/hub-spec.md)).
Zone trajectories go into the hub file as they are, and provinces and the country are their sums.
The submission walks a chain of candidates, `renewalbvd`, `renewalsp`, `renewal`, `randomwalknowcast` and `randomwalk`, and submits the first that passes its checks ([Submission workflow](../submission/workflow.md#renewal-chain)).
`renewalsp` is the [single-patch comparison](#single-patch-comparison).

## Limitations

- Fits are slow.
  A zone fit takes 1.5 to 2 times as long as the province fits it replaced, and the time grows with the number of zones fitted.
- The zone fits need two releases at most 42 days apart, because the silent zones take the random walk's introductions.
  At the 2026-10-09 origin they could not forecast.
  Nor could `renewalsp` or the random walks, which need the same pair, so the submission chain has no fallback at such an origin (#365).
- The province fits we replaced saturated the maximum tree depth in most draws, which made their convergence slow and fragile (#320).
  The zone fits do not, but `renewalsp` still does (#330).
- In #185's backtest, scored against each later release, the zone fits beat the province fits at every level.
  On weeks complete at the newest release they were mixed at the province level.
  Both did worse than `randomwalknowcast` at the national level at the two latest origins.
- The delay is by province, so zones in the same province cannot differ in how fast they report.
- Silent zones are forecast by the random walk's introduction layer, independent of the fit, so a new zone's cases do not come from its neighbours' transmission.
- The tempering $\zeta$ is fixed, not learnt or tuned on scores.

[^worldpop]: WorldPop (2025). Global high resolution population denominators. [www.worldpop.org](https://www.worldpop.org).

[^fraser]: Fraser C (2007). Estimating individual and household reproduction numbers in an emerging epidemic. PLoS ONE 2(8):e758. [doi:10.1371/journal.pone.0000758](https://doi.org/10.1371/journal.pone.0000758).

[^cori]: Cori A, Ferguson NM, Fraser C, Cauchemez S (2013). A new framework and software to estimate time-varying reproduction numbers during epidemics. American Journal of Epidemiology 178(9):1505-1512. [doi:10.1093/aje/kwt133](https://doi.org/10.1093/aje/kwt133).

[^pcens]: Abbott S, Brand S, Azam J, Pearson C, Funk S, Charniga K. primarycensored: Primary Event Censored Distributions. [doi:10.5281/zenodo.13632838](https://doi.org/10.5281/zenodo.13632838).

[^park]: Park SW, Akhmetzhanov AR, Charniga K, Cori A, Davies NG, Dushoff J, Funk S, Gostic K, Grenfell B, Linton NM, Lipsitch M, Lison A, Overton CE, Ward T, Abbott S (2024). Estimating epidemiological delay distributions for infectious diseases. [doi:10.1101/2024.01.12.24301247](https://doi.org/10.1101/2024.01.12.24301247).

[^macneil]: MacNeil A, Farnon EC, Wamala J, Okware S, Cannon DL, Reed Z, Towner JS, Tappero JW, Lutwama J, Downing R, Nichol ST, Ksiazek TG, Rollin PE (2010). Proportion of deaths and clinical features in Bundibugyo Ebola virus infection, Uganda. Emerging Infectious Diseases 16(12):1969-1972. [doi:10.3201/eid1612.100627](https://doi.org/10.3201/eid1612.100627).

[^bvd]: BVDOutbreakSize. [github.com/epiforecasts/BVDOutbreakSize](https://github.com/epiforecasts/BVDOutbreakSize), [doi:10.5281/zenodo.20312758](https://doi.org/10.5281/zenodo.20312758).

[^goudie]: Goudie RJB, Presanis AM, Lunn D, De Angelis D, Wernisch L (2019). Joining and splitting models with Markov melding. Bayesian Analysis 14(1):81-109. [doi:10.1214/18-BA1104](https://doi.org/10.1214/18-BA1104).

[^ibrahim]: Ibrahim JG, Chen MH (2000). Power prior distributions for regression models. Statistical Science 15(1):46-60. [doi:10.1214/ss/1009212673](https://doi.org/10.1214/ss/1009212673).

[^nuts]: Hoffman MD, Gelman A (2014). The No-U-Turn sampler: adaptively setting path lengths in Hamiltonian Monte Carlo. Journal of Machine Learning Research 15:1593-1623.

[^numpyro]: Phan D, Pradhan N, Jankowiak M (2019). Composable effects for flexible and accelerated probabilistic programming in NumPyro. [arXiv:1912.11554](https://arxiv.org/abs/1912.11554).

[^pyrenew]: CDC Center for Forecasting and Outbreak Analytics. PyRenew. [github.com/CDCgov/PyRenew](https://github.com/CDCgov/PyRenew).

[^vehtari]: Vehtari A, Gelman A, Simpson D, Carpenter B, Bürkner PC (2021). Rank-normalization, folding, and localization: an improved R-hat for assessing convergence of MCMC. Bayesian Analysis 16(2):667-718. [doi:10.1214/20-BA1221](https://doi.org/10.1214/20-BA1221).
