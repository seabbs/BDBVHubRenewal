# Coherent sample-trajectory forecasts across zones, provinces and national

Literature and package review, 2026-11-10.
Question: how to build province and national forecasts from zone forecasts so that zones sum to provinces and provinces to national in every sample, with calibrated uncertainty at each level and sample paths kept across weeks.
Items marked *(unverified)* come from memory and were not checked against the source.

## Where we start

`reconcile_zones` (`src/bdbvhubrenewal/forecasters/zones.py`) already gives exact, whole-number, path-preserving coherence.
It takes province sample $k$ and splits it over the zones in proportion to zone sample $k$, by largest remainders.
This is a top-down conditional allocation.
Its limit is that zone information never changes the province distribution.
The province forecast is taken as given and the zones only set the shares.
For the random walks the zone and province samples come from two separate `model.forecast` runs (`zone_forecast`), so sample $k$ of one shares no draw with sample $k$ of the other.
The pairing is in effect independent (the runs share a seed, so any correlation is incidental).
So the shares in a high province draw can come from a low zone draw.

## Methods

**Projection (MinT and relatives).**
Wickramasuriya, Athanasopoulos and Hyndman (2019, JASA 114(526):804–819, [doi:10.1080/01621459.2018.1448825](https://doi.org/10.1080/01621459.2018.1448825)) give the minimum-trace projection $\tilde y = S G \hat y$.
Wickramasuriya (2024, JBES 42(1):272–285, [arXiv:2103.11128](https://arxiv.org/abs/2103.11128)) shows MinT is log-score optimal when base forecasts are Gaussian.
Panagiotelis et al. (2023, EJOR 306(2):693–706, [RePEc](https://econpapers.repec.org/RePEc:eee:ejores:v:306:y:2023:i:2:p:693-706)) define reconciliation for samples: project each joint draw with $SG$.
They fit $G$ by minimising the energy or variogram score and show the log score is improper for comparing reconciled with unreconciled forecasts.
Projection works on any sample matrix and keeps paths if applied draw by draw.
It returns real values that can be negative, so it breaks counts, and the zeros that dominate our zones make it worse.
Girolimetto et al. (2024, IJF 40(3):1134–1151, [arXiv:2303.17277](https://arxiv.org/abs/2303.17277)) extend this across time with Gaussian or bootstrap sampling and multi-step residuals, which is the nearest to joint-over-horizons reconciliation in this line.

**Copula and reordering methods.**
Ben Taieb, Taylor and Hyndman (2021, JASA 116(533):27–43, [doi:10.1080/01621459.2020.1736081](https://doi.org/10.1080/01621459.2020.1736081)) build aggregates bottom up by summing reordered child samples (empirical copulas through permutations, PERMBU).
The permutation is taken from ranks of in-sample residuals, then the means are reconciled with MinT.
It handles any marginal, including counts, but restores cross-series dependence at one step only.
Jeon, Panagiotelis and Petropoulos (2019, EJOR 279(2):364–379, [doi:10.1016/j.ejor.2019.05.020](https://doi.org/10.1016/j.ejor.2019.05.020)) reconcile samples after reordering them.
Their finding that ranked (comonotonic) ordering suits temporal hierarchies is *(unverified)*.
The Schaake shuffle and ensemble copula coupling (ECC) do the same reordering with a template from history or from a raw ensemble ([Schefzik et al. 2013, arXiv:1302.7149](https://arxiv.org/abs/1302.7149)).
None of these change the marginals.
They change which draws get summed together, so they set the width of the aggregate.

**Reconciliation by conditioning.**
Corani, Azzimonti and Rubattu (2024, IJF 40(2):457–469, [doi:10.1016/j.ijforecast.2023.04.003](https://doi.org/10.1016/j.ijforecast.2023.04.003)) define the reconciled forecast as the bottom base forecast conditioned on the upper base forecast agreeing with the bottom-up sum.
This gives a proper pmf for counts.
Zambon, Azzimonti and Corani (2024, Stat Comput 34:21, [doi:10.1007/s11222-023-10343-y](https://doi.org/10.1007/s11222-023-10343-y)) make this fast with bottom-up importance sampling (BUIS).
Draw bottom samples, sum them, weight each draw by the upper base density at its sum, and resample, one upper node at a time.
Zambon et al. (2024, IJF, [arXiv:2303.15135](https://arxiv.org/abs/2303.15135)) study the reconciled mean and variance for Gaussian and count forecasts.
That for counts the reconciled variance can exceed the base variance, unlike Gaussian MinT, is *(unverified)*.
Zambon, Azzimonti, Rubattu and Corani (UAI 2024, PMLR 244) add mixed hierarchies (discrete bottom, Gaussian upper).
TD-cond reconciles the upper levels first, then pushes them down to the bottom with a probabilistic top-down step.
The published examples reconcile one time point at a time, and the docs do not say how to handle several horizons.
The papers assume independent base forecasts across series *(unverified for the Stat Comput paper)*.
Neither limit is essential, because the weight can be any density evaluated at a whole bottom-up trajectory.

**Coherent by construction.**
Olivares et al. (2024, IJF 40(2):470–489, [doi:10.1016/j.ijforecast.2023.04.007](https://doi.org/10.1016/j.ijforecast.2023.04.007)) model the bottom level as a joint Poisson mixture and get every aggregate by summing.
This handles counts and sample paths and is the same idea as fitting at zone level and summing (the zone-renewal and coherent-models prototypes).
Rangapuram et al. (ICML 2021, [PMLR 139:8832–8843](https://proceedings.mlr.press/v139/rangapuram21a.html)) project samples inside training.

**Epidemiology.**
Gibson, Moran, Reich and Osthus (2021, PLoS Comput Biol 17(1):e1007623, [doi:10.1371/journal.pcbi.1007623](https://doi.org/10.1371/journal.pcbi.1007623)) project FluSight regional and national forecast densities to be coherent, with higher skill for 79% of models.
A Sri Lanka dengue study uses MinT over spatial levels ([arXiv:2112.00992](https://arxiv.org/abs/2112.00992)).
HierarNet (AAAI, [link](https://ojs.aaai.org/index.php/AAAI/article/view/41315/45276)) learns provinces, regions and nations jointly *(not read)*.
We found no epinowcast or hub work on sample-level coherence across spatial levels for nowcasts.
We found no coherence enforcement in the COVID-19 or flu hubs, which collect state and national forecasts separately.

## Summary of methods

| Method | Counts | Joint over weeks | Compute | Fit to our setting |
|---|---|---|---|---|
| MinT / projection of samples | No (reals, negatives) | Yes if applied per draw | Needs a covariance of 500+ series; cheap once estimated | Poor: breaks counts and zeros |
| Score-optimised projection (Panagiotelis) | No | Yes | Stochastic gradient over weights | Poor, same reasons |
| PERMBU / empirical copula (Ben Taieb) | Yes | No, one permutation per step | Sorting, cheap | Partial: aggregation idea is useful, needs residual history we lack |
| Reordering (Jeon, Schaake, ECC) | Yes | Yes if ranked on whole paths | Sorting, cheap | Good as a pairing fix (method 2) |
| Conditioning, BUIS (Zambon, Corani) | Yes | Not as published, extends directly | One density evaluation per draw per province | Good: method 1 |
| TD-cond (mixed hierarchies) | Bottom yes, upper Gaussian | No | Cheap | Partial: Gaussian upper fits provinces with many cases only |
| Coherent by construction (Olivares; zone models) | Yes | Yes | A zone-level fit | Good, but is model building rather than reconciliation |

## Packages

| Package | Methods | Takes our sample matrices | Joint over weeks | Counts | Licence, status |
|---|---|---|---|---|---|
| Nixtla `hierarchicalforecast` 1.5.3 (Sep 2026) | BottomUp, TopDown, MiddleOut, MinTrace (+ sparse, nonnegative QP), ERM; probabilistic Normality, Bootstrap, PERMBU, Conformal | No. Samplers make their own draws from point forecasts plus residuals or `sigmah` | Bootstrap keeps residual blocks of length h; PERMBU and Normality are per step | No integer handling; nonnegative MinTrace cannot be used with Bootstrap or PERMBU | Apache-2.0, active; numpy, pandas, scikit-learn, narwhals, qpsolvers |
| `bayesreconpy` ([JOSS 2025](https://doi.org/10.21105/joss.08336)) | `reconc_buis`, `reconc_mix_cond`, `reconc_td_cond`, `reconc_mcmc`, `reconc_gaussian`, `reconc_mint` | Yes, per series as samples or Gaussian, Poisson or NB parameters | Not documented. Stacking zone-by-week as bottom nodes and province-by-week as upper nodes in `A` should give the product-of-weekly-pmfs version of method 1 *(untested)* | Yes (discrete samples, Poisson, NB) | Python port of R `bayesRecon`; Python ≥3.12 per docs; version and licence not checked |
| R `bayesRecon` | Same, plus `reconc_t` | Yes | No | Yes | CRAN, active |
| darts | BottomUp, TopDown, MinT reconciliators | Yes, stochastic `TimeSeries` | Linear $SG$ on each draw and step, so paths are kept | No, outputs reals | Apache-2.0 per docs |
| sktime | `Reconciler`, `ReconcilerForecaster` (bu, ols, wls, MinT shrink, td) | No, point forecasts | No | No | Docs v1.2.0 |
| scikit-hts | BU, TD, MO, OLS | No | No | No | MIT, unmaintained since 2021 |
| `scoringrules` 0.12.0 (Nov 2026) | Energy, variogram, CRPS ensemble, threshold-weighted ES | Yes | Scores any vector | Yes | Apache-2.0; numpy/numba, JAX, PyTorch backends; already a dependency |
| `properscoring` | CRPS ensemble only | n/a | n/a | n/a | Archived Sep 2026 |

We found no package that reconciles count trajectories jointly over weeks out of the box.
The conditioning step is about 100 lines of numpy, and `bayesreconpy` is a reference to test it against.

## Recommendation

**1. Trajectory BUIS per province (main recommendation).**
This combines bottom-up zone draws with the province's own direct forecast, and both stay informative.

1. For province $p$ and zone draw $k$, sum the zone trajectories to $u_p^{(k)}(w)$ over all nowcast and forecast weeks.
2. Fit a density $f_p$ to the province direct draws.
   The default is a product of weekly negative binomial pmfs.
   It drops the direct forecast's dependence over weeks, but the zone draws carry their own.
   A Gaussian copula with these margins keeps that dependence, but a discrete copula pmf needs rectangle probabilities, $2^W$ terms per trajectory, which is feasible at $W \approx 6$.
   A multivariate normal on $\log(1+y)$ is a cheaper continuous approximation.
3. Weight $\omega_k \propto f_p(u_p^{(k)})$ and resample whole zone trajectory blocks with these weights.
   Zones then sum to the province exactly, paths and within-province zone dependence are kept, and the province is the bottom-up sum reweighted towards the direct forecast.
4. Weight each province on its own.
   Weighting all 26 at once collapses the weights.
   Per-province resampling assumes provinces are conditionally independent given their zones.
   Restore cross-province dependence by reordering the resampled province blocks to the ranks of the original joint draw (ECC).
   Rank on a scalar summary of each block, such as its total over all weeks, so whole paths move together.
5. National is the sum of provinces.
   Optionally repeat step 3 at national level with a national direct density (the national-direct question).
6. Temper the weights, $\omega_k \propto f_p(u_p^{(k)})^\tau$ with $\tau \le 1$, and tune $\tau$ on backtest coverage per level.
   The zone and province forecasts are fitted to the same counts, so full conditioning counts that data twice and will make intervals too narrow, most where the two agree.
   $\tau$ is the calibration control.
7. Report the effective sample size $1/\sum_k \omega_k^2$ (normalised weights) per province and origin.
   With 100–1000 draws rather than BUIS's 20000, it will collapse where bottom-up and direct disagree.
   Lower $\tau$ also helps here, as does a mixture with the current top-down allocation.
   A zone model that can draw many more cheap trajectories than we submit would help most.

Where a province has no direct forecast the weights are flat and the method is pure bottom-up.

**2. Keep top-down allocation and fix the pairing (cheap fallback).**
Keep `reconcile_zones` but pair province draw $k$ with the zone block whose bottom-up total over all weeks ranks the same, rather than with zone draw $k$.
Ranking on the total, not week by week, keeps the paths.
This is ECC or Jeon-style ranked reordering.
It keeps the province marginal as calibrated by its own model and makes the zone shares consistent with the province level.
It does not let zones inform the province.

## What to measure

Compare four forecasts on the same backtest origins: pure bottom-up, current `reconcile_zones`, the province direct forecast alone (national as sum), and each recommended method.

- Energy score of $\log(1+y)$ over places by weeks per scale, as `energy_scores` already does, and over the stacked zone, province and national vector.
- Variogram score (order 0.5, `scoringrules.vs_ensemble`) over the same vectors, because the energy score is weak at detecting misspecified dependence ([Scheuerer and Hamill 2015, doi:10.1175/MWR-D-14-00269.1](https://doi.org/10.1175/MWR-D-14-00269.1)).
- WIS, and 50% and 90% coverage, per level (zone, province, national) and per week horizon.
- ESS per province and origin for method 1.
- A coherence check (`check_zone_coherence`) on every output.

## Prototype on randomwalknowcast

Scripts: `draws.py` draws the samples and `buis.py` reconciles and scores them.
Tables are in `results/`.
Origins are 2026-10-17, 10-24 and 11-01, scored on later releases in the release backtest, and 11-08 for intervals only.
The direct province forecast has 100 draws.
The zone forecast runs on the zone task with a pool of 2000 draws, and 100 zone blocks are resampled per province.
Only 8 provinces have zones with cases (BU, HU, IT, NK, NU, SK, SU, TO).
The other 18 keep their direct forecast in every method.
In this model zones are independent across provinces, as are the direct provinces, so per-province weighting is the exact joint weighting.
National is the sum.

Methods compared:
- `direct`, the direct province forecast, which is also the province margin of the current `reconcile_zones`.
- `zonesum`, the plain sum of zone draws.
- `buis_joint_t{τ}`, weighting on all 8 weeks with a product of weekly NB pmfs, tempered by τ.
- `buis_summary`, weighting on two summaries, the reference week and the total of the 3 forecast weeks.
- `buis_ess`, the largest τ in {1, 0.5, 0.25, 0.1} with ESS ≥ 200.

**ESS (of 2000)** at τ = 1 for the provinces with cases ranged from 20 (TO, 11-01) to 1549 (HU, 11-08), and was mostly 100–700.
At 100 draws these would be about 1–35, so the larger zone pool matters.
Weighting on the summary or at τ = 0.5 roughly doubles ESS (`results/ess.csv`).

**Scores**, reported target, mean over the 3 scored origins (n = 884 province and 34 national cells):

| Model | Province WIS | Province log WIS | Province 90% | National WIS | National log WIS | National 50% / 90% |
|---|---|---|---|---|---|---|
| direct | 1.93 | 0.033 | 0.96 (90%) | 30.2 | 0.098 | 0.41 / 0.74 |
| zonesum | 3.18 | 0.042 | 0.94 (90%) | 46.7 | 0.153 | 0.32 / 0.62 |
| buis_joint_t1 | 2.49 | 0.039 | 0.93 (90%) | 37.4 | 0.123 | 0.24 / 0.50 |
| buis_joint_t0.5 | 2.47 | 0.038 | 0.93 (90%) | 34.8 | 0.117 | 0.35 / 0.62 |
| buis_summary | 2.59 | 0.038 | 0.93 (90%) | 33.4 | 0.115 | 0.44 / 0.68 |
| buis_ess | 2.57 | 0.039 | 0.93 (90%) | 35.1 | 0.117 | 0.32 / 0.65 |

Log energy and log variogram scores (provinces by weeks, reported target) follow the same order: direct 1.46 / 412, BUIS 1.63–1.71 / 418–437, zonesum 1.84 / 474.
On the complete-week target BUIS and direct are within 2% of each other on both.

**11-08 intervals** (`results/intervals_2026-11-08.csv`), 90% for the week ending 2026-11-28:

| | direct | zonesum | buis_joint_t1 | buis_summary |
|---|---|---|---|---|
| CD-IT | 178–345 | 272–688 | 212–339 | 227–371 |
| CD-NK | 89–974 | 205–888 | 143–392 | 170–486 |
| CD (national) | 361–1279 | 641–1404 | 436–726 | 504–819 |

**What this shows.**
Trajectory BUIS is easy to run, coherent and path-preserving, and it improves on plain zone sums by about 20% in WIS at both levels.
It does not beat the direct province forecast here.
The zone sums run well above the direct forecasts in the forecast weeks, so the two densities overlap only in the zone sums' lower tail.
Conditioning then gives intervals narrower than either input, and national 90% coverage falls to 0.50 at τ = 1.
Tempering and summary weighting recover some width and coverage but not the direct forecast's.
So conditioning cannot fix a biased bottom level.
It is worth using only when the zone model is about as good as the province model on its own.
For randomwalknowcast, keeping the province forecast and allocating down (`reconcile_zones`) is the better choice.
The test to rerun is the same comparison with the zone-level renewal or generative models once they exist.
Three origins and 34 national cells are few, so treat these as indicative.
