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
Pairing zone sample $k$ with province sample $k$ is also arbitrary, so the shares in a high province draw can come from a low zone draw.

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
Zambon et al. (2024, IJF, [arXiv:2303.15135](https://arxiv.org/abs/2303.15135)) show that for counts the reconciled mean can move outside the range of the base means and the variance can grow, unlike Gaussian MinT.
Zambon, Azzimonti, Rubattu and Corani (UAI 2024, PMLR 244) add mixed hierarchies (discrete bottom, Gaussian upper).
TD-cond reconciles the upper levels first, then pushes them down to the bottom with a probabilistic top-down step.
As published, BUIS and TD-cond work one time point at a time.
The papers assume independent base forecasts across series *(unverified for the Stat Comput paper)*.
Neither limit is essential: the weight can be any density evaluated at a whole bottom-up trajectory.

**Coherent by construction.**
Olivares et al. (2024, IJF 40(2):470–489, [doi:10.1016/j.ijforecast.2023.04.007](https://doi.org/10.1016/j.ijforecast.2023.04.007)) model the bottom level as a joint Poisson mixture and get every aggregate by summing.
This handles counts and sample paths and is the same idea as fitting at zone level and summing (the zone-renewal and coherent-models prototypes).
Rangapuram et al. (ICML 2021, [PMLR 139:8832–8843](https://proceedings.mlr.press/v139/rangapuram21a.html)) project samples inside training.

**Epidemiology.**
Gibson, Moran, Reich and Osthus (2021, PLoS Comput Biol 17(1):e1007623, [doi:10.1371/journal.pcbi.1007623](https://doi.org/10.1371/journal.pcbi.1007623)) project FluSight regional and national forecast densities to be coherent, with higher skill for 79% of models.
A Sri Lanka dengue study uses MinT over spatial levels ([arXiv:2112.00992](https://arxiv.org/abs/2112.00992)).
HierarNet (AAAI, [link](https://ojs.aaai.org/index.php/AAAI/article/view/41315/45276)) learns provinces, regions and nations jointly *(not read)*.
We found no epinowcast or hub work on sample-level coherence across spatial levels for nowcasts.
The COVID-19 and flu hubs collect state and national forecasts separately and do not enforce coherence.

## Packages

| Package | Methods | Takes our sample matrices | Joint over weeks | Counts | Licence, status |
|---|---|---|---|---|---|
| Nixtla `hierarchicalforecast` 1.5.3 (Sep 2026) | BottomUp, TopDown, MiddleOut, MinTrace (+ sparse, nonnegative QP), ERM; probabilistic Normality, Bootstrap, PERMBU, Conformal | No. Samplers make their own draws from point forecasts plus residuals or `sigmah` | Bootstrap keeps residual blocks of length h; PERMBU and Normality are per step | No integer handling; nonnegative MinTrace cannot be used with Bootstrap or PERMBU | Apache-2.0, active; numpy, pandas, scikit-learn, narwhals, qpsolvers |
| `bayesreconpy` ([JOSS 2025](https://doi.org/10.21105/joss.08336)) | `reconc_buis`, `reconc_mix_cond`, `reconc_td_cond`, `reconc_mcmc`, `reconc_gaussian`, `reconc_mint` | Yes, per series as samples or Gaussian, Poisson or NB parameters | No, one time point per call | Yes (discrete samples, Poisson, NB) | Python port of R `bayesRecon`; Python ≥3.12 per docs; version and licence not checked |
| R `bayesRecon` | Same, plus `reconc_t` | Yes | No | Yes | CRAN, active |
| darts | BottomUp, TopDown, MinT reconciliators | Yes, stochastic `TimeSeries` | Linear $SG$ on each draw and step, so paths are kept | No, outputs reals | Apache-2.0 per docs |
| sktime | `Reconciler`, `ReconcilerForecaster` (bu, ols, wls, MinT shrink, td) | No, point forecasts | No | No | Docs v1.2.0 |
| scikit-hts | BU, TD, MO, OLS | No | No | No | MIT, unmaintained since 2021 |
| `scoringrules` 0.12.0 (Nov 2026) | Energy, variogram, CRPS ensemble, threshold-weighted ES | Yes | Scores any vector | Yes | Apache-2.0; numpy/numba, JAX, PyTorch backends; already a dependency |
| `properscoring` | CRPS ensemble only | n/a | n/a | n/a | Archived Sep 2026 |

No package reconciles count trajectories jointly over weeks.
The conditioning step is about 100 lines of numpy, and `bayesreconpy` is useful as a per-week reference to test it against.

## Recommendation

**1. Trajectory BUIS per province (main recommendation).**
This combines bottom-up zone draws with the province's own direct forecast, and both stay informative.

1. For province $p$ and zone draw $k$, sum the zone trajectories to $u_p^{(k)}(w)$ over all nowcast and forecast weeks.
2. Fit a density $f_p$ to the province direct draws, jointly over weeks: a negative binomial per week joined by a Gaussian copula on $\log(1+y)$ ranks.
   A product of weekly pmfs is simpler but drops the direct forecast's dependence over weeks.
3. Weight $\omega_k \propto f_p(u_p^{(k)})$ and resample whole zone trajectory blocks with these weights.
   Zones then sum to the province exactly, paths and within-province zone dependence are kept, and the province is the bottom-up sum reweighted towards the direct forecast.
4. Weight each province on its own.
   Weighting all 26 at once collapses the weights.
   Per-province resampling assumes provinces are conditionally independent given their zones.
   Restore cross-province dependence by reordering the resampled province blocks to the ranks of the original joint draw (ECC).
5. National is the sum of provinces.
   Optionally repeat step 3 at national level with a national direct density (the national-direct question).
6. Report the effective sample size $1/\sum \omega_k^2$ per province and origin.
   With 100–1000 draws rather than BUIS's 20000, it will collapse where bottom-up and direct disagree.
   Fallbacks are tempering ($\omega^\tau$, $\tau<1$, tuned on backtests) or a mixture with the current top-down allocation.
   A zone model that can draw many more cheap posterior predictive trajectories than we submit would help most.

Where a province has no direct forecast the weights are flat and the method is pure bottom-up.

**2. Keep top-down allocation and fix the pairing (cheap fallback).**
Keep `reconcile_zones` but pair province draw $k$ with the zone block whose bottom-up total ranks the same, rather than with zone draw $k$.
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
