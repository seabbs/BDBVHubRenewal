# Implementation

How the [models](model.md) are built in code, computed and sampled.
{ .page-subtitle }

## Code

The model extends [PyRenew](https://github.com/CDCgov/PyRenew), pinned to a commit, and runs in JAX with NumPyro.
Its code is in [`bdbvhubrenewal.model`][bdbvhubrenewal.model] and the variants are in [`forecasters.renewal`][bdbvhubrenewal.forecasters.renewal].
The model is configured by `ModelConfig`.

## PyRenew and what is built here

| Component | PyRenew piece | Built here |
|---|---|---|
| Pooled $R_p(t)$ | `BaseSubpopulationInfections`, `WeeklyTemporalProcess` | `NonCentredRandomWalk`, for the frozen seeding weeks, the separate forecast steps and their reversion. |
| Renewal with importation and depletion | `BaseSubpopulationInfections`, subclassed | `coupled_renewal`, called from `_infections_from_rt`, and `Importation`, which gives the mixing matrix as PyRenew's `mobility_matrix_rv` does. |
| Ascertainment $\alpha(s)$ | `IndependentAscertainment` with a temporal process | None. |
| Onset-to-report delay $f_{p,s}(d)$ | None | `HazardDelay`, with province offsets and drifts from `pooled_walk`, as the Rt deviations. |
| Missing onsets $m_p(t)$ | None | `MissingOnsetFraction`, a pooled walk of the same form as $R_p$ with its own parameters. |
| Known and imputed onsets | `BaseObservationProcess` with `temporal_pmf_rv` for the infection-to-onset pmf (the incubation period, a `CensoredPMF` with a random mean and SD), `NegativeBinomialNoise`, `compute_prop_already_reported` for the publication lag | `OnsetStreams`. |
| Publication lag | None | A Dirichlet simplex fitted to consecutive releases by `ReleasePairLikelihood`. |
| Delays of known-onset cases | None | `DirichletMultinomialNoise`, a noise model in the form of PyRenew's `CountNoise` for counts by delay given weekly totals. |
| BVDOutbreakSize Rt | None | `RtPseudoObservation`, a `RandomVariable` the latent process calls, as it holds both $R$ and infections. |

PyRenew pieces that give a different model, sample less well, or would only rename one line, are not used.

- The `RandomWalk` temporal process puts the step SD in the step distribution, so it is centred in the SD.
- `ARProcess` gives the same walk, log density and gradient, but NUTS needs more tree steps with it, so the walk samples its steps as one vectorised site.
- `AR1` draws its first state from the stationary distribution, and `DifferencedAR1` is AR(1) on the differences.
  Here walks start at a fixed value and stay there over the seeding weeks.
- `SubpopulationInfections` samples one deviation process per province and subtracts their mean, which leaves a direction the data do not identify.
  Here the $P_s - 1$ coordinates on the sum-to-zero basis $Q$ of the seeded provinces are sampled.
- `HierarchicalNormalPrior` has no sum-to-zero constraint.
  On the basis coordinates it is the same prior, but it would replace one sample statement.
- `GammaGroupSdPrior` gives each province its own SD with a floor, where here the provinces share one half-normal SD.
- `CountObservation` and `SubpopulationCounts` take one fixed delay pmf and one truncation pmf.
  Here the delay varies with onset day, and known and imputed onsets are separate streams.
- `MobilitySubpopulationInfections` uses a fixed mobility matrix, $R$ by location and depletion capped at the susceptibles.
  Here a share of transmission is sent from each origin through a fixed gravity kernel, and depletion is exponential in the pool $\nu$.

PyRenew has no delay that varies with time, no Dirichlet-multinomial count and no pseudo-observation of $R$.

## Baseline models

The baselines are in [`forecasters.randomwalk`][bdbvhubrenewal.forecasters.randomwalk] and [`forecasters.chain`][bdbvhubrenewal.forecasters.chain], on a JAX port of baselinenowcast 0.2.2 in [`bdbvhubrenewal.baselinenowcast`][bdbvhubrenewal.baselinenowcast].
The port is checked against fixtures from the R package, and has one named extension, the chain ladder of release snapshots, which skips unobserved cells and fits one daily delay shared by pairs of any gap.
The models use the port's chain ladder on snapshots.
The observation model is in `forecasters.obs`, and fits and draws its new reports with the port's `fit_by_horizon`, `fit_nb` and `sample_nb`.

| Part | Where |
|---|---|
| Snapshots as arrays `[location, release, day]` | `data.snapshots` |
| Pair sums, the daily delay fit and growth | `baselinenowcast.snapshots`, pair sums `jit` with locations on the leading axis |
| Own or pooled source of each estimate | `forecasters.pooling` |
| New reports minus removals | `forecasters.obs` |
| One random walk for both models | `forecasters.walk` |
| Backtest on real releases | `evaluation.nowcast` |

Shapes are fixed within an origin, so each new origin compiles once.
Pandas only builds the inputs and the output frames.

## Numerics

The model runs in single precision (float32), JAX's default.

- Depletion is carried on $\log S$, which needs no subtraction, and $1 - e^{-y}$ is computed with `expm1`.
- The generation interval, the incubation period and the hub's imputation delay are double interval censored in closed form, $G(d + 1) - 2G(d) + G(d - 1)$ with the integrated cdf (`model.censored`).
  The generation interval is computed once and the hub's delay once per release, both in float64, and the incubation period in float32 at each draw of its mean and SD (`CensoredPMF`); all match primarycensored.
- The simulator's delays use the same closed form, truncated at their maximum delay and renormalised as primarycensored's $D$.
  In float64 they are within $10^{-8}$ of primarycensored, and in float32 within $4 \times 10^{-5}$ of numerical integration.
- The negative binomial size is kept below 1000, which is Poisson for these counts, and the Dirichlet-multinomial concentration below $10^4$ ([Likelihood](model.md#likelihood)).
- The hub imputation sums are products of banded $7 \times 7k$ blocks for each onset week, so the cost grows with the number of weeks and not its square.

The tests check the renewal and the imputation against a NumPy reference, `tests/model/numpy_reference.py`.

## Sampling

PyRenew's `PyrenewBuilder` builds each model as a `MultiSignalModel`, and NumPyro's NUTS fits it.
Chains run vectorised unless a fit's settings say otherwise.
Stored backtest fits and CI submission fits run their chains in parallel.
Without other settings, NUTS runs 4 chains of 1000 warm-up and 1000 draws, with target acceptance 0.9 and maximum tree depth 10.
Each chain starts from its own draw from the prior.
Submission fits and the [fit store](../development/cli.md#fit-store)'s backtest fits take each model's settings from its candidate in `ci/policy.toml`.
`renewal`, `renewalsp` and `renewalbvd` use 2 chains of 500 warm-up and 750 draws, at target acceptance 0.9 (0.95 for `renewalbvd`).
`pprandomwalknowcast` has no candidate and uses its own settings ([Baseline models](baselines.md#partially-pooled-variant)).
Leave-one-province-out fits take theirs from `[sampler]` in `config/backtest.toml` when `BDBV_RENEWAL_SAMPLER` is unset.
Each fit reports its largest R-hat over parameters and the share of draws at the maximum tree depth.

The parameterisation is as follows.

- Every walk, the pooled $\log R(t)$ included, samples its steps non-centred, as standard normals scaled by the step SD.
- Province deviations sum to zero through an orthonormal basis $Q$ of the sum-to-zero subspace of the seeded provinces, so the $P_s - 1$ free processes are unconstrained.
- The starting offsets of the $\log R$ deviations are sampled centred, and those of the missing-onset fraction non-centred.
- Walk steps after the fit axis are not sampled by NUTS. They are drawn from the prior given each posterior draw when forecasting.
