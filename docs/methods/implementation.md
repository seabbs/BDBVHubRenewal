# Implementation

How the [models](model.md) are built in code, computed and sampled.
{ .page-subtitle }

## Code

The model extends [PyRenew](https://github.com/CDCgov/PyRenew), pinned to a commit, and runs in JAX with NumPyro.
Its code is in [`bdbvhubrenewal.model`][bdbvhubrenewal.model] and the variants are in [`forecasters.renewal`][bdbvhubrenewal.forecasters.renewal].
The model is configured by `ModelConfig`.
The model diagrams are mermaid blocks in `docs/methods/model.md`.

## PyRenew and what is built here

| Component | PyRenew piece | Built here |
|---|---|---|
| Pooled $R_p(t)$ | `SubpopulationInfections`, `WeeklyTemporalProcess` | `NonCentredRandomWalk`. |
| Renewal with importation and depletion | `SubpopulationInfections`, subclassed | `coupled_renewal`. |
| Ascertainment $\alpha(s)$ | `IndependentAscertainment` with a temporal process | None. |
| Onset-to-report delay $f_s(d)$ | None | `HazardDelay`. |
| Missing onsets $\pi_p(t)$ | None | `MissingOnsetFraction`, a pooled walk of the same form as $R_p$ with its own parameters. |
| Known and imputed onsets | `BaseObservationProcess`, `NegativeBinomialNoise` | `OnsetStreams`. |
| BVDOutbreakSize Rt | None | `RtPseudoObservation`. |
| Provinces with no cases | None | An importation-only component. |

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
- The generation interval is double interval censored in closed form, $G(d + 1) - 2G(d) + G(d - 1)$ with the Gamma's integrated cdf.
  It is computed once, in float64, and matches primarycensored.
- The negative binomial concentration is capped at $\phi \le 1000$, which is Poisson for these counts.
  The Dirichlet-multinomial concentration is capped at $\psi \le 10^4$.
- The hub imputation sums are products of banded $7 \times 7k$ blocks for each onset week, so the cost grows with the number of weeks and not its square.

The tests check the renewal and the imputation against a NumPy reference, `tests/model/numpy_reference.py`.

## Sampling

PyRenew's `PyrenewBuilder` builds each model as a `MultiSignalModel`, and NumPyro's NUTS fits it.
Chains run vectorised.
Without other settings, NUTS runs 4 chains of 1000 warm-up and 1000 draws, with target acceptance 0.9 and maximum tree depth 10.
Chains start from the median of 15 prior draws.
Submission fits take their settings from `ci/policy.toml`, which override JSON settings in `BDBV_RENEWAL_SAMPLER`.
Backtest and leave-one-province-out fits take theirs from `config/backtest.toml` when `BDBV_RENEWAL_SAMPLER` is unset.

The parameterisation is as follows.

- Every walk, the pooled $\log R(t)$ included, samples its steps non-centred, as standard normals scaled by the step SD.
- Province deviations sum to zero through an orthonormal basis $Q$ of the sum-to-zero subspace, so the $P - 1$ free processes are unconstrained.
- The starting offsets of the $\log R$ deviations are sampled centred, and those of the missing-onset fraction non-centred.
- Walk steps after the fit axis are not sampled by NUTS. They are drawn from the prior given each posterior draw when forecasting.
