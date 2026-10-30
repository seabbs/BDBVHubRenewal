# Implementation

How the [models](model.md) are built in code, computed and sampled.
{ .page-subtitle }

## Code

The model extends [PyRenew](https://github.com/CDCgov/PyRenew), pinned to a commit, and runs in JAX with NumPyro.
Its code is in [`bdbvhubrenewal.model`][bdbvhubrenewal.model] and the variants are in [`forecasters.renewal`][bdbvhubrenewal.forecasters.renewal].
The model is configured by `ModelConfig`.
The model figure is built from `docs/diagrams/model_dag.tex` by `scripts/build_diagrams.sh`.

## PyRenew and what is built here


| Component | PyRenew piece | Built here |
|---|---|---|
| Pooled $R_p(t)$ | `SubpopulationInfections`, `WeeklyTemporalProcess` | `KnotWalk` and `NonCentredRandomWalk`. |
| Renewal with importation and depletion | `SubpopulationInfections`, subclassed | `coupled_renewal`. |
| Ascertainment $\alpha(s)$ | `IndependentAscertainment` with a temporal process | None. |
| Onset-to-report delay $f_s(d)$ | None | `HazardDelay`. |
| Missing onsets $\pi_p(t)$ | None | `MissingOnsetFraction`, built from the same pooled walk as $R_p$. |
| Known and imputed onsets | `BaseObservationProcess`, `NegativeBinomialNoise` | `OnsetStreams`. |
| BVDOutbreakSize Rt | None | `RtPseudoObservation`. |
| Provinces with no cases | None | An importation-only component. |

## Numerics

The model runs in single precision (float32), JAX's default.

- Depletion is carried on $\log S$, which needs no subtraction, and $1 - e^{-y}$ is computed with `expm1`.
- The generation interval is double interval censored in closed form, $G(d + 1) - 2G(d) + G(d - 1)$ with the Gamma's integrated cdf. It is computed once, in float64, and matches primarycensored.
- The negative binomial concentration is capped at $\phi \le 1000$, which is Poisson for these counts. The Dirichlet-multinomial concentration is capped at $\psi \le 10^4$.
- The hub imputation sums are products of banded $7 \times 7k$ blocks for each onset week. So the cost grows with the number of weeks and not its square.

The tests check the renewal and the imputation against a NumPy reference, `tests/model/numpy_reference.py`.

## Sampling

Fits use NumPyro's NUTS through PyRenew's `PyrenewBuilder` and `MultiSignalModel`.
Chains run vectorised.
Without other settings, NUTS runs 4 chains of 1000 warm-up and 1000 draws, with target acceptance 0.9 and maximum tree depth 10.
Chains start from the median of 15 prior draws.
Submission and backtest fits take their settings from `ci/policy.toml` and `config/backtest.toml`.
The submission refit uses smaller steps (target acceptance 0.95) and inits drawn uniformly on $(-1, 1)$ on the unconstrained scale.

The parameterisation is as follows.

- The pooled $\log R(t)$ walk samples its weekly knots, not a level and its steps.
- The other walks sample their steps non-centred, as standard normals scaled by the step SD.
- Province deviations sum to zero through an orthonormal basis $Q$ of the sum-to-zero subspace, so the $P - 1$ free processes are unconstrained.
- The starting offsets of the deviations are sampled centred.
