# Implementation

How the [models](model.md) are built in code, computed and sampled.
{ .page-subtitle }

## Code

The model extends [PyRenew](https://github.com/CDCgov/PyRenew), pinned to a commit, and runs in JAX with NumPyro.
Its code is in [`bdbvhubrenewal.model`][bdbvhubrenewal.model] and the variants are in [`forecasters.renewal`][bdbvhubrenewal.forecasters.renewal].
The model's options are fields of `ModelConfig`, and the defaults give the model on the Model page.
The model figure is built from `docs/diagrams/model_dag.tex` by `scripts/build_diagrams.sh`.

## PyRenew and what is built here

PyRenew is used where it fits.
A component is built here where PyRenew cannot express the model or samples it poorly.

| Component | PyRenew piece | Built here, and why |
|---|---|---|
| Pooled $R_p(t)$ | `SubpopulationInfections`, `WeeklyTemporalProcess` | `KnotWalk` and `NonCentredRandomWalk`, as PyRenew's `RandomWalk` samples its steps one site at a time, a funnel. |
| Renewal with importation and depletion | `SubpopulationInfections`, subclassed | `coupled_renewal`, as PyRenew runs each province on its own and cannot carry cumulative depletion. |
| Ascertainment $\alpha(s)$ | `IndependentAscertainment` with a temporal process | None. |
| Onset-to-report delay $f_s(d)$ | None | `HazardDelay`, as PyRenew has no time-varying or nonparametric delay. |
| Missing onsets $\pi_p(t)$ | None | `MissingOnsetFraction`, built from the same pooled walk as $R_p$. |
| Known and imputed onsets | `BaseObservationProcess`, `NegativeBinomialNoise` | `OnsetStreams`, as PyRenew's count observation takes one fixed delay and no back-sampled imputation. |
| BVDOutbreakSize Rt | None | `RtPseudoObservation`, as PyRenew has no observation process on a latent Rt. |
| Provinces with no cases | None | An importation-only component. |

Model options map to `ModelConfig` fields.

| Option | Field |
|---|---|
| Rt held at its last fitted value in the forecast | `future_rt="hold"` |
| Weekly Rt knots joined linearly | `rt_cadence="linear"` |
| Estimated gravity kernel | `coupling="learnt"` |
| Importation share per origin | `importation="per_origin"` |
| Report day-of-week effect on the delay hazard | `report_dow=True` |
| Multinomial delay counts | `delay_counts="multinomial"` |

## Numerics

The model runs in single precision (float32), JAX's default.
Several parts are written to keep the log density smooth at that precision, since a noisy gradient stalls NUTS.

- Depletion is carried on $\log S$, which needs no subtraction, and $1 - e^{-y}$ is computed with `expm1`. PyRenew's form loses most of its digits at per-capita rates near $10^{-6}$.
- The generation interval is double interval censored in closed form, $G(d + 1) - 2G(d) + G(d - 1)$ with the Gamma's integrated cdf. It is computed once, in float64, and matches primarycensored.
- Parametric delay options are censored by Gauss-Legendre quadrature of $\int_0^1 \{F(d + 1 - u) - F(d - u)\}\, du$. This keeps each term on the scale of the pmf, where the closed form loses digits in float32.
- The negative binomial concentration is capped at $\phi \le 1000$, which is Poisson for these counts. The Dirichlet-multinomial concentration is capped at $\psi \le 10^4$. Nearer their limits both log densities are too noisy in float32.
- The hub imputation sums are products of banded $7 \times 7k$ blocks for each onset week. So the cost grows with the number of weeks and not its square.

The tests check the renewal and the imputation against a NumPy reference, `tests/model/numpy_reference.py`.

## Sampling

Fits use NumPyro's NUTS through PyRenew's `PyrenewBuilder` and `MultiSignalModel`.
Chains run vectorised rather than in parallel, as parallel chains leak memory on Linux CPU.
Without other settings, NUTS runs 4 chains of 1000 warm-up and 1000 draws, with target acceptance 0.9 and maximum tree depth 10.
Chains start from the median of 15 prior draws.
Submission and backtest fits take their settings from `ci/policy.toml` and `config/backtest.toml`.
The submission refit uses smaller steps (target acceptance 0.95) and inits drawn uniformly on $(-1, 1)$ on the unconstrained scale.

The parameterisation avoids funnels and ridges in the posterior.

- The pooled $\log R(t)$ walk samples its weekly knots, not a level and its steps. This removes the direction in which the level and every step move together.
- The other walks sample their steps non-centred, as standard normals scaled by the step SD.
- Province deviations sum to zero through an orthonormal basis $Q$ of the sum-to-zero subspace, so the $P - 1$ free processes are unconstrained.
- The starting offsets of the deviations are sampled centred.
