# Implementation

How the [model](../methods/model.md) is built in code.
{ .page-subtitle }

## Code

The model extends [PyRenew](https://github.com/CDCgov/PyRenew), pinned to a commit.
Its code is in [`bdbv_hub_renewal.model`][bdbv_hub_renewal.model] and the variants are in [`forecasters.renewal`][bdbv_hub_renewal.forecasters.renewal].
Fitting uses PyRenew's `PyrenewBuilder` and `MultiSignalModel`, with NumPyro's NUTS.
The BVDOutbreakSize releases are read by `bdbv_hub_renewal.data.bvd`.
The model choice and its date are logged in `results/model-choice.yml`.
The model figure is built from `docs/diagrams/model_dag.tex` by `scripts/build_diagrams.sh`.

## Components

PyRenew is used where it fits, and a component is built here where it does not.

| Component | PyRenew piece | Built here, and why |
|---|---|---|
| Pooled $R_p(t)$ | `SubpopulationInfections`, `WeeklyTemporalProcess` | `KnotWalk` and `NonCentredRandomWalk`, as PyRenew's `RandomWalk` samples its steps one site at a time, a funnel. |
| Renewal with importation and depletion | `SubpopulationInfections`, subclassed | `coupled_renewal`, as PyRenew runs each province on its own and cannot carry cumulative depletion. |
| Ascertainment $\alpha(s)$ | `IndependentAscertainment` with a temporal process | None. |
| Onset-to-report delay $f_s(d)$ | None | `HazardDelay`, as PyRenew has no time-varying or nonparametric delay. |
| Missing onsets $\pi_p(t)$ | None | `MissingOnsetFraction`, built from the same pooled walk as $R_p$. |
| Known and imputed onsets | `BaseObservationProcess`, `NegativeBinomialNoise` | `OnsetStreams`, as PyRenew's count observation takes one fixed delay and no back-sampled imputation. |
| Provinces with no cases | None | An importation-only component. |

## Options

The model's options are fields of `ModelConfig`.
The defaults give the model on the Model page.
Each option named there maps to a field.

| Option | Field |
|---|---|
| Rt held at its last fitted value in the forecast | `future_rt="hold"` |
| Weekly Rt knots joined linearly | `rt_cadence="linear"` |
| Estimated gravity kernel | `coupling="learnt"` |
| Importation share per origin | `importation="per_origin"` |
| Report day-of-week effect on the delay hazard | `report_dow=True` |
| Multinomial delay counts | `delay_counts="multinomial"` |

## Numerics

Depletion is computed with `expm1` and on $\log S$, which keeps the log density smooth in single precision.
The hub imputation sums are computed by week, as products of banded $7 \times 7k$ blocks for each onset week.
So the cost grows with the number of weeks and not its square.
The tests check the imputation against a NumPy reference, `tests/model/numpy_reference.py`.
