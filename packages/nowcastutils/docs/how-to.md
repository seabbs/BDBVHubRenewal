# How to

The examples continue from [Getting started](index.md).

## Find each release's cut-off

A release's cut-off is its latest onset day.

```python
from nowcastutils.releases import latest_onset_days

cutoffs = latest_onset_days(daily)
assert cutoffs.to_dict() == {
    "2026-03-07": "2026-03-07",
    "2026-03-14": "2026-03-14",
}
```

## Read the location hierarchy from the codes

Codes are parts joined by `-`, so a code's parent is the code without its last part.
There is no separate hierarchy table.

```python
from nowcastutils.hierarchy import depth, parent

assert parent("A-1") == "A"
assert parent("A") is None
assert depth("A-1-07") == 2
```

## Draw sample trajectories from quantile forecasts

A quantile forecast gives each place and week its own quantiles, but says nothing about how the weeks move together.
`couple` draws trajectories whose weeks each follow their quantiles.
With `comonotone`, a trajectory takes the same quantile level in every week, so a high week is followed by high weeks.

```python
import numpy as np

from nowcastutils.trajectories import couple

levels = np.array([0.1, 0.5, 0.9])
# Two places, three weeks, quantiles at each level.
quantiles = np.array(
    [
        [[8.0, 10.0, 12.0], [9.0, 12.0, 15.0], [10.0, 14.0, 18.0]],
        [[1.0, 2.0, 3.0], [1.0, 3.0, 5.0], [2.0, 4.0, 6.0]],
    ]
)
draws = couple(quantiles, levels, "comonotone", None, n_samples=500, seed=1)
assert draws.shape == (500, 2, 3)
# Each trajectory keeps its rank across the weeks.
rank = draws[:, 0, :].argsort(axis=0).argsort(axis=0)
assert (rank[:, 0] == rank[:, 2]).all()
```

`gaussian` joins the weeks and places with correlations given in `fit` (`rho` between weeks, `s` between places), and `independent` draws every cell on its own.
`fit_coupling` estimates that `fit` from the model's own earlier forecasts.
`pit_scores` scores what was later reported in them, and the correlations are shrunk towards independence as far as the scores' own noise says.

```python
from nowcastutils.trajectories import fit_coupling, pit_scores

rng = np.random.default_rng(2)
# Four earlier forecasts of the same two places and three weeks.
earlier = np.repeat(quantiles[None], 4, axis=0)
reported = earlier[..., 1] + rng.normal(0, 1, (4, 2, 3))
fit = fit_coupling(pit_scores(earlier, levels, reported, rng), "gaussian")
draws = couple(quantiles, levels, "gaussian", fit, n_samples=500, seed=1)
```

Next: [Reference](reference.md).
