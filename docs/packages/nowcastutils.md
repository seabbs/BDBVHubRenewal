# nowcastutils

The pieces the other packages build on: releases, snapshots, the location hierarchy, the forecast task and sample trajectories from quantiles.
{ .page-subtitle }

nowcastutils is the package in `packages/nowcastutils` of the repository ([Packages](index.md)).

## Install

```sh
uv pip install "git+https://github.com/seabbs/bdbv-hub-modelling#subdirectory=packages/nowcastutils"
```

## Getting started

Releases are daily onset counts: each release (`vintage`) lists every onset day up to its data cut-off (`cutoff`), with the cases with a known onset (`observed`) and with an imputed one (`imputed`).
We make two releases a week apart for two places.

```python
import pandas as pd

rows = []
for vintage in ("2026-03-07", "2026-03-14"):
    for place in ("A-1", "A-2"):
        for day in pd.date_range("2026-02-01", vintage):
            reported = 10 if (pd.Timestamp(vintage) - day).days > 6 else 4
            rows.append((vintage, place, day.date().isoformat(), reported, 0))
daily = pd.DataFrame(
    rows, columns=["vintage", "location", "onset_date", "observed", "imputed"]
)
daily["cutoff"] = daily["vintage"]
```

`snapshot_arrays` lays out every release up to a vintage as arrays by place, release and onset day.
It adds each place's ancestors from its code, so the country `A` comes first.

```python
from nowcastutils.snapshots import snapshot_arrays, weekly_sums

snap = snapshot_arrays(daily, "2026-03-14", ("A-1", "A-2"))
assert snap.locations == ("A", "A-1", "A-2")
assert snap.releases == ("2026-03-07", "2026-03-14")
weekly = weekly_sums(snap, len(snap.releases) - 1)
assert weekly.shape == (3, snap.n_weeks)
```

## How to

### Find each release's cut-off

A release's cut-off is its latest onset day.

```python
from nowcastutils.releases import latest_onset_days

cutoffs = latest_onset_days(daily)
assert cutoffs.to_dict() == {
    "2026-03-07": "2026-03-07",
    "2026-03-14": "2026-03-14",
}
```

### Read the location hierarchy from the codes

Codes are parts joined by `-`, so a code's parent is the code without its last part.
There is no separate hierarchy table.

```python
from nowcastutils.hierarchy import depth, parent

assert parent("A-1") == "A"
assert parent("A") is None
assert depth("A-1-07") == 2
```

### Draw sample trajectories from quantile forecasts

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

## Reference

[nowcastutils API](../reference/nowcastutils.md).

## How it is used in the workflow

Every model reads its data through these functions, so they all see the same releases, cut-offs and places.
Each model takes a `ForecastTask` from `nowcastutils.base`, returns a sample table, and is found by name in its registry.

Next: [nowcasteval](nowcasteval.md).
