# nowcastutils

The pieces the other packages build on: releases, snapshots, the location hierarchy, the forecast task and sample trajectories from quantiles.
{ .page-subtitle }

nowcastutils is the package in `packages/nowcastutils` of the repository ([Packages](https://samabbott.co.uk/bdbv-hub-modelling/packages/)).

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

Next: [How to](how-to.md).
