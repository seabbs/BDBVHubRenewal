# nowcastutils

Read data releases into the arrays the models use: cut-offs, snapshots and the location hierarchy.
{ .page-subtitle }

Until the package split, nowcastutils is `bdbvhubrenewal.data` ([Packages](index.md)).

## Install

```sh
uv pip install "git+https://github.com/seabbs/bdbv-hub-modelling"
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
from bdbvhubrenewal.data.snapshots import snapshot_arrays, weekly_sums

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
from bdbvhubrenewal.data.releases import latest_onset_days

cutoffs = latest_onset_days(daily)
assert cutoffs.to_dict() == {"2026-03-07": "2026-03-07", "2026-03-14": "2026-03-14"}
```

### Read the location hierarchy from the codes

Codes are parts joined by `-`, so a code's parent is the code without its last part.
There is no separate hierarchy table.

```python
from bdbvhubrenewal.data.hierarchy import depth, parent

assert parent("A-1") == "A"
assert parent("A") is None
assert depth("A-1-07") == 2
```

## Reference

[Data API](../reference/data.md).

## How it is used in the workflow

Every model reads its data through these functions, so they all see the same releases, cut-offs and places.

Next: [nowcasteval](nowcasteval.md).
