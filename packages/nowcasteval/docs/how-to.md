# How to

The examples continue from [Getting started](index.md).

## Score a table of forecasts

`score_forecasts` scores one row per sample against the observed counts, on the natural and log scales.
`relative_skill` divides each model's mean score by a baseline's, on the forecasts both made.

```python
import pandas as pd

from nowcasteval.scoring import relative_skill, score_forecasts

rows = []
for model, rate in (("baseline", 20), ("candidate", 17)):
    for week in ("2026-01-10", "2026-01-17"):
        for i, value in enumerate(rng.poisson(rate, 100)):
            rows.append(
                {
                    "model": model,
                    "kind": "real",
                    "vintage": "2026-01-05",
                    "reference_date": "2026-01-10",
                    "location": "A",
                    "target_end_date": week,
                    "horizon": 0,
                    "sample": i,
                    "value": value,
                }
            )
forecasts = pd.DataFrame(rows)
truth = pd.DataFrame(
    {
        "location": ["A", "A"],
        "target_end_date": ["2026-01-10", "2026-01-17"],
        "truth": [16, 18],
    }
)
scores = score_forecasts(forecasts, truth)
skill = relative_skill(scores, baseline="baseline")
assert {"wis", "log_wis", "crps", "coverage_90"} <= set(scores.columns)
assert skill.set_index("model").loc["baseline", "relative_wis"] == 1.0
```

## Score a joint forecast of several places

`energy_sample` scores samples of a vector, such as the same week in several places, so that a forecast is rewarded for getting their dependence right.

```python
from nowcasteval.scoring import energy_sample

joint = rng.poisson([20, 5], size=(1000, 2))
energy = energy_sample([18, 7], joint)
assert energy > 0
```

Next: [Reference](reference.md).
