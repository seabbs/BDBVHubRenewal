# nowcasteval

Score sample and quantile forecasts, and compare models against a baseline.
{ .page-subtitle }

Until the package split, nowcasteval is `bdbvhubrenewal.evaluation` ([Packages](index.md)).

## Install

```sh
uv pip install "git+https://github.com/seabbs/bdbv-hub-modelling"
```

## Getting started

We score a sample forecast of one count with the CRPS, the weighted interval score (WIS) and interval coverage.

```python
import numpy as np

from nowcasteval.scoring import (
    coverage_sample,
    crps_sample,
    wis_quantile,
)

rng = np.random.default_rng(1)
samples = rng.poisson(20, size=1000)
observed = 18

crps = crps_sample(observed, samples)
covered = coverage_sample(observed, samples, level=0.9)
levels = np.array([0.05, 0.25, 0.5, 0.75, 0.95])
wis = wis_quantile([observed], np.quantile(samples, levels)[None, :], levels)
assert covered
assert np.isclose(
    wis["wis"][0],
    wis["dispersion"][0] + wis["overprediction"][0] + wis["underprediction"][0],
)
```

The CRPS is the fair form, whose expectation does not depend on the number of samples.
The WIS comes with its three parts, dispersion, overprediction and underprediction, which sum to it.

## How to

### Score a table of forecasts

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

### Score a joint forecast of several places

`energy_sample` scores samples of a vector, such as the same week in several places, so that a forecast is rewarded for getting their dependence right.

```python
from nowcasteval.scoring import energy_sample

joint = rng.poisson([20, 5], size=(1000, 2))
energy = energy_sample([18, 7], joint)
assert energy > 0
```

## Reference

[Evaluation API](../reference/evaluation.md).

## How it is used in the workflow

The backtest, the checkpoint scores of the fine-tuned model and the paper's case studies all score through these functions, so every score shown on the site and in the paper is computed the same way.

Next: [nowcastbaselines](nowcastbaselines.md).
