# nowcasteval

Score sample and quantile forecasts, and compare models against a baseline.
{ .page-subtitle }

nowcasteval is the package in `packages/nowcasteval` of the repository, and builds on [nowcastutils](https://samabbott.co.uk/bdbv-hub-modelling/nowcastutils/) ([Packages](https://samabbott.co.uk/bdbv-hub-modelling/packages/)).

## Install

```sh
uv pip install "git+https://github.com/seabbs/bdbv-hub-modelling#subdirectory=packages/nowcasteval"
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

Next: [How to](how-to.md).
