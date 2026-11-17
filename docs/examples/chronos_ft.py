"""Nowcast a reporting triangle with our fine-tuned Chronos-2."""

import numpy as np
import pandas as pd

from bdbvhubrenewal import chronos_ft

# Twelve weeks of reports, each week's count spread over delays 0 to 3.
rng = np.random.default_rng(1)
weeks = pd.date_range("2026-08-01", periods=12, freq="7D")
cases = rng.poisson(np.linspace(5, 60, len(weeks)))
share = np.array([0.4, 0.3, 0.2, 0.1])
counts = np.stack([rng.multinomial(n, share) for n in cases]).astype(float)
# Row t is reported up to delay 11 - t at the last cut-off.
t, d = np.indices(counts.shape)
counts[t + d > len(weeks) - 1] = np.nan
triangle = pd.DataFrame(counts, index=weeks, columns=range(4))

# The newest chronos-ft release, its adapter checked against its SHA-256.
model = chronos_ft.load("latest")
forecast = chronos_ft.forecast_triangle(
    triangle, horizon=2, model=model, quantile_levels=[0.1, 0.5, 0.9]
)
if __name__ == "__main__":
    print(f"checkpoint {model.checkpoint.hash}")
    print(
        forecast.pivot_table(
            index="reference", columns="quantile_level", values="value"
        )
    )
