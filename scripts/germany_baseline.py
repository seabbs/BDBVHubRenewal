"""German development-half forecasts of sample models, one file per date.

Writes OUT/<date>.csv.gz per development nowcast date, with the 40-day
truth, skipping dates already written, so a rerun resumes. Draws are the
benchmark's common random numbers (`config/germany.toml` seed) with
N_SAMPLES draws per forecast.

Usage: python germany_baseline.py OUT N_SAMPLES MODEL...
"""

from __future__ import annotations

import sys
from pathlib import Path

from bdbvhubrenewal.data.germany import hub_triangle
from bdbvhubrenewal.evaluation import germany as de


def main() -> None:
    """Write each development date's forecasts."""
    out, n_samples = Path(sys.argv[1]), int(sys.argv[2])
    models = sys.argv[3:]
    out.mkdir(parents=True, exist_ok=True)
    config = de.load_germany_config("config/germany.toml")
    triangle = hub_triangle(
        Path("data/raw/germany") / Path(config.triangle).name
    )
    for date in de.nowcast_dates(config, held_out=False):
        path = out / f"{date}.csv.gz"
        if path.exists():
            continue
        fc = de.with_truth(
            de.germany_forecasts(
                models, triangle, [date], n_samples, config.seed
            ),
            triangle,
            config.truth_delay,
        )
        tmp = path.with_suffix(".tmp")
        fc.to_csv(tmp, index=False, compression="gzip")
        tmp.rename(path)
        print(date, flush=True)


if __name__ == "__main__":
    main()
