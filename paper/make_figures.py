"""Render the results panel and SI plots for each study to figures/.

Uses dummy data until the studies write real forecasts in the schema.

Usage: `python make_figures.py [--out figures] [--studies ebola ...]`.
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

import matplotlib.pyplot as plt

from paperfigs.dummy import dummy_factors, dummy_split, dummy_study
from paperfigs.plots import (
    results_panel,
    si_scores_by_factor,
    si_scores_by_location_horizon,
    si_wis_over_time,
)
from paperfigs.schema import STUDIES
from paperfigs.scores import score_forecasts
from paperfigs.style import theme


def render(
    out: Path, studies: Sequence[str] = STUDIES, n_samples: int = 100
) -> list[Path]:
    """Write every figure for `studies` to `out` and return the paths."""
    out.mkdir(parents=True, exist_ok=True)
    written = []
    with theme():
        for study in studies:
            fc, obs = dummy_study(study, n_samples=n_samples)
            scores = score_forecasts(fc)
            if study == "synthetic":
                # Outbreaks are many, so scores break down by scenario
                # factor rather than by location.
                factors = dummy_factors(fc)
                split = None
                figures = {
                    "results": results_panel(fc, obs, scores, factors=factors),
                    "si-factors": si_scores_by_factor(scores, factors),
                    "si-factors-log": si_scores_by_factor(
                        scores, factors, col="rel_log_score"
                    ),
                }
            else:
                # Germany is split into a development and a held-out
                # half; the dummy split is the middle origin.
                split = dummy_split(fc) if study == "germany" else None
                figures = {
                    "results": results_panel(fc, obs, scores, split=split),
                    "si-location-horizon": si_scores_by_location_horizon(
                        scores, split=split
                    ),
                    "si-location-horizon-log": si_scores_by_location_horizon(
                        scores, col="rel_log_score", split=split
                    ),
                }
            figures["si-wis-over-time"] = si_wis_over_time(scores, split=split)
            for name, fig in figures.items():
                path = out / f"{study}-{name}.png"
                fig.savefig(path, dpi=200)
                plt.close(fig)
                written.append(path)
    return written


def main() -> None:
    """Parse arguments and render."""
    parser = argparse.ArgumentParser(description=__doc__)
    here = Path(__file__).parent
    parser.add_argument("--out", type=Path, default=here / "figures")
    parser.add_argument("--studies", nargs="+", default=list(STUDIES))
    args = parser.parse_args()
    for path in render(args.out, args.studies):
        print(path)


if __name__ == "__main__":
    main()
