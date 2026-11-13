"""The synthetic scenario grid, a main-text table of the paper.

`GRID` lists each setting `scripts/finetune_simulate.R` draws for an
outbreak, with its range, its distribution and the epinowcast function it
enters through. `grid_table` adds the values realised over the simulated
outbreaks, from their `settings-*.csv` files, after outbreaks outside 50
to 30,000 cases were drawn again. Run from the repo root:

    python -m paper.tables.synthetic_grid output/finetune/full paper/tables
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Final

import numpy as np
import pandas as pd

__all__ = ["COLUMNS", "GRID", "grid_table", "to_markdown"]

COLUMNS: Final = (
    "setting",
    "factor",
    "range",
    "distribution",
    "epinowcast",
    "realised",
)
"""Columns of the table."""

_HEADERS: Final = {
    "factor": "Factor",
    "range": "Range or levels",
    "distribution": "Distribution",
    "epinowcast": "Enters through",
    "realised": "Realised, median [min, max]",
}

GRID: Final = pd.DataFrame(
    [
        (
            "weeks",
            "Weeks of cut-offs",
            "16 to 40",
            "Uniform on integers",
            "`enw_preprocess_data()` grid, with 12 more weeks of reports",
        ),
        (
            "offset",
            "Start day offset (days)",
            "0 to 6",
            "Uniform on integers",
            "`enw_preprocess_data()` grid",
        ),
        (
            "max_delay",
            "Longest delay (days)",
            "56",
            "Fixed",
            "`enw_preprocess_data(max_delay)`",
        ),
        (
            "n_places",
            "Places",
            "1, or 2 to 26",
            "1 with probability 0.4, else uniform on integers",
            "`enw_preprocess_data(by = 'location')`",
        ),
        (
            "coupling",
            "Weight of the shared log Rt path",
            "0 to 1",
            "Uniform",
            "`enw_simulate(growth_rate)`",
        ),
        (
            "gt_mean",
            "Generation time mean (days)",
            "6 to 20",
            "Uniform",
            "`enw_expectation(generation_time)`, discretised gamma on 1 to"
            " 45 days",
        ),
        (
            "gt_cv",
            "Generation time coefficient of variation",
            "0.3 to 0.7",
            "Uniform",
            "`enw_expectation(generation_time)`",
        ),
        (
            "log_r0",
            "Initial log Rt",
            "Unbounded",
            "Normal(0.1, 0.2)",
            "`enw_simulate(growth_rate)`, with `enw_expectation(r = ~1)`",
        ),
        (
            "n_steps",
            "Step changes in log Rt",
            "0 or more",
            "Poisson(1.2), at uniform days",
            "`enw_simulate(growth_rate)`",
        ),
        (
            "step_sd",
            "Step size sd (mean -0.2)",
            "0.1 to 0.5",
            "Uniform",
            "`enw_simulate(growth_rate)`",
        ),
        (
            "drift_sd",
            "Daily random walk sd of log Rt (clamped to -1.2 to 1.2)",
            "0 to 0.03",
            "Uniform",
            "`enw_simulate(growth_rate)`",
        ),
        (
            "seed_size",
            "Initial expected cases a day, over all places",
            "0.5 to 20",
            "Log-uniform",
            "`expr_lelatent_int` of `enw_expectation()`",
        ),
        (
            "place_sd",
            "Sd of the places' log seed weights",
            "0.5 to 1.5",
            "Uniform",
            "`expr_lelatent_int` of `enw_expectation()`",
        ),
        (
            "delay_mean",
            "Delay median, exp of the lognormal log mean (days)",
            "2 to 21",
            "Log-uniform",
            "`refp_mean_int` of `enw_reference(parametric = ~1 + week)`",
        ),
        (
            "delay_sd",
            "Delay lognormal log sd",
            "0.3 to 1.1",
            "Uniform",
            "`refp_sd_int` of `enw_reference()`",
        ),
        (
            "delay_drift",
            "Daily change in the delay log mean (applied weekly)",
            "Unbounded",
            "Normal(0, 0.02)",
            "`week` effect of `enw_reference(parametric)`",
        ),
        (
            "delay_place_sd",
            "Sd of the places' delay log means",
            "0 to 0.4",
            "Uniform",
            "`(1 | location)` of `enw_reference(parametric)`",
        ),
        (
            "hazard_sd",
            "Sd of the delay hazard effects",
            "0 to 0.4",
            "Uniform",
            "`enw_reference(non_parametric = ~ 0 + (1 | delay))`",
        ),
        (
            "report_days",
            "Days a week with reports",
            "7, or 1 to 3 weekdays",
            "7 with probability 0.5, else uniform on 1 to 3 days of"
            " Monday to Friday",
            "`enw_dayofweek_structural_reporting()` in"
            " `enw_report(structural)`",
        ),
        (
            "dow_sd",
            "Sd of the report day-of-week effects",
            "0 to 0.8",
            "Uniform",
            "`enw_report(non_parametric = ~ 0 + (1 | day_of_week))`",
        ),
        (
            "sqrt_phi",
            "Negative binomial overdispersion, 1 / sqrt(phi)",
            "0.05 to 1",
            "Uniform",
            "`enw_obs(family = 'negbin')`",
        ),
        (
            "cases",
            "Reported cases in an outbreak",
            "50 to 30,000",
            "Outbreaks outside the range are drawn again",
            "`enw_simulate()`",
        ),
    ],
    columns=list(COLUMNS[:-1]),
)
"""One row per setting of `draw_settings` in `scripts/finetune_simulate.R`,
and the fixed longest delay and the accepted outbreak size."""


def _number(x: float) -> str:
    """Return a number to three significant figures, without exponents."""
    if abs(x) >= 1000:
        return f"{x:,.0f}"
    return np.format_float_positional(
        x, precision=3, unique=False, fractional=False, trim="-"
    )


def _realised(name: str, values: pd.Series) -> str:
    """Return a setting's realised median and range over outbreaks."""
    if name == "report_days":
        return f"{(values == 7).mean():.0%} every day"
    out = (
        f"{_number(values.median())} "
        f"[{_number(values.min())}, {_number(values.max())}]"
    )
    if name == "n_places":
        out += f"; {(values == 1).mean():.0%} one place"
    return out


def grid_table(settings: pd.DataFrame) -> pd.DataFrame:
    """Return `GRID` with the values realised over the outbreaks.

    Parameters
    ----------
    settings : pandas.DataFrame
        One row per outbreak, from the simulator's `settings-*.csv`.

    Returns
    -------
    pandas.DataFrame
        Columns `COLUMNS`. A setting with no column in `settings` has an
        empty `realised`.
    """
    out = GRID.copy()
    out["realised"] = [
        _realised(name, settings[name]) if name in settings else ""
        for name in out["setting"]
    ]
    return out[list(COLUMNS)]


def to_markdown(table: pd.DataFrame) -> str:
    """Return the table as Markdown, without the `setting` key."""
    cols = list(_HEADERS)
    lines = [
        "| " + " | ".join(_HEADERS[c] for c in cols) + " |",
        "|" + "---|" * len(cols),
    ]
    for _, row in table.iterrows():
        cells = [str(row[c]).replace("|", "\\|") for c in cols]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def main() -> None:
    """Write `synthetic-grid.csv` and `synthetic-grid.md`."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sim_dir", type=Path)
    parser.add_argument("out_dir", type=Path)
    args = parser.parse_args()
    settings = pd.concat(
        [pd.read_csv(p) for p in sorted(args.sim_dir.glob("settings-*.csv"))]
    )
    table = grid_table(settings)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.out_dir / "synthetic-grid.csv", index=False)
    caption = (
        f"Settings drawn for each simulated outbreak. Realised values are "
        f"over the {len(settings):,} training outbreaks.\n\n"
    )
    (args.out_dir / "synthetic-grid.md").write_text(
        caption + to_markdown(table)
    )


if __name__ == "__main__":
    main()
