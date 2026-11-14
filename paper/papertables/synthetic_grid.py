"""The synthetic scenario grid, a main-text table of the paper.

`GRID` lists each setting `scripts/finetune_simulate.R` draws for an
outbreak, with its range, its distribution and the epinowcast function it
enters through. `grid_table` adds the values realised over the simulated
outbreaks, from their `settings-*.csv` files, for the first setting of a
row. Run from `paper/`:

    python -m papertables.synthetic_grid ../output/finetune/full tables
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Final

import numpy as np
import pandas as pd

__all__ = [
    "COLUMNS",
    "GRID",
    "MAIN",
    "MAIN_HEADERS",
    "grid_table",
    "main_table",
    "to_markdown",
    "write_tables",
]

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
    "realised": "Realised",
}

GRID: Final = pd.DataFrame(
    [
        (
            "strata",
            "Strata",
            "One place; 2 to 26 places; 2 to 8 age groups in one place; 6 to "
            "16 places by 3 to 7 age groups",
            "Probabilities 0.3, 0.4, 0.2 and 0.1",
            "`enw_preprocess_data(by = c('place', 'age'))`",
        ),
        (
            "n_places",
            "Places",
            "1 to 26",
            "Uniform on the strata's range",
            "`enw_preprocess_data()`",
        ),
        (
            "n_ages",
            "Age groups",
            "1 to 8",
            "Uniform on the strata's range",
            "`enw_preprocess_data()`",
        ),
        (
            "process",
            "Generative process",
            "Renewal; renewal with incubation; weekly growth walk; log "
            "incidence walk; growth steps; smooth growth",
            "Uniform",
            "`enw_expectation()` and `enw_simulate(growth_rate)`",
        ),
        (
            "reference",
            "Reference date model",
            "Parametric; non-parametric; both",
            "Probabilities 0.3, 0.2 and 0.5",
            "`enw_reference()`",
        ),
        (
            "delay_family",
            "Parametric delay",
            "Lognormal; gamma; exponential",
            "Uniform, if there is a parametric delay",
            "`enw_reference(distribution)`",
        ),
        (
            "hazard",
            "Delay hazard effects",
            "None; by delay; by week; both",
            "None with a parametric delay alone, by delay if stable, else "
            "uniform",
            "`enw_reference(non_parametric)`",
        ),
        (
            "stable",
            "Stable delay",
            "Yes or no",
            "Yes with probability 0.2, which turns off every delay and report "
            "change",
            "All delay changes",
        ),
        (
            "delay_drift",
            "Drift in the delay's location",
            "Yes or no",
            "Probability 0.6 if not stable",
            "`week` in `enw_reference(parametric)`",
        ),
        (
            "spread_drift",
            "Drift in the delay's spread",
            "Yes or no",
            "Probability 0.5 if not stable",
            "`week` in `enw_reference(parametric)`",
        ),
        (
            "delay_steps",
            "Step changes in the delay",
            "0 to 3",
            "Probabilities 0.3, 0.3, 0.2 and 0.2 if not stable",
            "Step covariates in `enw_reference(parametric)`",
        ),
        (
            "delay_walk",
            "Weekly random walk in the delay",
            "Yes or no",
            "Probability 0.3 if not stable",
            "`rw(week)` in `enw_reference(parametric)`",
        ),
        (
            "report_dow",
            "Day-of-week report effects",
            "Yes or no",
            "Probability 0.7",
            "`(1 | day_of_week)` in `enw_report()`",
        ),
        (
            "report_holiday",
            "Holidays",
            "Yes or no",
            "Probability 0.3",
            "`holiday` in `enw_report()`",
        ),
        (
            "report_week",
            "Report week effects",
            "None; `(1 | week)`; `rw(week)`",
            "Probabilities 0.6, 0.2 and 0.2",
            "`enw_report()`",
        ),
        (
            "report_days",
            "Days with reports",
            "Every day, or 1 to 3 of Monday to Friday",
            "Every day with probability 0.5, else uniform on 1 to 3 days",
            "`enw_report(structural)`",
        ),
        (
            "report_days_after",
            "Days with reports after a change",
            "Every day, or 1 to 3 of Monday to Friday",
            "As the days before",
            "`enw_report(structural)`",
        ),
        (
            "report_change",
            "Change of report days",
            "Yes or no",
            "Probability 0.3 if not stable",
            "`enw_report(structural)`",
        ),
        (
            "family",
            "Observation model",
            "NB2; NB1; Poisson",
            "Probabilities 0.4, 0.4 and 0.2",
            "`enw_obs(family)`",
        ),
        (
            "max_delay",
            "Longest delay (days)",
            "56 or 84",
            "Probabilities 0.75 and 0.25",
            "`enw_preprocess_data(max_delay)`",
        ),
        (
            "weeks",
            "Weeks of cut-offs",
            "16 to 40, or 16 to 24 for places by ages",
            "Uniform on integers",
            "`enw_preprocess_data()` grid, with the longest delay and 4 more "
            "weeks of reports",
        ),
        (
            "offset",
            "Start day offset (days)",
            "0 to 6",
            "Uniform on integers",
            "`enw_preprocess_data()` grid",
        ),
        (
            "quiet",
            "Probability a place is quiet",
            "0 to 0.9",
            "Uniform",
            "`expr_lelatent_int` of `enw_expectation()`",
        ),
        (
            "coupling",
            "Weight of the shared growth path",
            "0 to 1",
            "Uniform",
            "`enw_simulate(growth_rate)`",
        ),
        (
            "gt_mean;gt_cv",
            "Generation time mean (days) and coefficient of variation",
            "6 to 20; 0.3 to 0.7",
            "Uniform",
            "`enw_expectation(generation_time)`, gamma on 1 to 45 days",
        ),
        (
            "incubation_mean;incubation_cv",
            "Incubation period mean (days) and coefficient of variation",
            "3 to 12; 0.3 to 0.7",
            "Uniform",
            "`enw_expectation(latent_reporting_delay)`, gamma on 1 to 30 days",
        ),
        (
            "log_r0",
            "Initial log Rt",
            "Unbounded, paths held to -1.2 to 1.2",
            "Normal(0.1, 0.2)",
            "`enw_simulate(growth_rate)`",
        ),
        (
            "rt_steps;rt_step_sd",
            "Step changes in log Rt, and their size sd (mean -0.2)",
            "0 or more; 0.1 to 0.5",
            "Poisson(1.2); uniform",
            "`enw_simulate(growth_rate)`",
        ),
        (
            "rt_drift_sd",
            "Daily random walk sd of log Rt",
            "0 to 0.03",
            "Uniform",
            "`enw_simulate(growth_rate)`",
        ),
        (
            "r0",
            "Initial daily growth rate",
            "Unbounded, paths held to -0.06 to 0.06",
            "Normal(0.01, 0.02)",
            "`enw_simulate(growth_rate)`",
        ),
        (
            "walk_sd",
            "Weekly growth walk sd",
            "0.005 to 0.015",
            "Uniform",
            "`enw_simulate(growth_rate)`",
        ),
        (
            "incidence_sd",
            "Daily growth noise of the log incidence walk",
            "0.05 to 0.3",
            "Uniform",
            "`enw_simulate(growth_rate)`",
        ),
        (
            "growth_steps;growth_step_sd",
            "Step changes in growth, and their size sd (mean -0.01)",
            "1 to 3; 0.02 to 0.06",
            "Uniform",
            "`enw_simulate(growth_rate)`",
        ),
        (
            "smooth_amp;smooth_period;smooth_phase",
            "Smooth growth amplitude, period (days) and phase",
            "0.01 to 0.06; 40 to 200; 0 to 2 pi",
            "Uniform",
            "`enw_simulate(growth_rate)`",
        ),
        (
            "seed_size",
            "Initial expected cases a day, over all strata",
            "0.2 to 200",
            "Log-uniform",
            "`expr_lelatent_int` of `enw_expectation()`",
        ),
        (
            "stratum_sd",
            "Sd of the places' log seed weights (half for age groups)",
            "0.5 to 2",
            "Uniform",
            "`expr_lelatent_int` of `enw_expectation()`",
        ),
        (
            "delay_mean",
            "Delay mean (days)",
            "2 to 21",
            "Log-uniform",
            "`refp_mean_int`, or the hazard intercept with no parametric delay",
        ),
        (
            "delay_cv",
            "Delay coefficient of variation",
            "0.3 to 1.2 (lognormal); 0.7 to 1.2 (gamma); 1 (exponential)",
            "Uniform",
            "`refp_sd_int` of `enw_reference()`",
        ),
        (
            "delay_change;spread_change",
            "Log change in the delay's location and spread over the outbreak",
            "Unbounded",
            "Normal(0, 0.5); Normal(0, 0.3)",
            "`week` effects of `enw_reference(parametric)`",
        ),
        (
            "step_at_1;step_at_2;step_at_3",
            "When each delay step happens, as a share of the outbreak",
            "0.1 to 0.9",
            "Uniform",
            "Step covariates in `enw_reference(parametric)`",
        ),
        (
            "step_mean_1;step_mean_2;step_mean_3;step_spread_1;step_spread_2;s"
            "tep_spread_3",
            "Delay steps in log location and spread",
            "Unbounded",
            "Normal(0, 0.4); Normal(0, 0.3)",
            "Step effects of `enw_reference(parametric)`",
        ),
        (
            "walk_mean_sd;walk_spread_sd",
            "Weekly walk sd of the delay's location and spread",
            "0 to 0.1",
            "Uniform",
            "`rw(week)` of `enw_reference(parametric)`",
        ),
        (
            "delay_stratum_sd",
            "Sd of the strata's delay effects",
            "0 to 0.4",
            "Uniform",
            "`(1 | place)` and `(1 | age)` of `enw_reference()`",
        ),
        (
            "hazard_sd;hazard_week_sd",
            "Sd of the hazard effects by delay and by week",
            "0 to 0.4",
            "Uniform",
            "`enw_reference(non_parametric)`",
        ),
        (
            "dow_sd",
            "Sd of the day-of-week report effects",
            "0 to 0.8",
            "Uniform",
            "`(1 | day_of_week)` of `enw_report()`",
        ),
        (
            "holidays;holiday_effect",
            "Holidays, and their effect on the logit hazard",
            "2 to 6; -2.5 to -0.5",
            "Uniform",
            "`holiday` of `enw_report()`",
        ),
        (
            "report_week_sd",
            "Sd of the report week effects",
            "0 to 0.3",
            "Uniform",
            "`enw_report()`",
        ),
        (
            "report_change_at",
            "When report days change, as a share of the outbreak",
            "0.2 to 0.8",
            "Uniform",
            "`enw_report(structural)`",
        ),
        (
            "sqrt_phi",
            "Overdispersion, 1 / sqrt(phi), on the NB2 scale",
            "0.05 to 1.5",
            "Uniform; NB1 scaled to match",
            "`enw_obs()`",
        ),
        (
            "cases",
            "Reported cases in an outbreak",
            "10 to 300,000, more with many strata",
            "Drawn again up to 5 times, else the design is skipped",
            "`enw_simulate()`",
        ),
    ],
    columns=list(COLUMNS[:-1]),
)
"""One row per setting, or per set of `;`-separated settings, that
`draw_design` and `draw_values` draw in `scripts/finetune_simulate.R`
(grid r2-real), and the accepted outbreak size."""


def _number(x: float) -> str:
    """Return a number to three significant figures, without exponents."""
    if abs(x) >= 1000:
        return f"{x:,.0f}"
    return np.format_float_positional(
        x, precision=3, unique=False, fractional=False, trim="-"
    )


def _realised(name: str, values: pd.Series) -> str:
    """Return a setting's realised values over outbreaks.

    Numbers give the median and range, yes or no settings the share of
    yes, and other levels the share of each.
    """
    if name in ("report_days", "report_days_after"):
        return f"{(values.astype(str) == 'all').mean():.0%} every day"
    if values.dtype == bool or set(values.unique()) <= {True, False}:
        return f"{values.astype(bool).mean():.0%} yes"
    if not pd.api.types.is_numeric_dtype(values):
        shares = values.value_counts(normalize=True)
        return ", ".join(f"{k} {v:.0%}" for k, v in shares.items())
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
    first = [row.split(";")[0] for row in out["setting"]]
    out["realised"] = [
        _realised(name, settings[name].dropna()) if name in settings else ""
        for name in first
    ]
    return out[list(COLUMNS)]


MAIN_HEADERS: Final = {
    "factor": "Design",
    "levels": "Levels and shares",
    "epinowcast": "Enters through",
    "realised": "Realised",
}
"""Headers of the main-text table."""

MAIN: Final = pd.DataFrame(
    [
        (
            "strata",
            "Strata",
            "One place 30%; 2 to 26 places 40%; 2 to 8 age groups in one place"
            " 20%; 6 to 16 places by 3 to 7 age groups 10%",
            "`enw_preprocess_data(by)`",
        ),
        (
            "process",
            "Generative process",
            "Renewal, renewal with incubation, weekly growth walk, log "
            "incidence walk, growth steps or smooth growth, one sixth each",
            "`enw_expectation()`, `enw_simulate(growth_rate)`",
        ),
        (
            "reference",
            "Delay model",
            "Parametric 30%; non-parametric hazard 20%; both 50%",
            "`enw_reference()`",
        ),
        (
            "delay_family",
            "Parametric delay",
            "Lognormal, gamma or exponential, one third each",
            "`enw_reference(distribution)`",
        ),
        (
            "hazard",
            "Hazard effects",
            "By delay, by week or both, one third each; by delay if the delay "
            "is stable",
            "`enw_reference(non_parametric)`",
        ),
        (
            "stable;delay_drift;spread_drift;delay_steps;delay_walk",
            "Delay changes",
            "Stable 20%; otherwise drift in location 60%, drift in spread 50%,"
            " 0 to 3 steps (30%, 30%, 20%, 20%) and a weekly walk 30%",
            "`enw_reference(parametric)`",
        ),
        (
            "report_dow;report_holiday;report_week",
            "Report date effects",
            "Day of week 70%; holidays 30%; report week effects none 60%, `(1 "
            "| week)` 20%, `rw(week)` 20%",
            "`enw_report()`",
        ),
        (
            "report_days;report_change;report_days_after",
            "Report days",
            "Every day 50%; 1 to 3 weekdays 50%; a change of days 30% if not "
            "stable",
            "`enw_report(structural)`",
        ),
        (
            "family",
            "Observation model",
            "NB2 40%; NB1 40%; Poisson 20%",
            "`enw_obs(family)`",
        ),
        (
            "max_delay",
            "Longest delay",
            "8 weeks 75%; 12 weeks 25%",
            "`enw_preprocess_data(max_delay)`",
        ),
    ],
    columns=["setting", "factor", "levels", "epinowcast"],
)
"""One row per design branch of r2-real, for the main text; `GRID` has
every setting, for the SI."""


def main_table(settings: pd.DataFrame) -> pd.DataFrame:
    """Return `MAIN` with each branch's realised shares over outbreaks."""
    out = MAIN.copy()
    first = [row.split(";")[0] for row in out["setting"]]
    out["realised"] = [
        _realised(name, settings[name].dropna()) if name in settings else ""
        for name in first
    ]
    return out


def to_markdown(
    table: pd.DataFrame, headers: dict[str, str] | None = None
) -> str:
    """Return the table as Markdown, without the `setting` key."""
    headers = _HEADERS if headers is None else headers
    cols = list(headers)
    lines = [
        "| " + " | ".join(headers[c] for c in cols) + " |",
        "|" + "---|" * len(cols),
    ]
    for _, row in table.iterrows():
        cells = [str(row[c]).replace("|", "\\|") for c in cols]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def main() -> None:
    """Write the main-text table and the SI table, as CSV and Markdown."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sim_dir", type=Path)
    parser.add_argument("out_dir", type=Path)
    args = parser.parse_args()
    settings = pd.concat(
        [pd.read_csv(p) for p in sorted(args.sim_dir.glob("settings-*.csv"))]
    )
    write_tables(settings, args.out_dir)


def write_tables(settings: pd.DataFrame, out_dir: Path) -> None:
    """Write `synthetic-grid` (main text) and `synthetic-grid-si` tables.

    Each is a CSV and a Markdown table that the paper includes under its
    own caption.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, table, headers in (
        ("synthetic-grid", main_table(settings), MAIN_HEADERS),
        ("synthetic-grid-si", grid_table(settings), _HEADERS),
    ):
        table.to_csv(out_dir / f"{name}.csv", index=False)
        (out_dir / f"{name}.md").write_text(to_markdown(table, headers))


if __name__ == "__main__":
    main()
