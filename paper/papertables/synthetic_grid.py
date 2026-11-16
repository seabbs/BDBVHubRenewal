"""The synthetic scenario grid tables and SI notes, read from the simulator.

`scripts/finetune_simulate.R` defines the r3 grid: a crossed design of
`factors`, each level balanced in every block of simulations, and
continuous settings drawn within a cell by `draw_values`. Everything here
is read from that code, so the tables and notes cannot drift from it;
`LABELS` only names each setting and the epinowcast function it enters
through, and a test fails if a setting has no label.

- `main_table`: one row per factor, with its levels (main text);
- `si_table`: every factor, derived setting and drawn value, with the R
  expression that draws it (SI);
- `model_note` and `heldout_note`: the SI notes on the epinowcast model
  and on the held-out set (the `synthetic` block of the workflow config).

The workflow's `grid_tables` rule runs it, with the simulations whose
settings give the realised values when they exist.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Final

import numpy as np
import pandas as pd

__all__ = [
    "DERIVED",
    "LABELS",
    "MAIN_HEADERS",
    "SI_HEADERS",
    "heldout_note",
    "main_table",
    "model_note",
    "parse_factors",
    "parse_values",
    "si_table",
    "to_markdown",
]

DERIVED: Final = ("n_places", "n_ages", "report_days")
"""Settings `draw_design` derives from a cell's strata and reporting."""

LABELS: Final = {
    # Factors of the crossed design.
    "process": ("Generative process", "`enw_expectation()`"),
    "gt_shape": ("Generation time shape", "`enw_expectation(generation_time)`"),
    "trajectory": (
        "Trajectory of the growth rate",
        "`enw_simulate(growth_rate)`",
    ),
    "delay_family": (
        "Parametric delay family",
        "`enw_reference(distribution)`",
    ),
    "delay_form": ("Delay form", "`enw_reference(parametric, non_parametric)`"),
    "delay_change": ("Delay over time", "`enw_reference()`"),
    "reporting": ("Report days", "`enw_report(structural)`"),
    "reference_dow": (
        "Day-of-week effect on expected counts",
        "`enw_expectation(observation)`",
    ),
    "observation": ("Observation model", "`enw_obs(family)`"),
    "strata": ("Strata", "`enw_preprocess_data(by)`"),
    # Derived from the cell.
    "n_places": ("Places", "`enw_preprocess_data(by)`"),
    "n_ages": ("Age groups", "`enw_preprocess_data(by)`"),
    "report_days": ("Weekdays with reports", "`enw_report(structural)`"),
    # Drawn within the cell.
    "weeks": ("Weeks of cut-offs", "`enw_preprocess_data()` grid"),
    "offset": ("Start day offset (days)", "`enw_preprocess_data()` grid"),
    "quiet": ("Probability a place is quiet", "`expr_lelatent_int`"),
    "coupling": (
        "Weight of the shared growth path",
        "`enw_simulate(growth_rate)`",
    ),
    "stratum_sd": (
        "Sd of the strata's log seed weights",
        "`expr_lelatent_int`",
    ),
    "gt_mean": (
        "Generation time mean (days)",
        "`enw_expectation(generation_time)`",
    ),
    "rate": (
        "Steady daily growth or decline rate",
        "`enw_simulate(growth_rate)`",
    ),
    "flat": (
        "Daily growth rate of a flat trajectory",
        "`enw_simulate(growth_rate)`",
    ),
    "peak": (
        "Time of the peak or trough, as a share",
        "`enw_simulate(growth_rate)`",
    ),
    "amplitude": (
        "Amplitude of oscillations and waves",
        "`enw_simulate(growth_rate)`",
    ),
    "period": ("Period of oscillations (days)", "`enw_simulate(growth_rate)`"),
    "phase": ("Phase of oscillations and waves", "`enw_simulate(growth_rate)`"),
    "waves": (
        "Waves of a multi-wave trajectory",
        "`enw_simulate(growth_rate)`",
    ),
    "noise_sd": (
        "Sd of the weekly AR(1) growth noise",
        "`enw_simulate(growth_rate)`",
    ),
    "target_size": (
        "Expected outbreak size, which scales the seed",
        "`expr_lelatent_int`",
    ),
    "delay_median": ("Lognormal delay median (days)", "`refp_mean_int`"),
    "delay_sdlog": ("Lognormal delay log sd", "`refp_sd_int`"),
    "delay_mean": (
        "Gamma, exponential or geometric delay mean (days)",
        "`refp_mean_int` or the hazard intercept",
    ),
    "delay_cv": ("Gamma delay coefficient of variation", "`refp_sd_int`"),
    "delay_place_sd": (
        "Sd of the places' delay effects",
        "`(1 | place)` of `enw_reference()`",
    ),
    "delay_age_sd": (
        "Sd of the age groups' delay effects",
        "`(1 | age)` of `enw_reference()`",
    ),
    "hazard_walk_sd": (
        "Sd of the smooth walk over delays",
        "`enw_reference(non_parametric)`",
    ),
    "hazard_sd": (
        "Sd of the iid hazard effects by delay",
        "`enw_reference(non_parametric)`",
    ),
    "drift": (
        "Delay drift over the outbreak",
        "`trend` covariate of `enw_reference()`",
    ),
    "step": ("Delay step size", "`step` covariate of `enw_reference()`"),
    "step_at": (
        "Time of the delay step, as a share",
        "Step covariate of `enw_reference()`",
    ),
    "periodic": (
        "Amplitude of periodic delay change",
        "`periodic` covariate of `enw_reference()`",
    ),
    "periodic_weeks": (
        "Period of periodic delay change (weeks)",
        "`enw_reference()`",
    ),
    "walk_sd": (
        "Sd of the weekly delay walk",
        "`rw(week)` of `enw_reference()`",
    ),
    "reference_dow_sd": (
        "Sd of the day-of-week effect on expected counts",
        "`enw_expectation(observation)`",
    ),
    "dow_sd": (
        "Sd of the report day-of-week effects",
        "`(1 | day_of_week)` of `enw_report()`",
    ),
    "sqrt_phi": ("Overdispersion on the NB2 scale", "`enw_obs()`"),
    "max_delay": (
        "Longest delay (days), from the delay's 99th centile",
        "`enw_preprocess_data(max_delay)`",
    ),
}
"""Each setting's label and the epinowcast function it enters through."""

MAIN_HEADERS: Final = {
    "factor": "Factor",
    "levels": "Levels",
    "epinowcast": "Enters through",
    "realised": "Realised",
}
"""Headers of the main-text table."""

SI_HEADERS: Final = {
    "factor": "Setting",
    "drawn": "Drawn as",
    "epinowcast": "Enters through",
    "realised": "Realised, median [min, max] or shares",
}
"""Headers of the SI table."""


def parse_factors(text: str) -> dict[str, list[str]]:
    """Return the simulator's design factors and their levels, in order."""
    body = text[text.index("factors <- list(") :]
    body = body[: body.index("\n)\n")]
    return {
        name: re.findall(r'"([^"]+)"', levels)
        for name, levels in re.findall(r"(\w+) = c\(([^)]*)\)", body)
    }


def _function(text: str, name: str) -> str:
    body = text[text.index(f"{name} <- function") :]
    return body[: body.index("\n}\n")]


def parse_values(text: str) -> dict[str, str]:
    """Return the settings `draw_values` draws and their R expressions."""
    body = _function(text, "draw_values")
    block = body[body.index("  v <- list(\n") :]
    block = block[len("  v <- list(\n") : block.index("\n  )\n")]
    out: dict[str, list[str]] = {}
    name = ""
    for line in block.split("\n"):
        if line.strip().startswith("#"):
            continue
        m = re.match(r"^    (\w+) = (.*)$", line)
        if m:
            name = m.group(1)
            out[name] = [m.group(2)]
        elif name:
            out[name].append(line.strip())
    values = {k: " ".join(v).rstrip(",").strip() for k, v in out.items()}
    m = re.search(r"v\$max_delay <- (.*)\n", body)
    if m:
        values["max_delay"] = m.group(1).strip()
    return values


def _derived(text: str) -> dict[str, str]:
    """Return the R expressions `draw_design` derives settings with."""
    body = _function(text, "draw_design")
    out = {}
    for name in ("n_places", "n_ages"):
        m = re.search(rf"  {name} <- (switch\(.*?\n  \))", body, flags=re.S)
        out[name] = " ".join(m.group(1).split()) if m else ""
    m = re.search(r"report_days = (if .*?else \{.*?\})", body, flags=re.S)
    out["report_days"] = " ".join(m.group(1).split()) if m else ""
    return out


def _number(x: float) -> str:
    """Return a number to three significant figures, without exponents."""
    if abs(x) >= 1000:
        return f"{x:,.0f}"
    return np.format_float_positional(
        x, precision=3, unique=False, fractional=False, trim="-"
    )


def _realised(name: str, settings: pd.DataFrame) -> str:
    """Return a setting's realised values over simulations, or ''."""
    if name not in settings:
        return ""
    values = settings[name].dropna()
    if values.empty:
        return ""
    if name == "report_days":
        return f"{(values.astype(str) == 'all').mean():.0%} every day"
    if not pd.api.types.is_numeric_dtype(values):
        shares = values.astype(str).value_counts(normalize=True)
        return ", ".join(f"{k} {v:.0%}" for k, v in shares.items())
    out = (
        f"{_number(values.median())} "
        f"[{_number(values.min())}, {_number(values.max())}]"
    )
    if name == "n_places":
        out += f"; {(values == 1).mean():.0%} one place"
    return out


def main_table(text: str, settings: pd.DataFrame) -> pd.DataFrame:
    """Return one row per design factor, with its levels and shares."""
    rows = []
    for name, levels in parse_factors(text).items():
        label, enters = LABELS[name]
        listed = ", ".join(f"`{level}`" for level in levels)
        rows.append(
            {
                "setting": name,
                "factor": label,
                "levels": f"{listed}; balanced, 1/{len(levels)} each",
                "epinowcast": enters,
                "realised": _realised(name, settings),
            }
        )
    return pd.DataFrame(rows)


def si_table(text: str, settings: pd.DataFrame) -> pd.DataFrame:
    """Return every factor, derived setting and drawn value."""
    drawn = {
        **{
            name: "One of " + ", ".join(f"`{x}`" for x in levels)
            for name, levels in parse_factors(text).items()
        },
        **{k: f"`{v}`" for k, v in _derived(text).items()},
        **{k: f"`{v}`" for k, v in parse_values(text).items()},
    }
    return pd.DataFrame(
        [
            {
                "setting": name,
                "factor": LABELS[name][0],
                "drawn": how,
                "epinowcast": LABELS[name][1],
                "realised": _realised(name, settings),
            }
            for name, how in drawn.items()
        ]
    )


def to_markdown(table: pd.DataFrame, headers: dict[str, str]) -> str:
    """Return the table as Markdown, without the `setting` key."""
    cols = list(headers)
    lines = [
        "| " + " | ".join(headers[c] for c in cols) + " |",
        "|" + "---|" * len(cols),
    ]
    for _, row in table.iterrows():
        cells = [str(row[c]).replace("|", "\\|") for c in cols]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def _sha(text: str) -> str:
    m = re.search(r'epinowcast_sha <- "(\w+)"', text)
    return m.group(1) if m else "@placeholder"


def model_note(text: str) -> str:
    """Return the SI note on the epinowcast model that simulates."""
    grid = re.search(r'grid_name <- "([\w-]+)"', text)
    factors = parse_factors(text)
    lines = [
        "#### The epinowcast model used to simulate outbreaks",
        "",
        "We simulated each synthetic outbreak from the generative model of"
        " epinowcast @placeholder.",
        "We used `enw_simulate()`, which runs the epinowcast Stan model with"
        " fixed parameters and no likelihood, and draws reported counts from"
        " its generated quantities.",
        "`enw_simulate()` is on the `feature/829-simulate-forecast` branch"
        f" of epinowcast, at commit `{_sha(text)}`.",
        "The simulation script is `scripts/finetune_simulate.R`, with grid"
        f" `{grid.group(1) if grid else ''}`.",
        "Each simulation is one set of fixed parameters with several"
        " observation draws, and we evaluate one draw of each.",
        "Simulations follow a crossed design, in which every level of every"
        " factor is balanced within each block of simulations.",
        "Continuous settings are drawn within a simulation's cell, as the"
        " SI table lists.",
        "A parameter we did not set is drawn from its epinowcast prior.",
        "",
        "##### Design factors",
        "",
    ]
    for name, levels in factors.items():
        label, enters = LABELS[name]
        listed = ", ".join(f"`{level}`" for level in levels)
        lines.append(f"- {label} (`{name}`, through {enters}): {listed}.")
    lines += [
        "",
        "##### Model",
        "",
        "Expected onsets come from `enw_expectation()`, driven by a daily"
        " growth rate given to `enw_simulate()` as `growth_rate`.",
        "In a renewal process the growth rate is turned into log $R_t$"
        " under the generation time, and expected latent cases follow"
        " $\\lambda_t = R_t \\sum_s g_s \\lambda_{t-s}$.",
        "The seed is scaled so the expected outbreak size is a drawn target.",
        "The reporting delay is set by `enw_reference()`, as a parametric"
        " delay, a non-parametric hazard over delays, or both, and it may"
        " change over time.",
        "Report date effects and report days are set by `enw_report()`.",
        "Counts by onset day and delay are negative binomial (NB2 or NB1)"
        " or Poisson, set by `enw_obs()`, with the NB1 dispersion matched"
        " to the NB2 scale (epinowcast issue 937).",
        "The longest delay is the shortest of 3, 4, 6, 8 and 12 weeks at or"
        " above the delay's 99th centile.",
        "A draw that fails or is out of size is drawn again in its cell, up"
        " to five times, after which the cell is skipped.",
    ]
    return "\n".join(lines) + "\n"


def _seeds(seeds: list[int]) -> str:
    """Return seeds as `seed 7001` or `seeds 8016 to 8025`."""
    if len(seeds) == 1:
        return f"seed {seeds[0]}"
    return f"seeds {seeds[0]} to {seeds[-1]}"


def _ordinal(n: int) -> str:
    """Return `n` as an ordinal, such as 6th."""
    suffix = {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
    if 10 <= n % 100 <= 20:
        suffix = "th"
    return f"{n}{suffix}"


def heldout_note(config: dict) -> str:
    """Return the SI note on the held-out set, from its config.

    `config` is the workflow config's `synthetic` block with the run's
    `heldout_seeds` and `development_seeds`.
    """
    protocol = config["protocol"]
    min_releases = protocol["min_releases"]
    horizon_weeks = protocol["horizon_weeks"]
    horizons = protocol["horizons"]
    seeds = config["heldout_seeds"]
    n = config["simulations_per_seed"]
    blocks = config["blocks"]
    dev = config["development_seeds"]
    lines = [
        "#### The held-out synthetic set",
        "",
        f"The held-out set is simulated on grid `{config['grid']}` with"
        f" {_seeds(seeds)}, {n} simulations each.",
        f"These cover the complete balanced blocks {blocks[0]} to"
        f" {blocks[-1]}, and simulations outside them are not used.",
        f"We evaluate draw {config['draw']} of each simulation, so the"
        " outbreaks are independent.",
        "The seeds are disjoint from the fine-tune's training and held-out"
        " seeds (`config/finetune/r3.env`).",
        f"The simulations are written to `{config['simulations']['paper']}`,"
        " and the workflow config records them.",
        "The synthetic runner runs the models on the set.",
        "",
        "The held-out set was reserved for the final evaluation.",
        "No model was scored on it during development.",
        f"We checked the pipeline on the r3 pilot {_seeds(dev)}, whose"
        " forecasts are not results.",
        "",
        "##### Protocol",
        "",
        "Each outbreak is released weekly, on one weekday from Wednesday to"
        " Saturday drawn from its id, and a release's cut-off is its origin.",
        "Each outbreak has four origins, spread evenly from its"
        f" {_ordinal(min_releases)} release to the last release at least its"
        f" longest delay plus {horizon_weeks} weeks before its last report.",
        f"Each forecast covers weeks {horizons[0]} to {horizons[-1]} from"
        " the week of the cut-off, as weekly totals by onset week for the"
        " country and each stratum.",
        "Scores use rolling truth, the count reported D days after the end"
        " of a target week, with D the outbreak's own 98% reporting delay in"
        " whole days, and are relative to `randomwalknowcast`.",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    """Write the grid tables and the SI notes."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--simulator", type=Path, required=True)
    parser.add_argument(
        "--config", required=True, help="The `synthetic` block as JSON."
    )
    parser.add_argument("--settings", type=Path, default=None)
    parser.add_argument("--main-dir", type=Path, required=True)
    parser.add_argument("--si-dir", type=Path, required=True)
    args = parser.parse_args()
    text = args.simulator.read_text()
    files = (
        sorted(args.settings.glob("settings-*.csv")) if args.settings else []
    )
    settings = (
        pd.concat([pd.read_csv(p) for p in files])
        if files
        else pd.DataFrame({"outbreak": []})
    )
    for folder in (args.main_dir, args.si_dir):
        folder.mkdir(parents=True, exist_ok=True)
    tables = (
        (args.main_dir / "synthetic-design.md", main_table, MAIN_HEADERS),
        (args.si_dir / "synthetic-design.md", si_table, SI_HEADERS),
    )
    for path, build, headers in tables:
        path.write_text(to_markdown(build(text, settings), headers))
    (args.si_dir / "epinowcast-model.md").write_text(model_note(text))
    config = json.loads(args.config)
    (args.si_dir / "heldout-set.md").write_text(heldout_note(config))


if __name__ == "__main__":
    main()
