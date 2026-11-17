"""The synthetic scenario grid tables and SI notes, read from the simulator.

`scripts/finetune_simulate.R` defines the grid: a crossed design of
`factors`, with levels balanced or repeated for their shares, and
continuous settings drawn within a cell by `draw_values`. Everything here
is read from that code, so the tables and notes cannot drift from it;
`LABELS` names each setting, `FACTOR_OF` and `LAYER` place it, and
`SUMMARIES` gives plain text for draws that are not a simple call, each
failing if its R changes.

- `layer_tables`: the SI's two grid tables, epidemic and places, and
  reporting and observation, with each factor's levels and continuous
  draws;
- `model_note` and `heldout_note`: the SI notes on the epinowcast model
  and on the held-out set (the `synthetic` block of the workflow config).

The workflow's `grid_tables` rule runs it.
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
    "heldout_note",
    "layer_tables",
    "model_note",
    "parse_bounds",
    "parse_factors",
    "parse_shares",
    "parse_values",
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
    "spread": ("How cases spread over strata", "`stratum_seed()`"),
    "place_rt": ("How places' growth paths relate", "`place_growth()`"),
    "seeding": ("When places start", "`late_strata()`"),
    "mixing": ("How places infect each other", "`place_mixing()`"),
    "mixing_strength": (
        "Strength of mixing between places",
        "`place_mixing()`",
    ),
    "imports": ("Imported cases", "`stratum_imports()`"),
    "shocks": ("Reporting shocks", "`shock_window()`"),
    "delay_places": ("How places' delays relate", "`delay_parameters()`"),
    "obs_strata": ("Observation noise across strata", "`observe()`"),
    "strata_link": ("Linked or independent strata", "`simulate_independent()`"),
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
    "place_rho": (
        "Correlation of places' growth deviations",
        "`place_growth()`",
    ),
    "place_sd": ("Sd of places' daily growth deviations", "`place_growth()`"),
    "place_lags": (
        "Lag of each place's growth path (days)",
        "`place_growth()`",
    ),
    "late_at": ("Time a late place starts, as a share", "`late_strata()`"),
    "late_days": ("Days over which a late place starts", "`late_strata()`"),
    "shock_at": ("Time of a reporting shock, as a share", "`shock_window()`"),
    "shock_places": (
        "Places with shocks, and shocks in each",
        "`shock_window()`",
    ),
    "wave_scale": ("Relative size of each wave", "`enw_simulate(growth_rate)`"),
    "size_per_stratum": ("Expected cases per stratum", "`expected_scale`"),
    "bunch_every": ("Weeks between bunched reports", "`enw_reference()`"),
    "bunch_at": ("Times of bunched reports, as shares", "`enw_report()`"),
    "bunch_len": ("Report dates held in a bunch", "`enw_report()`"),
    "slow_at": ("Time of a slowdown, as a share", "`enw_reference()`"),
    "slow_weeks": ("Length of a slowdown (weeks)", "`enw_reference()`"),
    "slow_mult": ("Delay multiplier in a slowdown", "`enw_reference()`"),
    "obs_phi_sd": ("Sd of strata's log overdispersion", "`observe()`"),
    "shock_weeks": ("Length of a reporting shock (weeks)", "`shock_window()`"),
    "shock_mult": ("Delay multiplier of a backlog", "`shock_window()`"),
    "mixing_eps": ("Share of infections from other places", "`place_mixing()`"),
    "mixing_order": (
        "Order of places for asymmetric mixing",
        "`place_mixing()`",
    ),
    "mixing_x": ("Places' x coordinates", "`place_mixing()`"),
    "mixing_y": ("Places' y coordinates", "`place_mixing()`"),
    "mixing_range": ("Range of distance-based mixing", "`place_mixing()`"),
    "mixing_switch": ("Whether mixing switches on or off", "`place_mixing()`"),
    "mixing_switch_at": (
        "Time mixing switches, as a share",
        "`place_mixing()`",
    ),
    "mixing_switch_days": (
        "Days over which mixing switches",
        "`place_mixing()`",
    ),
    "import_rate": ("Daily rate of imported cases", "`stratum_imports()`"),
    "import_at": ("Times of import pulses, as shares", "`stratum_imports()`"),
    "import_size": ("Size of import pulses", "`stratum_imports()`"),
    "max_delay": (
        "Longest delay (days), from the delay's 99th centile",
        "`enw_preprocess_data(max_delay)`",
    ),
    # Bounds on a simulation; one outside them is drawn again.
    "min_size": (
        "Fewest cases (the draws' mean) a simulation keeps",
        "`simulate_cell()`",
    ),
    "max_size": (
        "Most cases (the draws' mean) a simulation keeps",
        "`simulate_cell()`",
    ),
    "max_independent": (
        "Most independent strata",
        "`simulate_independent()`",
    ),
}
"""Each setting's label and the epinowcast function it enters through."""


def _factor_items(text: str) -> dict[str, list[str]]:
    """Return each factor's levels as listed, repeats included."""
    body = text[text.index("factors <- list(") :]
    body = body[: body.index("\n)\n")]
    out = {}
    for m in re.finditer(r"(\w+) = c\(", body):
        depth, k = 1, m.end()
        while depth:
            depth += {"(": 1, ")": -1}.get(body[k], 0)
            k += 1
        items: list[str] = []
        for rep_level, times, level in re.findall(
            r'rep\("([^"]+)",\s*(\d+)\)|"([^"]+)"', body[m.end() : k - 1]
        ):
            items += [rep_level] * int(times) if rep_level else [level]
        out[m.group(1)] = items
    return out


def parse_factors(text: str) -> dict[str, list[str]]:
    """Return the simulator's design factors and their levels, in order."""
    return {
        name: list(dict.fromkeys(items))
        for name, items in _factor_items(text).items()
    }


def parse_shares(text: str) -> dict[str, dict[str, float]]:
    """Return each factor's levels and their shares of the design."""
    return {
        name: {
            level: items.count(level) / len(items)
            for level in dict.fromkeys(items)
        }
        for name, items in _factor_items(text).items()
    }


def _function(text: str, name: str) -> str:
    body = text[text.index(f"{name} <- function") :]
    return body[: body.index("\n}\n")]


DRAW: Final = r"\b(runif|rnorm|sample|log_uniform)\("
"""R calls that make an expression a random draw."""


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
    # Draws set after the list, one line each.
    for name, expr in re.findall(r"^  v\$(\w+) <- (.+)$", body, flags=re.M):
        if name not in values and re.search(DRAW, expr):
            values[name] = expr.strip()
    return values


def parse_bounds(text: str) -> dict[str, str]:
    """Return the simulator's size bounds and their R expressions."""
    out = {}
    for name in ("min_size", "max_independent"):
        m = re.search(rf"^{name} <- (.*)$", text, flags=re.M)
        if m:
            out[name] = m.group(1).strip()
    m = re.search(
        r"^max_size <- function\(design\) \{\n(.*?)\n\}", text, re.M | re.S
    )
    if m:
        out["max_size"] = " ".join(m.group(1).split())
    return {
        k: out[k]
        for k in ("min_size", "max_size", "max_independent")
        if k in out
    }


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


LAYER: Final = {
    **dict.fromkeys(
        (
            "process",
            "gt_shape",
            "trajectory",
            "strata",
            "spread",
            "place_rt",
            "seeding",
            "mixing",
            "imports",
            "strata_link",
        ),
        "epidemic",
    ),
    **dict.fromkeys(
        (
            "delay_family",
            "delay_form",
            "delay_change",
            "reporting",
            "reference_dow",
            "shocks",
            "delay_places",
            "observation",
            "obs_strata",
        ),
        "reporting",
    ),
}
"""The SI table each design factor is in: epidemic and places, or reporting
and observation."""

FACTOR_OF: Final = {
    **{name: name for name in LAYER},
    **dict.fromkeys(("n_places", "n_ages", "min_size", "max_size"), "strata"),
    "size_per_stratum": "strata",
    "stratum_sd": "strata",
    "report_days": "reporting",
    "offset": "reporting",
    "dow_sd": "reporting",
    "gt_mean": "gt_shape",
    **dict.fromkeys(
        (
            "weeks",
            "rate",
            "flat",
            "peak",
            "amplitude",
            "period",
            "phase",
            "waves",
            "wave_scale",
            "noise_sd",
        ),
        "trajectory",
    ),
    "quiet": "spread",
    **dict.fromkeys(
        ("coupling", "place_rho", "place_sd", "place_lags"), "place_rt"
    ),
    "max_independent": "strata_link",
    **dict.fromkeys(("late_at", "late_days"), "seeding"),
    **dict.fromkeys(
        (
            "mixing_eps",
            "mixing_order",
            "mixing_x",
            "mixing_y",
            "mixing_range",
            "mixing_switch",
            "mixing_switch_at",
            "mixing_switch_days",
        ),
        "mixing",
    ),
    **dict.fromkeys(("import_rate", "import_at", "import_size"), "imports"),
    **dict.fromkeys(
        ("delay_median", "delay_sdlog", "delay_mean", "delay_cv", "max_delay"),
        "delay_family",
    ),
    "hazard_sd": "delay_form",
    **dict.fromkeys(
        (
            "drift",
            "step",
            "step_at",
            "periodic",
            "periodic_weeks",
            "walk_sd",
            "hazard_walk_sd",
            "bunch_every",
            "bunch_at",
            "bunch_len",
            "slow_at",
            "slow_weeks",
            "slow_mult",
        ),
        "delay_change",
    ),
    "reference_dow_sd": "reference_dow",
    **dict.fromkeys(
        ("shock_places", "shock_at", "shock_weeks", "shock_mult"), "shocks"
    ),
    **dict.fromkeys(("delay_place_sd", "delay_age_sd"), "delay_places"),
    "sqrt_phi": "observation",
    "obs_phi_sd": "obs_strata",
}
"""The design factor each derived setting, draw and bound belongs to."""

HEADERS: Final = {
    "factor": "Factor",
    "levels": "Levels",
    "draws": "Continuous draws",
}
"""Headers of the SI grid tables."""


SUMMARIES: Final = {
    "n_places": (
        (
            "switch(cell$strata, single = 1L, ages = 1L, # Concentrated o"
            "utbreaks spread over many places, as BDBV's provinces. place"
            's = if (cell$spread == "concentrated") { sample(15:26, 1) } '
            "else { sample(2:26, 1) }, places_ages = { places <- 6:min(16"
            "L, cap %/% 3L) places[sample.int(length(places), 1)] } )"
        ),
        (
            "1 for single or ages; {15, ..., 26} concentrated or {2, ...,"
            " 26} even for places; {6, ..., 16} for places and ages, with"
            "in the strata cap"
        ),
    ),
    "n_ages": (
        (
            "switch(cell$strata, single = 1L, places = 1L, ages = sample("
            "2:8, 1), # At most `max_strata`; `sample` on a single number"
            " n draws from 1:n, # so the range is indexed instead. places"
            "_ages = { ages <- 3:max(3L, min(7L, cap %/% n_places)) ages["
            "sample.int(length(ages), 1)] } )"
        ),
        (
            "1 for single or places; {2, ..., 8} for ages; {3, ..., 7} fo"
            "r places and ages, within the strata cap"
        ),
    ),
    "report_days": (
        (
            'if (cell$reporting == "weekdays") { paste(sort(sample(days, '
            'sample(1:3, 1))), collapse = ";") } else { "all" }'
        ),
        "every day, or {1, ..., 3} weekdays",
    ),
    "weeks": (
        ("if (is.null(design$weeks)) sample(16:52, 1) else design$weeks"),
        "{16, ..., 52}",
    ),
    "quiet": (
        (
            'if (design$spread == "concentrated") { 1 - active / design$n'
            "_places } else { runif(1, 0, 0.9) }"
        ),
        "all but the active places if concentrated; U(0, 0.9) if even",
    ),
    "place_lags": (
        "c(0L, sample(1:28, design$n_places - 1L, replace = TRUE))",
        "0 for the first place, {1, ..., 28} for the others",
    ),
    "mixing_order": (
        "sample.int(design$n_places)",
        "a random order",
    ),
    "mixing_x": (
        "runif(design$n_places)",
        "U(0, 1) each",
    ),
    "mixing_y": (
        "runif(design$n_places)",
        "U(0, 1) each",
    ),
    "mixing_switch": (
        'sample(c("on", "off"), 1)',
        "on or off",
    ),
    "import_at": (
        "sort(runif(sample(1:4, 1)))",
        "{1, ..., 4} times, U(0, 1) each",
    ),
    "stratum_sd": (
        (
            'if (design$spread == "concentrated") { runif(1, 2.5, 4) } el'
            "se { runif(1, 0.5, 2) }"
        ),
        "U(2.5, 4) if concentrated; U(0.5, 2) if even",
    ),
    "gt_mean": (
        (
            'if (renewal) { as.numeric(sub("renewal_", "", design$process'
            ")) * exp(runif(1, -0.2, 0.2)) } else { NA_real_ }"
        ),
        "the process's mean (3, 7 or 15) times exp(U(-0.2, 0.2))",
    ),
    "waves": (
        "waves <- sample(2:3, 1)",
        "{2, 3}",
    ),
    "wave_scale": (
        "exp(runif(waves + 1L, log(0.2), 0))",
        "exp(U(log 0.2, 0)) each",
    ),
    "drift": (
        "sample(c(-1, 1), 1) * runif(1, 0.1, 0.4)",
        "±U(0.1, 0.4)",
    ),
    "step": (
        "sample(c(-1, 1), 1) * runif(1, 0.2, 0.5)",
        "±U(0.2, 0.5)",
    ),
    "max_delay": (
        "max_delay_for(delay_parameters(family, v)$q99)",
        "the delay's 99th centile",
    ),
    "bunch_at": (
        "sort(runif(n_bunch, 0.05, 0.95))",
        "U(0.05, 0.95) each, one per bunch",
    ),
    "shock_places": (
        "rep(places, sample(1:3, length(places), replace = TRUE))",
        "{1, ..., 3} places with cases, {1, ..., 3} shocks each",
    ),
    "min_size": (
        "10",
        "10",
    ),
    "max_size": (
        "1e5 * design$n_places * design$n_ages",
        "10^5 per stratum",
    ),
    "max_independent": (
        "8L",
        "8",
    ),
}
"""Plain summaries of the draws whose R is not a simple call, each with the
exact expression it summarises, so a changed draw fails until its summary
is updated."""


def describe(name: str, expr: str) -> str:
    """Return a draw in short notation, or its plain summary.

    Raises
    ------
    ValueError
        If the simulator's expression differs from the one summarised.
    """
    if name not in SUMMARIES:
        return compact(expr)
    expected, text = SUMMARIES[name]
    if " ".join(expr.split()) != expected:
        raise ValueError(f"{name}'s draw changed; update SUMMARIES")
    return text


def compact(expr: str) -> str:
    """Return a draw's R expression in short notation where it is simple."""
    num = r"([-\w.()/*+ ]+?)"
    for pattern, out in (
        (rf"^runif\(\w+, {num}, {num}\)$", "U({0}, {1})"),
        (rf"^log_uniform\({num}, {num}\)$", "log-uniform({0}, {1})"),
        (rf"^rnorm\(\w+, {num}, {num}\)$", "N({0}, {1})"),
        (
            r"^sample\((\d+):(\d+), \w+(?:, replace = TRUE)?\)$",
            "{{{0}, ..., {1}}}",
        ),
    ):
        m = re.match(pattern, expr)
        if m:
            return out.format(*m.groups())
    return f"`{expr}`"


def _levels(shares: dict[str, float]) -> str:
    listed = ", ".join(f"`{level}`" for level in shares)
    if len(set(shares.values())) == 1:
        return listed
    return ", ".join(f"`{lv}` {share:.0%}" for lv, share in shares.items())


def layer_tables(text: str) -> dict[str, pd.DataFrame]:
    """Return the grid's two SI tables, epidemic and reporting.

    One row per design factor, with its levels (and their shares where
    they differ) and its continuous draws, each as `FACTOR_OF` assigns
    them. A factor or draw with no layer fails, so a new one is placed.
    """
    draws: dict[str, list[str]] = {}
    exprs = {
        **_derived(text),
        **parse_values(text),
        **parse_bounds(text),
    }
    for name, expr in exprs.items():
        if name not in FACTOR_OF:
            raise KeyError(f"{name} has no factor in FACTOR_OF")
        draws.setdefault(FACTOR_OF[name], []).append(
            f"{LABELS[name][0]}: {describe(name, expr)}"
        )
    tables: dict[str, list[dict]] = {"epidemic": [], "reporting": []}
    for name, shares in parse_shares(text).items():
        tables[LAYER[name]].append(
            {
                "setting": name,
                "factor": LABELS[name][0],
                "levels": _levels(shares),
                "draws": "; ".join(draws.get(name, [])),
            }
        )
    return {k: pd.DataFrame(v) for k, v in tables.items()}


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
    blocks = config.get("blocks") or []
    dev = config["development_seeds"]
    cover = (
        [
            f"These cover the complete balanced blocks {blocks[0]} to"
            f" {blocks[-1]}, and simulations outside them are not used."
        ]
        if blocks
        else []
    )
    lines = [
        "#### The held-out synthetic set",
        "",
        f"The held-out set is simulated on grid `{config['grid']}` with"
        f" {_seeds(seeds)}, {n} simulations each.",
        *cover,
        f"We evaluate draw {config['draw']} of each simulation, so the"
        " outbreaks are independent.",
        "The fine-tune never trains on these seeds.",
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
    parser.add_argument("--si-dir", type=Path, required=True)
    args = parser.parse_args()
    text = args.simulator.read_text()
    args.si_dir.mkdir(parents=True, exist_ok=True)
    for layer, table in layer_tables(text).items():
        (args.si_dir / f"grid-{layer}.md").write_text(
            to_markdown(table, HEADERS)
        )
    (args.si_dir / "epinowcast-model.md").write_text(model_note(text))
    config = json.loads(args.config)
    (args.si_dir / "heldout-set.md").write_text(heldout_note(config))


if __name__ == "__main__":
    main()
