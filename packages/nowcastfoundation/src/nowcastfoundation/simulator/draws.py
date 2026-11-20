# Copyright (c) 2026 Sam Abbott. MIT licence, see LICENSE.
"""Draw a simulation's design and values from the grid config.

The Python form of the R engine's `draw_design` and `draw_values`
(`packages/nowcastfoundation/r/R/grid.R`, `design.R`): the same settings,
names, sampling distributions and shares, read from the same grid file.
A design holds the categorical settings and the structure they imply; the
values hold every continuous setting, by layer.
Vectors are lists, and place indices count from 1 as in R, so a design and
its values pass to the R reporting step and the settings files as they are.

Every draw takes a numpy `Generator`; `generator(seed, *keys)` gives one
keyed by the batch's seed and the simulation's place in it, so one
simulation's draws do not depend on the others'.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from scipy import stats

__all__ = [
    "RISING_PATH",
    "WEEK",
    "delay_q99",
    "draw_cell",
    "draw_count",
    "draw_design",
    "draw_setting",
    "draw_share",
    "draw_strata",
    "draw_values",
    "generator",
    "log_r_walk",
    "max_delay_for",
    "max_strata",
    "read_grid",
    "stratum_values",
    "system_locations",
]

Grid = Mapping[str, Any]

WEEK = (
    "Monday",
    "Tuesday",
    "Wednesday",
    "Thursday",
    "Friday",
    "Saturday",
    "Sunday",
)
"""The weekdays, in the order of `report_weekdays`."""

RISING_PATH = {
    "growth": "growth",
    "decline": "growth",
    "flat": "rise_fall_gradual",
    "rise_fall_sharp": "rise_fall_sharp",
    "rise_fall_gradual": "rise_fall_gradual",
    "fall_rise": "rise_fall_sharp",
    "oscillating": "multi_wave",
    "multi_wave": "multi_wave",
}
"""The rising shape an early outbreak not brought under control follows:
its drawn trajectory if it rises, else that trajectory's rising
counterpart."""


def read_grid(path: str | Path) -> dict[str, Any]:
    """Read a grid config file (YAML), as `config/finetune/grid.yaml`."""
    grid: dict[str, Any] = yaml.safe_load(Path(path).read_text())
    return grid


def generator(seed: int, *keys: int) -> np.random.Generator:
    """Return a numpy Generator keyed by a seed and integer keys."""
    return np.random.default_rng(np.random.SeedSequence([seed, *keys]))


def draw_cell(grid: Grid, rng: np.random.Generator) -> dict[str, str]:
    """Draw each factor's level from its categorical distribution."""
    out = {}
    for name, levels in grid["factors"].items():
        w = np.asarray(list(levels.values()), dtype=float)
        out[name] = list(levels)[rng.choice(len(w), p=w / w.sum())]
    return out


def draw_setting(
    grid: Grid, name: str, rng: np.random.Generator, n: int | None = None
) -> Any:
    """Draw a setting from its named sampling distribution.

    `lognormal` takes its median and sdlog, `normal` its mean and sd,
    `halfnormal` its scale, `gamma` its mean and sd, `beta` its shapes on
    its support `lower` to `upper`, and `uniform` its range.
    None is truncated.
    One draw (a float) if `n` is `None`, else an array of `n`.
    """
    d = grid["distributions"].get(name)
    if d is None:
        raise KeyError(f"no sampling distribution for {name}")
    size = 1 if n is None else n
    match d["dist"]:
        case "lognormal":
            x = rng.lognormal(np.log(d["median"]), d["sdlog"], size)
        case "normal":
            x = rng.normal(d["mean"], d["sd"], size)
        case "halfnormal":
            x = np.abs(rng.normal(0.0, d["scale"], size))
        case "gamma":
            shape = (d["mean"] / d["sd"]) ** 2
            x = rng.gamma(shape, d["sd"] ** 2 / d["mean"], size)
        case "beta":
            x = d["lower"] + (d["upper"] - d["lower"]) * rng.beta(
                d["a"], d["b"], size
            )
        case "uniform":
            x = rng.uniform(d["lower"], d["upper"], size)
        case other:
            raise ValueError(f"unknown distribution {other}")
    return float(x[0]) if n is None else x


def draw_count(
    grid: Grid, name: str, rng: np.random.Generator, n: int | None = None
) -> Any:
    """Draw a count: its base plus a Poisson with the grid's mean."""
    k = grid["counts"].get(name)
    if k is None:
        raise KeyError(f"no count for {name}")
    x = int(k["base"]) + rng.poisson(k["mean"], 1 if n is None else n)
    return int(x[0]) if n is None else x.astype(int)


def draw_share(
    grid: Grid, name: str, rng: np.random.Generator, n: int | None = None
) -> Any:
    """Draw a Bernoulli choice with the grid's share."""
    p = grid["shares"].get(name)
    if p is None:
        raise KeyError(f"no share for {name}")
    x = rng.random(1 if n is None else n) < p
    return bool(x[0]) if n is None else x


def max_strata(grid: Grid, weeks: int) -> int:
    """Return the most strata in a run of `weeks`.

    `max_strata` up to 28 weeks, then fewer, so the reporting triangle and
    the batch's memory stay about as at 28 weeks.
    """
    cap = int(grid["max_strata"])
    return min(cap, int(cap * 28 / weeks))


def draw_strata(
    grid: Grid, cell: Mapping[str, str], rng: np.random.Generator
) -> tuple[int, int]:
    """Draw a system's places and ages from the grid's counts."""
    match cell["strata"]:
        case "single":
            return 1, 1
        case "places":
            even = cell["spread"] != "concentrated"
            name = "places_even" if even else "places_concentrated"
            return draw_count(grid, name, rng), 1
        case "ages":
            return 1, draw_count(grid, "ages", rng)
        case "places_ages":
            places = draw_count(grid, "places_by_ages", rng)
            return places, draw_count(grid, "ages_by_places", rng)
        case other:
            raise ValueError(f"unknown strata {other}")


def system_locations(places: int, ages: int, regions: int = 6) -> int:
    """Return a system's locations in a whole-system training block.

    The country, then for places by ages each place's and each age's
    total, for flat places up to `regions` region totals, and every leaf.
    """
    leaves = places * ages
    if leaves == 1:
        return 1
    if places > 1 and ages > 1:
        totals = places + ages
    elif places > 1:
        totals = regions
    else:
        totals = 0
    return 1 + totals + leaves


def draw_design(
    grid: Grid, rng: np.random.Generator, weeks: int | None = None
) -> dict[str, Any]:
    """Draw a simulation's design: its categorical settings and structure.

    Parameters
    ----------
    grid : Mapping
        A grid (`read_grid`).
    rng : numpy.random.Generator
        The simulation's generator.
    weeks : int, optional
        Fix the series length (pilots); drawn if `None`.

    Returns
    -------
    dict
        The factors' levels, `weeks`, `n_places`, `n_ages` and
        `report_weekdays` (`"all"` or weekdays joined by ";").

    Raises
    ------
    RuntimeError
        If no strata draw in 1,000 fits the budget.
    """
    cell = draw_cell(grid, rng)
    if weeks is None:
        weeks = max(
            int(grid["floors"]["weeks"]),
            int(np.round(draw_setting(grid, "weeks", rng))),
        )
    # The strata, drawn again until the whole system fits the simulator's
    # memory and the trainer's block.
    for _ in range(1000):
        n_places, n_ages = draw_strata(grid, cell, rng)
        fits = (
            n_places * n_ages <= max_strata(grid, weeks)
            and system_locations(n_places, n_ages, grid["regions"])
            <= grid["max_locations"]
        )
        if fits:
            break
    else:
        raise RuntimeError("no strata draw fits the budget")
    if cell["strata_link"] == "independent":
        # Each independent stratum is a simulation of its own, so at most
        # `max_independent` strata; concentrated outbreaks keep their many
        # places, of which only the active ones are simulated.
        m = grid["max_independent"]
        if cell["strata"] == "places" and cell["spread"] != "concentrated":
            n_places = min(n_places, m)
        if cell["strata"] == "ages":
            n_ages = min(n_ages, m)
        if cell["strata"] == "places_ages":
            n_places = 2
            n_ages = min(n_ages, m // 2)
    match cell["report_days"]:
        case "daily":
            report = "all"
        case "weekdays":
            report = ";".join(WEEK[:5])
        case "fixed_days":
            k = int(rng.choice([2, 3]))
            report = ";".join(
                WEEK[i] for i in sorted(rng.choice(7, k, replace=False))
            )
        case "weekly":
            report = WEEK[int(rng.integers(7))]
        case other:
            raise ValueError(f"unknown report_days {other}")
    return {
        **cell,
        "weeks": weeks,
        "n_places": int(n_places),
        "n_ages": int(n_ages),
        "report_weekdays": report,
    }


def log_r_walk(grid: Grid, n: int, rng: np.random.Generator) -> list[float]:
    """Return `n` weeks of an AR(1) walk on log R around 0 (R = 1)."""
    e = draw_setting(grid, "walk_r_step", rng, n)
    x = e.copy()
    ar = grid["constants"]["walk_ar"]
    for w in range(1, n):
        x[w] = ar * x[w - 1] + e[w]
    return [float(e) for e in x]


def stratum_values(
    grid: Grid,
    structure: str,
    n: int,
    name: str,
    rng: np.random.Generator,
    probability: bool = False,
) -> list[float]:
    """Return per-stratum values of setting `name` for `n` strata.

    As `structure` says: shared (one value), pooled (the shared value
    times LogNormal(0, tau), tau half-normal; on the logit scale for a
    `probability`) or independent (each its own draw).
    """
    match structure:
        case "shared":
            return [draw_setting(grid, name, rng)] * n
        case "pooled":
            shared = draw_setting(grid, name, rng)
            tau = draw_setting(grid, "pool_tau", rng)
            dev = tau * draw_setting(grid, "pool_z", rng, n)
            if probability:
                x = stats.logistic.cdf(stats.logistic.ppf(shared) + dev)
            else:
                x = shared * np.exp(dev)
            return [float(e) for e in np.atleast_1d(x)]
        case "independent":
            return [float(e) for e in draw_setting(grid, name, rng, n)]
        case other:
            raise ValueError(f"unknown strata structure {other}")


def delay_q99(family: str, v: Mapping[str, Any]) -> float:
    """Return the delay's 99th centile (the R engine's `delay_parameters`).

    With no parametric delay (`"none"`), the delay is geometric with the
    drawn mean, counted in failures as R's `qgeom`.
    """
    match family:
        case "lognormal":
            sdlog, median = v["delay_sdlog"], v["delay_median"]
            return float(stats.lognorm.ppf(0.99, sdlog, scale=median))
        case "gamma":
            shape = 1 / v["delay_cv"] ** 2
            return float(
                stats.gamma.ppf(0.99, shape, scale=v["delay_mean"] / shape)
            )
        case "exponential":
            return float(stats.expon.ppf(0.99, scale=v["delay_mean"]))
        case "none":
            hazard = 1 / (1 + v["delay_mean"])
            return float(stats.geom.ppf(0.99, hazard) - 1)
        case other:
            raise ValueError(f"unknown delay family {other}")


def max_delay_for(q99: float) -> int:
    """Return the longest delay in days.

    The shortest of 3, 4, 6, 8 and 12 weeks at or above the delay's 99th
    centile, at most 12 weeks.
    """
    for days in (21, 28, 42, 56):
        if days >= q99:
            return days
    return 84


def _sample(rng: np.random.Generator, x: Sequence[int], k: int) -> list[int]:
    """Draw `k` of `x` without replacement, in the order drawn."""
    return [int(x[i]) for i in rng.choice(len(x), k, replace=False)]


def draw_values(
    grid: Grid, design: Mapping[str, Any], rng: np.random.Generator
) -> dict[str, Any]:
    """Draw the continuous settings of a simulation, by layer.

    Every continuous setting from its sampling distribution in the grid,
    counts as a base plus a Poisson, and choices as Bernoulli draws with
    the grid's shares.

    Parameters
    ----------
    grid : Mapping
        A grid (`read_grid`).
    design : Mapping
        A design (`draw_design`); an `offset` in it fixes the calendar
        weekday the series starts on (the strata of one outbreak share it).
    rng : numpy.random.Generator
        The simulation's generator.

    Returns
    -------
    dict
        The values, under the R engine's names: scalars as numbers or
        text, vectors as lists, place indices from 1.
    """

    def ds(name: str, n: int | None = None) -> Any:
        return draw_setting(grid, name, rng, n)

    def dc(name: str, n: int | None = None) -> Any:
        return draw_count(grid, name, rng, n)

    def sh(name: str, n: int | None = None) -> Any:
        return draw_share(grid, name, rng, n)

    def vec(x: Any) -> list[Any]:
        out: list[Any] = np.asarray(x).tolist()
        return out

    floors, rates = grid["floors"], grid["rates"]
    n_p = int(design["n_places"])
    n_s = n_p * int(design["n_ages"])
    early = design["window"] == "early"
    endemic = design["window"] == "endemic"
    weeks = int(design["weeks"])
    offset = design.get("offset")
    v: dict[str, Any] = {
        "weeks": weeks,
        "offset": int(rng.integers(7)) if offset is None else int(offset),
    }

    # Layer 1: infections.
    active = min(n_p, dc("active_places"))
    concentrated = design["spread"] == "concentrated"
    v["quiet"] = 1 - active / n_p if concentrated else ds("quiet")
    v["coupling"] = ds("coupling")
    v["place_rho"] = ds("place_rho")
    v["place_sd"] = ds("place_sd")
    v["place_lags"] = [0, *vec(dc("place_lag_days", n_p - 1))]
    v["late_at"] = ds("late_at")
    v["late_days"] = dc("late_days")
    v["mixing_eps"] = ds("mixing_eps")
    v["mixing_order"] = vec(rng.permutation(n_p) + 1)
    v["mixing_x"] = vec(ds("mixing_xy", n_p))
    v["mixing_y"] = vec(ds("mixing_xy", n_p))
    v["mixing_range"] = ds("mixing_range")
    v["mixing_switch"] = "on" if sh("mixing_switch_on") else "off"
    v["mixing_switch_at"] = ds("mixing_switch_at")
    v["mixing_switch_days"] = dc("mixing_switch_days")
    v["import_rate"] = ds("import_rate")
    v["import_at"] = sorted(vec(ds("import_at", dc("import_pulses"))))
    v["import_size"] = ds("import_size")
    v["stratum_sd"] = ds(
        "stratum_sd_concentrated" if concentrated else "stratum_sd_even"
    )
    if early:
        controlled = sh("controlled")
        v["path"] = (
            "controlled" if controlled else RISING_PATH[design["trajectory"]]
        )
    elif endemic:
        v["path"] = "endemic"
    else:
        v["path"] = design["trajectory"]
    v["gt_mean"] = ds("gt_mean")
    v["gt_cv"] = ds("gt_cv")
    v["amplitude"] = ds("amplitude")
    v["flat"] = ds("flat")
    v["peak"] = ds("peak")
    v["period"] = float(max(floors["period_days"], ds("period")))
    v["phase"] = ds("phase")
    v["waves"] = 2 + int(rng.poisson(rates["extra_waves"]))
    v["wave_scale"] = vec(ds("wave_scale", v["waves"] + 1))
    v["flip_days"] = ds("flip_days")
    # Weeks of the series, the longest delay (12 weeks) and the horizon.
    v["walk_r"] = log_r_walk(grid, weeks + 18, rng)
    v["control_days"] = ds("control_days")
    v["r_drift"] = ds("r_drift") if not endemic and sh("drift") else 1.0
    v["population"] = vec(
        ds("endemic_population" if endemic else "population", n_p)
    )
    v["seed"] = float(max(floors["seed"], ds("seed")))
    v["level"] = ds("endemic_level" if endemic else "level")
    v["trickle"] = ds("trickle")
    n_shifts = int(rng.poisson(rates["shifts"]))
    v["shift_at"] = vec(ds("shift_at", n_shifts))
    v["shift_size"] = vec(ds("shift_size", n_shifts))
    v["shift_days"] = vec(ds("shift_days", n_shifts))
    v["shift_reactive"] = vec(sh("reactive", n_shifts))
    v["shift_up"] = vec(sh("shift_up", n_shifts))
    per_year = rates["variants_per_year"]
    n_variants = 0 if early else int(rng.poisson(per_year * weeks / 52))
    v["variant_at"] = vec(ds("variant_at", n_variants))
    v["variant_escape"] = vec(ds("escape", n_variants))
    v["variant_r"] = vec(ds("variant_r", n_variants))
    v["variant_days"] = vec(ds("shift_days", n_variants))
    # Places with no cases; the first has cases if every place would not.
    quiet = [i + 1 for i in np.flatnonzero(rng.random(n_p) < v["quiet"])]
    if len(quiet) == n_p:
        quiet = [q for q in quiet if q != 1]
    v["quiet_places"] = quiet
    # Places seeded mid-outbreak: one to three places with cases after the
    # first, which starts at the outset.
    cases = [p for p in range(1, n_p + 1) if p not in quiet]
    later = cases[1:]
    v["late_places"] = (
        _sample(rng, later, min(len(later), dc("late_places"))) if later else []
    )

    # Layer 2: observation.
    obs = design["observation_strata"]
    v["ascertainment"] = stratum_values(
        grid, obs, n_s, "ascertainment", rng, probability=True
    )
    v["incubation_mean"] = stratum_values(
        grid, obs, n_s, "incubation_mean", rng
    )
    v["sqrt_phi"] = ds("sqrt_phi")
    v["obs_phi_sd"] = ds("obs_phi_sd")
    if design["obs_strata"] == "shared":
        v["obs_sqrt_phi"] = [v["sqrt_phi"]] * n_s
    else:
        z = ds("pool_z", n_s)
        v["obs_sqrt_phi"] = vec(v["sqrt_phi"] * np.exp(v["obs_phi_sd"] * z))
    v["obs_family"] = [design["observation"]] * n_s

    # Layer 3: reporting delays.
    v["delay_median"] = ds("delay_median")
    v["delay_sdlog"] = ds("delay_sdlog")
    v["delay_mean"] = ds("delay_mean")
    v["delay_cv"] = 1 / float(np.sqrt(ds("delay_shape")))
    v["delay_place_sd"] = ds("delay_place_sd")
    v["delay_age_sd"] = ds("delay_age_sd")
    v["hazard_walk_sd"] = ds("hazard_walk_sd")
    v["hazard_sd"] = ds("hazard_sd")
    v["drift"] = (1 if rng.random() < 0.5 else -1) * ds("drift")
    v["step"] = (1 if rng.random() < 0.5 else -1) * ds("step")
    v["step_at"] = ds("step_at")
    v["periodic"] = ds("periodic")
    v["periodic_weeks"] = ds("periodic_weeks")
    v["walk_sd"] = ds("walk_sd")
    steps = np.concatenate([[0.0], ds("walk_step", weeks + 17)])
    v["delay_walk"] = vec(v["walk_sd"] * np.cumsum(steps))
    v["smooth_amp"] = ds("smooth_amp")
    v["smooth_period"] = float(max(floors["period_days"], ds("smooth_period")))
    v["smooth_phase"] = ds("phase")
    v["slow_at"] = ds("slow_at")
    v["slow_weeks"] = dc("slow_weeks")
    v["slow_mult"] = ds("slow_mult")
    parametric = design["delay_form"] != "non_parametric"
    family = design["delay_family"] if parametric else "none"
    v["delay_dow_sd"] = ds("delay_dow_sd")
    v["max_delay"] = max_delay_for(delay_q99(family, v))

    # Layer 4: the reporting system.
    v["reference_dow_sd"] = ds("reference_dow_sd")
    v["dow_sd"] = ds("dow_sd")
    if design["shocks"] == "none":
        places: list[int] = []
    else:
        places = _sample(rng, cases, min(len(cases), dc("shock_places")))
    per_place = dc("shocks_per_place", len(places))
    v["shock_places"] = [
        p for p, k in zip(places, per_place, strict=True) for _ in range(k)
    ]
    n_shocks = len(v["shock_places"])
    v["shock_at"] = vec(ds("shock_at", n_shocks))
    v["shock_weeks"] = vec(dc("shock_weeks", n_shocks))
    v["shock_mult"] = vec(ds("shock_mult", n_shocks))
    rep = design["reporting_strata"]
    v["revise_over"] = stratum_values(
        grid, rep, n_s, "revise_over", rng, probability=True
    )
    v["revise_move"] = stratum_values(
        grid, rep, n_s, "revise_move", rng, probability=True
    )
    v["revise_keep"] = stratum_values(grid, rep, n_s, "revise_keep", rng)
    return v
