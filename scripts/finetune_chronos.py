# Copyright (c) 2026 Sam Abbott. MIT licence, see LICENSE.
"""Blocks, batches and memory budget for fine-tuning Chronos-2.

Cuts outbreaks from the simulator (packages/nowcastfoundation/r) into
training blocks
(`outbreak_blocks`), packs them into batches within a memory budget
(`plan_batches`, `max_series`), without torch.
`scripts/finetune_stream.py` trains on them and
publishes each checkpoint as the release `chronos-ft-<hash>`, which the
`_ft` models load the newest one
(`forecasters.foundation.resolve_checkpoint`).
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd

from nowcastfoundation import forecaster as foundation
from nowcastfoundation.finetune import (
    AHEAD_WEEKS,
    BINNED,
    MIXED,
    ROLLING,
    release_days,
    training_block,
    weekly_cutoffs,
)
from nowcastutils.base import NATIONAL
from nowcastutils.dates import epiweek_end

CONTEXT = 364
"""Steps of context in a block, left-padded with missing values."""


MAX_LAGS = {
    "weekly": tuple(range(2, 8)),
    "daily": tuple(range(14, 50)),
    "3day": tuple(range(5, 17)),
    "daily+weekly": tuple(range(14, 50)),
    "daily+rolling7": tuple(range(14, 50)),
    "daily+3day": tuple(range(14, 50)),
    "daily+rolling3": tuple(range(14, 50)),
}
"""Release lags $D$ the blocks are drawn from; daily blocks with period
totals have daily lags."""

LAG_CAP = (
    "a block's drawn lag capped at the steps of history before its cut-off, "
    "less two of context, so early cut-offs are kept"
)
"""How a drawn lag meets a short history (`outbreak_blocks`)."""


DAYS_PER_STEP = {
    "weekly": 7,
    "daily": 1,
    "3day": 3,
    "daily+weekly": 1,
    "daily+rolling7": 1,
    "daily+3day": 1,
    "daily+rolling3": 1,
}
"""Days in a step of each source."""


NATIVE_CADENCE = {1: "daily", 3: "3day", 7: "weekly"}
"""The regular release cadence of each step width, in days."""


BLOCK_DESIGN = {
    "source_shares": {
        "daily": 0.25,
        "weekly": 0.25,
        "3day": 0.15,
        "daily+weekly": 0.125,
        "daily+rolling7": 0.125,
        "daily+3day": 0.05,
        "daily+rolling3": 0.05,
    },
    "cadences": {
        "daily": {
            "daily": 0.6,
            "weekly": 0.2,
            "daily_gaps": 0.1,
            "weekdays": 0.1,
        },
        "3day": {"3day": 0.8, "3day_gaps": 0.2},
        "weekly": {"weekly": 0.8, "weekly_gaps": 0.2},
    },
    "whole_system": True,
}
"""The block design: each layout's share of blocks (`source_shares`), the
release cadences of each step width with their shares (`cadences`,
`finetune.release_days`), and whole-system blocks (`whole_system`).

Releases follow the layout's steps, daily, every 3 days or weekly.
Daily layouts also get weekly releases (the BDBV hub's daily data), daily
releases on weekdays only, and missed releases (`_gaps`); 3-day and weekly
layouts get missed releases. Daily blocks with weekly or 3-day totals have
them either on period-end days only (`finetune.MIXED`) or as rolling sums
on every day (`finetune.ROLLING`). Overall, 36% of blocks have daily
releases, 32% weekly (12% daily blocks, 20% weekly blocks), 12% 3-day,
6% daily with gaps, 6% weekdays only, 5% weekly with gaps and 3% 3-day
with gaps. Each block is a whole simulated system (`whole_system`), or
its country alone.
"""

GRID_FILE = Path(__file__).parents[1] / "config" / "finetune" / "grid.yaml"
"""The grid file the simulations are drawn from (`use_grid`)."""

SOURCE_SHARES = dict(BLOCK_DESIGN["source_shares"])
"""Share of blocks of each layout (`BLOCK_DESIGN`)."""

CADENCES = {k: dict(v) for k, v in BLOCK_DESIGN["cadences"].items()}
"""Release cadences of each step width and their shares (`BLOCK_DESIGN`)."""

WHOLE_SYSTEM = bool(BLOCK_DESIGN["whole_system"])
"""Whether each block is a whole simulated system (`whole_system`), rather
than a drawn share of its places (`choose_places`)."""

SOURCES = tuple(SOURCE_SHARES)
"""Sources blocks are cut from, drawn at `SOURCE_SHARES`."""


def use_grid(grid_file: Path | str = GRID_FILE) -> None:
    """Set the parents from the grid file (`grid_parents`).

    Raises
    ------
    FileNotFoundError
        If the grid file is missing, so a wrong path is not silently
        trained with the even parent mix.
    """
    global PARENTS, SIMULATOR, CODE
    path = Path(grid_file)
    if not path.is_file():
        raise FileNotFoundError(f"no grid file at {path}")
    PARENTS = grid_parents(path)
    # Named relative to the repository, so the hash does not depend on
    # where the checkout is.
    name = os.path.relpath(path.resolve(), ROOT.resolve())
    SIMULATOR = simulator_files(Path(name).as_posix())
    CODE = code_files(SIMULATOR)


def native_cadence(source: str) -> str:
    """Return the regular cadence of a source's steps (`NATIVE_CADENCE`)."""
    return NATIVE_CADENCE[DAYS_PER_STEP[source]]


def draw_cadence(source: str, rng: np.random.Generator) -> str:
    """Draw a block's release cadence at its source's shares (`CADENCES`)."""
    mix = CADENCES[native_cadence(source)]
    return str(rng.choice(list(mix), p=list(mix.values())))


def check_sources(sources: tuple[str, ...]) -> None:
    """Raise unless every source has its own triangle or is binned.

    A source neither in `foundation.SOURCES` nor in `BINNED` would be cut
    as daily without notice.
    """
    bad = [
        s
        for s in sources
        if (
            s not in foundation.SOURCES
            and s not in BINNED
            and s not in MIXED
            and s not in ROLLING
        )
        or s not in DAYS_PER_STEP
        or s not in MAX_LAGS
    ]
    if bad:
        raise ValueError(f"sources {bad} are neither native nor binned")


def check_package(root: Path) -> None:
    """Raise unless `bdbvhubrenewal` is imported from `root`'s src.

    A venv linked from another worktree imports that worktree's package,
    whose blocks may differ from this one's.
    """
    import bdbvhubrenewal

    where = Path(bdbvhubrenewal.__file__).resolve()
    if not where.is_relative_to((Path(root) / "src").resolve()):
        raise RuntimeError(
            f"bdbvhubrenewal is imported from {where}, not this worktree "
            f"({Path(root).resolve()})"
        )


MEMORY_GB = 36.0
"""The GPU memory budget of a training step, in GB: the trainer's cap on
MPS memory (0.85 of a 55.7 GB working set) less room for the model and
the held-out set."""

MAX_SERIES = 2000
"""Most series in a block. A block's series count is drawn log-uniformly
up to it. With gradient checkpointing a block of 2,400 series fits the
laptop's GPU memory (64 GB) but peaks at about 56 GB in training, so
2,000 leaves headroom."""

CUTS = 32
"""Snapshots (cut-off slots) of each observation draw, spread over its
course: 32 blocks from one draw."""

MEMORY_COST = {False: (0.0, 6.7e-5), True: (6.25e-3, 3.35e-6)}
"""GPU memory of one training step in GB per series and per series
squared, without and with gradient checkpointing: fitted to 800 series
taking about 43 GB without it, and 1,000, 1,500 and 2,000 series taking
9.6, 17.3 and 25.9 GB with it (Chronos-2 with LoRA and AdamW on an Apple
M4 Max). Group attention makes it grow with the square of the series."""


def memory_gb(series: int, checkpointed: bool) -> float:
    """Return a training step's GPU memory in GB for `series` series."""
    per, squared = MEMORY_COST[checkpointed]
    return per * series + squared * series**2


def _most_within(budget_gb: float, checkpointed: bool) -> int:
    """Return the most series whose step fits `budget_gb`."""
    per, squared = MEMORY_COST[checkpointed]
    root = (-per + (per**2 + 4 * squared * budget_gb) ** 0.5) / (2 * squared)
    return int(root)


def checkpoint_above(budget_gb: float) -> int:
    """Return the series above which a step needs gradient checkpointing."""
    return _most_within(budget_gb, checkpointed=False)


def max_series(budget_gb: float) -> int:
    """Return the most series of a block that fits with checkpointing."""
    return _most_within(budget_gb, checkpointed=True)


ENGINE = "packages/nowcastfoundation/r"
"""The simulator engine, whose R files and description are hashed."""

WRAPPER = "workflow/scripts/simulate.R"
"""The simulate rule's wrapper around the engine."""

ROOT = Path(__file__).parents[1]
"""The repository root, which the hashed paths are relative to."""


def simulator_files(grid_file: str) -> tuple[str, ...]:
    """Return the simulator's files: the grid, the wrapper and the engine.

    The engine is its description and every R file (`ENGINE/R/*.R`,
    found by glob so a new file is hashed without listing it).
    """
    engine = sorted(p.name for p in (ROOT / ENGINE / "R").glob("*.R"))
    return (
        grid_file,
        WRAPPER,
        f"{ENGINE}/DESCRIPTION",
        *(f"{ENGINE}/R/{name}" for name in engine),
    )


def workflow_grid_file() -> str:
    """Return the workflow's grid file (`simulate.grid_file`)."""
    import yaml

    config = yaml.safe_load((ROOT / "workflow" / "config.yaml").read_text())
    return str(config["simulate"]["grid_file"])


def code_files(simulator: tuple[str, ...]) -> dict[str, str]:
    """Return `CODE` for a simulator's files, each named by its path."""
    files = (
        *simulator,
        "scripts/finetune_chronos.py",
        "packages/nowcastfoundation/src/nowcastfoundation/finetune.py",
    )
    return {path: path for path in files}


SIMULATOR = simulator_files(workflow_grid_file())
"""The simulator's files, with the workflow's grid until a run sets its
own (`use_grid`)."""

CODE = code_files(SIMULATOR)
"""Files the checkpoint depends on, hashed into its settings: each
file's name in `code_sha256`, then its path. The paths also go into the
settings outside the hash (`code_paths`), for `finetune_verify`."""

TRAINER = "scripts/finetune_stream.py"
"""The trainer's name in `code_sha256` (its path is its own file)."""

FOUNDATION_PATH = (
    "packages/nowcastfoundation/src/nowcastfoundation/forecaster.py"
)
"""Where the module of `FOUNDATION_CODE` is now (`code_paths`
`foundation`)."""

FOUNDATION_CODE = (foundation.hub_triangle, foundation.layout_block)
"""The functions of `foundation` the blocks are laid out with, hashed into
the settings."""

PREFIXES = (1000, 2000, 4000, 8000, 16000, 32000, 64000)
"""Outbreaks seen at each saved checkpoint. The last is every training
outbreak."""

EVAL_BLOCKS = 100
"""Held-out blocks whose loss is logged during training."""

EVAL_EVERY = 100
"""Steps between logged held-out losses."""

OUTPUT_PATCH = 16
"""Chronos-2's output patch length."""

LOSS_HORIZON_DAYS: int | None = None
"""Days after the cut-off that a training block's horizon, and so its
loss, runs to; None for the whole horizon, to `AHEAD_WEEKS` past the
cut-off's week."""

SIM_DELAY_WEEKS = 8
"""The simulations' longest delay in weeks, after which a count is
complete, unless an outbreak's settings give `max_delay` in days."""


def _sha256(path: Path) -> str:
    """Return a file's SHA-256."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


QUIET_WINDOW_DAYS = 28
"""Days up to a cut-off over which a block's reported cases are counted
for the quiet rule (`quiet`)."""

QUIET_MIN_CASES: int | None = None
"""Cases a block needs reported across its places in the
`QUIET_WINDOW_DAYS` up to its cut-off to be trained on (`quiet`); None
keeps every block."""


def reported_cases(block: dict, days_per_step: int, days: int) -> float:
    """Return a block's cases reported in `days` up to its cut-off.

    Only what is known at the cut-off counts, never the counts after it.
    Each step counts once, at its latest report: before `start`, the
    targets' counts; from `start` to the cut-off, each lag series'
    value at the step's lag from the cut-off (lag series hold a step's
    count as reported at that lag, so summing them would count a case
    once per lag). The places are summed within each level (national,
    intermediate, leaf) and the busiest level counts, so a total does
    not count its places' cases again, and a block of some places with
    their country counts the country's cases. `block` is
    `finetune.training_block`'s, before padding.
    """
    x, places = block["context"], block["places"]
    t, n = block["n_targets"], len(places)
    lags, extra = divmod(block["n_covariates"], n)
    if extra:
        raise ValueError(
            f"{block['n_covariates']} lag series over {n} places: not the "
            "same number a place"
        )
    if block["nowcast"] > lags:
        raise ValueError(
            f"nowcast of {block['nowcast']} steps past {lags} lags: it would "
            "read the next place's rows"
        )
    start = block["start"]
    cut = start + block["nowcast"]
    lo = max(cut - -(-days // days_per_step), 0)
    count = np.nansum(x[:n, lo:start], axis=1)
    rows = np.arange(n)
    for step in range(max(lo, start), cut):
        count += np.nan_to_num(x[t + rows * lags + (cut - 1 - step), step])
    levels = np.array(target_levels(places))
    return float(max(count[levels == lv].sum() for lv in set(levels)))


def quiet(block: dict, days_per_step: int, min_cases: int) -> bool:
    """Return whether a block has fewer than `min_cases` reported cases.

    The cases are those reported across its places in the
    `QUIET_WINDOW_DAYS` up to its cut-off (`reported_cases`), so the
    rule depends only on what is known at the cut-off.
    """
    return reported_cases(block, days_per_step, QUIET_WINDOW_DAYS) < min_cases


def _pad(
    block: dict,
    context: int,
    loss_days: int | None = None,
    days_per_step: int = 1,
) -> dict:
    """Pad a block to `context` steps and its horizon in whole patches.

    Chronos-2 predicts in patches of `OUTPUT_PATCH` steps, so a block's
    prediction length is its horizon, its nowcast and forecast steps,
    rounded up to whole patches; the steps past the horizon are missing.
    With `loss_days`, the horizon stops at the nowcast steps and the
    steps covering `loss_days` days after them, of `days_per_step` days
    each. Long context is cropped. The arrays are numpy, so blocks are
    cut without torch; the trainer makes them tensors
    (`finetune_stream.tensors`).
    """
    start, h = block["start"], block["horizon"]
    if loss_days is not None:
        h = min(h, block["nowcast"] + -(-loss_days // days_per_step))
    length = -(-h // OUTPUT_PATCH) * OUTPUT_PATCH
    x = block["context"]
    keep = min(start, context)
    out = np.full((x.shape[0], context + length), np.nan, dtype=np.float32)
    out[:, context - keep : context + h] = x[:, start - keep : start + h]
    return {
        "context": out,
        "future_covariates": out[:, context:],
        "n_targets": block["n_targets"],
        "n_covariates": block["n_covariates"],
        "n_future_covariates": block["n_future_covariates"],
        "prediction_length": length,
        "nowcast": block["nowcast"],
    }


def strata(settings: pd.Series) -> list[str]:
    """Return an outbreak's strata, with or without cases."""
    places = [f"P{i:02d}" for i in range(1, int(settings["n_places"]) + 1)]
    n_ages = settings.get("n_ages", 1)
    n_ages = 1 if pd.isna(n_ages) else int(n_ages)
    ages = [f"A{i}" for i in range(1, n_ages + 1)]
    if len(places) > 1 and len(ages) > 1:
        return [f"{p}-{a}" for p in places for a in ages]
    return ages if len(ages) > 1 else places


LEVEL_MIX = ("all_levels", "leaves_national", "leaves", "national", "single")
"""A block's levels: the country, intermediate totals and leaves (as the
deployed block: national, provinces, zones), leaves and the country,
leaves only, the country only, or one leaf only. The last two are
univariate series from the same outbreaks. All levels need nested strata
(regions of places, or places by ages)."""

PARENT_MIX = ("exact", "unallocated", "late_parent")
"""How the country of a block with places relates to them, drawn evenly:
their sum, more than their sum (cases with no place), or reported through
a later stream (`finetune.country_counts`). A block of some of the
places is also not their sum. Parent means belongs to, not sums to."""

PARENTS: dict | None = None
"""The grid's parents, from its file's `aggregation` (`grid_parents`), or
None before `use_grid` or for a grid with no `aggregation`, whose parents
are `PARENT_MIX` drawn evenly with `finetune.PARENT_DRAWS`."""


def grid_parents(path: Path) -> dict | None:
    """Return the grid file's parents, or None if it sets none.

    The share of blocks with places whose country is their exact sum,
    has unallocated cases, or is reported later (`aggregation.parents`,
    weights), and the named distributions of the unallocated share and
    the lag (`finetune.country_counts`).
    """
    import yaml

    config = yaml.safe_load(Path(path).read_text())
    aggregation = config.get("aggregation")
    if aggregation is None:
        return None
    weights = aggregation["parents"]
    total = sum(weights.values())
    dists = config["distributions"]
    return {
        "shares": {k: w / total for k, w in weights.items()},
        "unallocated_share": dists[aggregation["unallocated_share"]],
        "parent_lag_days": dists[aggregation["parent_lag_days"]],
    }


UNIVARIATE_SHARE = 0.2
"""Share of blocks that are one series, the country or one leaf; the rest
are multivariate, shared evenly among their mixes."""

REGIONS = 1.0
"""Chance an outbreak of four or more flat places is grouped into 2 to 6
regions for its blocks, so blocks can nest places in regions in the
country."""


def regions(places: list[str], rng: np.random.Generator) -> dict[str, str]:
    """Return each place's id in a seeded grouping into 2 to 6 regions."""
    k = int(rng.integers(2, min(6, len(places) // 2) + 1))
    order = rng.permutation(len(places))
    group = {places[i]: 1 + j % k for j, i in enumerate(order)}
    return {p: f"R{group[p]}-{p}" for p in places}


def target_levels(places: tuple[str, ...]) -> tuple[str, ...]:
    """Return the level of each place of a block.

    The levels are national, intermediate (a total of strata) or leaf.
    """
    nested = any("-" in p for p in places)
    return tuple(
        "national"
        if p == NATIONAL
        else "leaf"
        if "-" in p or not nested
        else "intermediate"
        for p in places
    )


def whole_system(
    provinces: list[str], rng: np.random.Generator
) -> tuple[tuple[str, ...], str]:
    """Return a block of a whole simulated system and its level mix.

    Every stratum with every total it is part of: the country, each
    group's total (a place's over its ages, or a region's over its
    places), each age group's total across places, and the leaves; flat
    places are the country and its places. At `UNIVARIATE_SHARE`, and for
    a system of one stratum, the country alone. Strata are never cut out
    of the system they were simulated with.
    """
    if len(provinces) < 2 or rng.random() < UNIVARIATE_SHARE:
        return (NATIONAL,), "national"
    if not any("-" in p for p in provinces):
        return (NATIONAL, *sorted(provinces)), "leaves_national"
    groups = sorted({p.split("-")[0] for p in provinces})
    ages = sorted(
        {
            p.split("-")[1]
            for p in provinces
            if re.fullmatch(r"A\d+", p.split("-")[1])
        }
    )
    return (NATIONAL, *groups, *ages, *sorted(provinces)), "all_levels"


def choose_places(
    provinces: list[str], budget: int, rng: np.random.Generator
) -> tuple[tuple[str, ...], str]:
    """Return a block's places and its level mix (`LEVEL_MIX`).

    At most `budget` places are below the country. All levels are the
    country, the totals of the chosen leaves' groups (and, for places by
    ages, the age groups' totals) and the leaves.
    """
    nested = any("-" in p for p in provinces)
    if rng.random() < UNIVARIATE_SHARE:
        mix = str(rng.choice(LEVEL_MIX[3:]))
    else:
        mix = str(rng.choice(LEVEL_MIX[:3] if nested else LEVEL_MIX[1:3]))
    # A block's size gives way to its level mix: leaves need at least one
    # place below the country, all levels a group's total and a leaf.
    budget = max(budget, 2 if mix == "all_levels" else 1)
    if mix == "national":
        return (NATIONAL,), mix
    if mix == "single":
        return (str(rng.choice(provinces)),), mix
    if mix in ("leaves_national", "leaves"):
        k = min(budget, len(provinces))
        leaves = tuple(str(p) for p in sorted(rng.choice(provinces, k, False)))
        return ((NATIONAL,) if mix == "leaves_national" else ()) + leaves, mix
    groups: dict[str, list[str]] = {}
    for p in provinces:
        groups.setdefault(p.split("-")[0], []).append(p)
    ages = sorted(
        {
            p.split("-")[1]
            for p in provinces
            if re.fullmatch(r"A\d+", p.split("-")[1])
        }
    )
    chosen: list[str] = []
    leaves: list[str] = []
    for parent in rng.permutation(sorted(groups)):
        room = budget - len(chosen) - len(leaves) - 1
        if room < 1:
            break
        chosen.append(str(parent))
        leaves.extend(sorted(groups[parent])[:room])
        if room < len(groups[parent]):
            break
    if not chosen:
        # Too small a budget for a group: one group's total and a leaf.
        parent = sorted(groups)[0]
        chosen, leaves = [parent], [sorted(groups[parent])[0]]
    totals = sorted(chosen)
    if ages and rng.random() < 0.5:
        room = budget - len(chosen) - len(leaves)
        totals += ages[: max(room, 0)]
    return (NATIONAL, *totals, *sorted(leaves)), mix


AGGREGATE_END = "a random weekday, or 3-day offset, per block"
"""The day a block's period totals end on (`finetune.period_totals`):
any, as the German target is a 7-day sum ending on every day."""


def block_options(source: str, rng: np.random.Generator) -> dict:
    """Return a block's period end, drawn from `rng`.

    For a source with period totals, a random end day. Reports come in
    weekly batches only from the grid's weekday reporting.
    """
    end = int(rng.integers(7 if MIXED.get(source) == "weekly" else 3))
    return {"period_end": end if source in MIXED else None}


def outbreak_blocks(
    counts: pd.DataFrame,
    provinces: list[str],
    rng: np.random.Generator,
    delay_weeks: int = SIM_DELAY_WEEKS,
    cuts: int = CUTS,
    picks: set[tuple[str, int]] | None = None,
    empty: list[str] | None = None,
) -> list[dict]:
    """Return the training blocks of one outbreak.

    Each source is cut at `cuts` forecast origins spread over the
    outbreak's course, one in each of `cuts` equal stretches of its
    cut-offs, so they fall in its rising, peak and falling phases. With
    `picks`, only those (source, slot) blocks are cut. A block's series
    count is drawn log-uniformly from 1 to `MAX_SERIES`,
    and its places (`choose_places`) are about that many series at its
    lag, with the country first. A lag longer than the history before the
    cut-off is capped to it (`LAG_CAP`). A block whose places have no release
    rows by its cut (strata with no cases) is skipped and, with `empty`,
    described there. Each block's releases run up to its cut at a cadence
    drawn for its source (`draw_cadence`), recorded as its `cadence`.
    """
    # Cut-offs fall Wednesday to Saturday, as the hub's releases do.
    weekday = int(rng.integers(2, 6))
    first = str(counts["report_date"].min())
    end = pd.Timestamp(counts["report_date"].max()).date()
    last = end - dt.timedelta(weeks=delay_weeks + 5)
    cutoffs = weekly_cutoffs(first, last.isoformat(), weekday)
    # Flat places may nest in regions, each a total of its places.
    # Only places nest in regions; age groups total across places.
    flat = len(provinces) >= 4 and all(
        "-" not in p and p.startswith("P") for p in provinces
    )
    if flat and rng.random() < REGIONS:
        nest = regions(provinces, rng)
        counts = counts.assign(location=counts["location"].map(nest))
        provinces = sorted(nest.values())
    out = []
    for source in SOURCES:
        for j in range(cuts):
            if picks is not None and (source, j) not in picks:
                continue
            lag = int(rng.choice(MAX_LAGS[source]))
            width = DAYS_PER_STEP[source]
            # At least two steps of context before one lag.
            least = -(-3 * width // 7)
            if len(cutoffs) <= least:
                continue
            share = (j + rng.random()) / cuts
            cut = least + int(share * (len(cutoffs) - least))
            # As many lags as the history before the cut-off allows, with
            # two steps of context before them.
            history = (
                dt.date.fromisoformat(cutoffs[cut])
                - dt.date.fromisoformat(first[:10])
            ).days // width
            lag = min(lag, history - 2)
            if lag < 1:
                continue
            places: tuple[str, ...] = (NATIONAL,)
            mix = "national"
            # A place's rows: its counts, its lags and, with period totals,
            # one more.
            rows = lag + 1 + (source in MIXED or source in ROLLING)
            if WHOLE_SYSTEM:
                places, mix = whole_system(provinces, rng)
                if len(places) * rows > MAX_SERIES:
                    if empty is not None:
                        empty.append(
                            f"{source} cut {j}: {len(places)} places over "
                            f"the series cap ({MAX_SERIES})"
                        )
                    continue
            elif len(provinces) > 1:
                size = np.exp(rng.uniform(0, np.log(MAX_SERIES)))
                fits = MAX_SERIES // rows - 1
                budget = min(round(size / rows) - 1, fits)
                places, mix = choose_places(provinces, budget, rng)
            # How the country relates to its places, drawn from a child
            # generator, so the other draws are as without it.
            parent_rng, option_rng = rng.spawn(2)
            parent = "exact"
            parents = PARENTS
            if len(places) > 1 and places[0] == NATIONAL:
                if parents is None:
                    parent = str(parent_rng.choice(PARENT_MIX))
                else:
                    kinds = list(parents["shares"])
                    shares = list(parents["shares"].values())
                    parent = str(parent_rng.choice(kinds, p=shares))
            options = block_options(source, option_rng)
            # The releases follow the source's cadence (`CADENCES`), up to
            # the slot's cut.
            cadence = draw_cadence(source, option_rng)
            block = training_block(
                counts,
                source,
                places,
                release_days(first, cutoffs[cut], cadence, option_rng),
                max_lag=lag,
                totals=any("-" in p for p in provinces),
                parent=parent,
                rng=parent_rng,
                parent_draws=parents,
                **options,
            )
            if block is None:
                if empty is not None:
                    empty.append(f"{source} cut {j}: {', '.join(places)}")
                continue
            if block["start"] < 2:
                continue
            if QUIET_MIN_CASES is not None and quiet(
                block, DAYS_PER_STEP[source], QUIET_MIN_CASES
            ):
                continue
            padded = _pad(
                block, CONTEXT, LOSS_HORIZON_DAYS, DAYS_PER_STEP[source]
            )
            levels = target_levels(places)
            padded["levels"] = tuple(levels[i] for i in block["target_places"])
            padded["places"] = tuple(places)
            padded["target_places"] = list(block["target_places"])
            padded["parent"] = parent
            padded |= options
            padded["mix"] = mix
            padded["source"] = source
            padded["cadence"] = cadence
            out.append(padded)
    return out


def size_class(n: int) -> int:
    """Return a block's size class, by powers of two of its series."""
    return int(np.floor(np.log2(n)))


def plan_batches(
    sizes: np.ndarray,
    batch_size: int,
    rng: np.random.Generator,
    most: int | None = None,
) -> list[list[int]]:
    """Return batches of block indices that use every block once.

    Blocks are shuffled and packed in that order into batches of one size
    class until a batch holds `batch_size` series, so batches of small
    blocks pack tightly and a large block is a batch on its own. No batch
    of several blocks holds more than `most` series, the most the memory
    budget allows (`max_series`). The batches are then shuffled.
    """
    filling: dict[int, list[int]] = {}
    held: dict[int, int] = {}
    batches = []
    for i in rng.permutation(len(sizes)):
        c = size_class(int(sizes[i]))
        if (
            most is not None
            and filling.get(c)
            and (held[c] + int(sizes[i]) > most)
        ):
            batches.append(filling.pop(c))
            held[c] = 0
        filling.setdefault(c, []).append(int(i))
        held[c] = held.get(c, 0) + int(sizes[i])
        if held[c] >= batch_size:
            batches.append(filling.pop(c))
            held[c] = 0
    batches.extend(b for b in filling.values() if b)
    return [batches[j] for j in rng.permutation(len(batches))]


def plan_by_length(
    lengths: np.ndarray,
    sizes: np.ndarray,
    batch_size: int,
    rng: np.random.Generator,
    most: int | None = None,
) -> list[tuple[int, list[int]]]:
    """Return batches of blocks that share a prediction length.

    Blocks of each prediction length are batched by `plan_batches`, and
    the batches of every length are then shuffled together. Each batch is
    `(length, indices)`.
    """
    plan = []
    for length in sorted(set(lengths.tolist())):
        idx = np.flatnonzero(lengths == length)
        for batch in plan_batches(sizes[idx], batch_size, rng, most):
            plan.append((int(length), [int(idx[i]) for i in batch]))
    return [plan[j] for j in rng.permutation(len(plan))]


def _order(outbreak: str) -> tuple[int, int]:
    """Sort key of an outbreak id, `<seed>-<index>`."""
    seed, index = outbreak.split("-")
    return int(seed), int(index)


def _settings_hash(settings: dict) -> str:
    """Return the first 12 hex digits of the settings' SHA-256."""
    text = json.dumps(settings, sort_keys=True).encode()
    return hashlib.sha256(text).hexdigest()[:12]


def real_blocks(
    line_list: pd.DataFrame, cuts: list[str], last_week: str, lag: int
) -> list[dict]:
    """Return weekly validation blocks of a real outbreak's delay table.

    Each case of `line_list`, a public per-case delay table such as the
    hub's (`location`, `onset_week`, the week's
    Saturday, and `delay`, days from onset to report) is reported on its
    onset week's end plus its delay. Each block is the outbreak as known
    at a cut in `cuts`, weekly releases to it laid out as the ragged
    layout does, with the line list's counts ahead as the truth. A cut
    whose forecast weeks run past `last_week` is refused, so the blocks
    stay clear of weeks a later evaluation scores.

    Raises
    ------
    ValueError
        If a cut's forecast weeks run past `last_week`.
    """
    onset = pd.to_datetime(line_list["onset_week"])
    counts = (
        pd.DataFrame(
            {
                "location": line_list["location"],
                "reference_date": onset.dt.date.astype(str),
                "report_date": (
                    onset
                    + pd.to_timedelta(line_list["delay"].clip(lower=0), "D")
                ).dt.date.astype(str),
                "count": 1,
            }
        )
        .groupby(["location", "reference_date", "report_date"], as_index=False)
        .sum()
    )
    places = (NATIONAL, *sorted(counts["location"].unique()))
    first = counts["reference_date"].min()
    out = []
    for cut in cuts:
        end = epiweek_end(cut) + dt.timedelta(weeks=AHEAD_WEEKS)
        if end > dt.date.fromisoformat(last_week):
            raise ValueError(f"cut {cut} forecasts {end}, past {last_week}")
        weekday = dt.date.fromisoformat(cut).weekday()
        block = training_block(
            counts,
            "weekly",
            places,
            weekly_cutoffs(first, cut, weekday),
            max_lag=lag,
        )
        padded = _pad(block, CONTEXT)
        padded["levels"] = target_levels(places)
        padded["places"] = tuple(places)
        padded["mix"] = "real"
        padded["source"] = "weekly"
        padded["cadence"] = "weekly"
        out.append(padded)
    return out
