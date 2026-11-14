# Copyright (c) 2026 Sam Abbott. MIT licence, see LICENSE.
"""Blocks, batches and memory budget for fine-tuning Chronos-2.

Cuts outbreaks from `scripts/finetune_simulate.R` into training blocks
(`outbreak_blocks`), packs them into batches within a memory budget
(`plan_batches`, `max_series`) and scores held-out blocks
(`held_out_loss`). `scripts/finetune_stream.py` trains on them and
publishes each checkpoint as the release `chronos-ft-<hash>`, which the
`_ft` models load the newest one
(`forecasters.foundation.resolve_checkpoint`).
"""

from __future__ import annotations

import datetime as dt
import hashlib
import inspect
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

from bdbvhubrenewal.forecasters import foundation
from bdbvhubrenewal.forecasters.base import NATIONAL
from bdbvhubrenewal.forecasters.finetune import training_block, weekly_cutoffs

CONTEXT = 288
"""Steps of context in a block, left-padded with missing values."""

HORIZON = 96
"""Steps of horizon in a block, right-padded with missing values: the
longest daily horizon is 6 + 28 + 49 days, in six patches of 16."""

MAX_LAGS = {"weekly": tuple(range(2, 8)), "daily": tuple(range(14, 50))}
"""Release lags $D$ the blocks are drawn from."""

MEMORY_GB = 36.0
"""The GPU memory budget of a training step, in GB: the trainer's cap on
MPS memory (0.85 of a 55.7 GB working set) less room for the model and
the held-out set."""

MAX_SERIES = 2000
"""Most series in a block. A block's series count is drawn log-uniformly
up to it. With gradient checkpointing a block of 2,400 series fits the
laptop's GPU memory (64 GB) but peaks at about 56 GB in training, so
2,000 leaves headroom."""

CUTS = 8
"""Cut-off slots of each source (weekly, daily) per observation draw."""

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


CODE = (
    "scripts/finetune_simulate.R",
    "scripts/finetune_chronos.py",
    "src/bdbvhubrenewal/forecasters/finetune.py",
)
"""Files the checkpoint depends on, hashed into its settings."""

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

SIM_DELAY_WEEKS = 8
"""The simulations' longest delay in weeks, after which a count is
complete, unless an outbreak's settings give `max_delay` in days."""


def _sha256(path: Path) -> str:
    """Return a file's SHA-256."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _pad(block: dict, context: int, horizon: int) -> dict:
    """Pad a block to `context + horizon` steps, cropping long context."""
    import torch

    start, h = block["start"], block["horizon"]
    x = block["context"]
    keep = min(start, context)
    out = np.full((x.shape[0], context + horizon), np.nan, dtype=np.float32)
    out[:, context - keep : context + h] = x[:, start - keep : start + h]
    return {
        "context": torch.from_numpy(out),
        "future_covariates": torch.from_numpy(out)[:, context:],
        "n_targets": block["n_targets"],
        "n_covariates": block["n_covariates"],
        "n_future_covariates": block["n_future_covariates"],
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


LEVEL_MIX = ("all_levels", "leaves_national", "leaves", "national")
"""The blocks of an outbreak with strata, a quarter each: the country,
intermediate totals and leaves (as the deployed block: national,
provinces, zones), leaves and the country, leaves only, or the country
only. All levels need nested strata (regions of places, or places by
ages)."""

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


def choose_places(
    provinces: list[str], budget: int, rng: np.random.Generator
) -> tuple[tuple[str, ...], str]:
    """Return a block's places and its level mix (`LEVEL_MIX`).

    At most `budget` places are below the country. All levels are the
    country, the totals of the chosen leaves' groups (and, for places by
    ages, the age groups' totals) and the leaves.
    """
    nested = any("-" in p for p in provinces)
    mixes = LEVEL_MIX if nested else LEVEL_MIX[1:]
    mix = str(rng.choice(mixes))
    # A block's size gives way to its level mix: leaves need at least one
    # place below the country, all levels a group's total and a leaf.
    budget = max(budget, 2 if mix == "all_levels" else 1)
    if mix == "national":
        return (NATIONAL,), mix
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


def outbreak_blocks(
    counts: pd.DataFrame,
    provinces: list[str],
    rng: np.random.Generator,
    delay_weeks: int = SIM_DELAY_WEEKS,
    cuts: int = CUTS,
    picks: set[tuple[str, int]] | None = None,
) -> list[dict]:
    """Return the training blocks of one outbreak.

    Each source is cut at `cuts` forecast origins spread over the
    outbreak's course, one in each of `cuts` equal stretches of its
    cut-offs, so they fall in its rising, peak and falling phases. With
    `picks`, only those (source, slot) blocks are cut. A block's series
    count is drawn log-uniformly from 1 to `MAX_SERIES`,
    and its places (`choose_places`) are about that many series at its
    lag, with the country first.
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
    for source in ("weekly", "daily"):
        for j in range(cuts):
            if picks is not None and (source, j) not in picks:
                continue
            lag = int(rng.choice(MAX_LAGS[source]))
            width = 7 if source == "weekly" else 1
            # At least two steps of context before the lags.
            least = -(-(lag + 2) * width // 7)
            if len(cutoffs) <= least:
                continue
            share = (j + rng.random()) / cuts
            cut = least + int(share * (len(cutoffs) - least))
            places: tuple[str, ...] = (NATIONAL,)
            mix = "national"
            if len(provinces) > 1:
                size = np.exp(rng.uniform(0, np.log(MAX_SERIES)))
                fits = MAX_SERIES // (lag + 1) - 1
                budget = min(round(size / (lag + 1)) - 1, fits)
                places, mix = choose_places(provinces, budget, rng)
            # Leaves only, with nothing reported by the cut, have nothing
            # to lay out.
            if NATIONAL not in places:
                seen = counts["location"].isin(places) & (
                    pd.to_datetime(counts["report_date"])
                    <= pd.Timestamp(cutoffs[cut])
                )
                if not seen.any():
                    continue
            block = training_block(
                counts,
                source,
                places,
                cutoffs[: cut + 1],
                max_lag=lag,
                totals=any("-" in p for p in provinces),
            )
            if block["start"] < 2:
                continue
            padded = _pad(block, CONTEXT, HORIZON)
            padded["levels"] = target_levels(places)
            padded["mix"] = mix
            out.append(padded)
    return out


def checkpoint_large_batches(above: int) -> None:
    """Recompute encoder activations in backward for large batches.

    Chronos-2's group attention runs over every series of a batch, so its
    memory grows with their square. Above `above` series each encoder block
    is checkpointed (`torch.utils.checkpoint`).
    """
    import functools

    import torch
    from chronos.chronos2.model import Chronos2EncoderBlock
    from torch.utils.checkpoint import checkpoint

    forward = Chronos2EncoderBlock.forward

    def checkpointed(self, hidden_states, **kwargs):
        large = hidden_states.shape[0] > above
        if large and self.training and torch.is_grad_enabled():
            return checkpoint(
                functools.partial(forward, self, **kwargs),
                hidden_states,
                use_reentrant=False,
            )
        return forward(self, hidden_states, **kwargs)

    Chronos2EncoderBlock.forward = checkpointed


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


def _order(outbreak: str) -> tuple[int, int]:
    """Sort key of an outbreak id, `<seed>-<index>`."""
    seed, index = outbreak.split("-")
    return int(seed), int(index)


def held_out_loss(model, blocks: list[dict], batch_size: int) -> float:
    """Return the mean quantile loss of a Chronos-2 model on blocks."""
    import torch
    from chronos.chronos2.dataset import Chronos2Dataset, DatasetMode
    from chronos.chronos2.model import Chronos2Model

    data = Chronos2Dataset(
        inputs=blocks,
        context_length=CONTEXT,
        prediction_length=HORIZON,
        batch_size=batch_size,
        output_patch_size=OUTPUT_PATCH,
        mode=DatasetMode.VALIDATION,
    )
    accepted = inspect.signature(Chronos2Model.forward).parameters
    training = model.training
    model.eval()
    losses, weights = [], []
    with torch.no_grad():
        for batch in data:
            batch = {
                k: v.to(model.device) if torch.is_tensor(v) else v
                for k, v in batch.items()
                if k in accepted
            }
            losses.append(float(model(**batch).loss))
            weights.append(len(batch["context"]))
    model.train(training)
    return float(np.average(losses, weights=weights))


def _settings_hash(settings: dict) -> str:
    """Return the first 12 hex digits of the settings' SHA-256."""
    text = json.dumps(settings, sort_keys=True).encode()
    return hashlib.sha256(text).hexdigest()[:12]
