# Copyright (c) 2026 Sam Abbott. MIT licence, see LICENSE.
"""Fine-tune Chronos-2 with LoRA on simulated outbreaks.

`blocks` cuts the outbreaks from `scripts/finetune_simulate.R` into
training blocks (`forecasters.finetune.training_block`), and `train`
fine-tunes Chronos-2 on them in one pass, saving a checkpoint with its
settings hash after each prefix of the outbreaks (`PREFIXES`). Run from
the repo root:

    uv run --group chronos python scripts/finetune_chronos.py blocks \
        output/finetune/blocks.pt output/finetune/sims...
    uv run --group chronos python scripts/finetune_chronos.py train \
        output/finetune/blocks.pt output/finetune/ckpt

The `_ft` models download the checkpoint from the release
`chronos-ft-<hash>` (`forecasters.foundation.finetuned_adapter`).
Publish a checkpoint (`output/finetune/ckpt/n<N>`), then set its hash and
adapter SHA-256 in `config/finetune.toml`:

    gh release create chronos-ft-<hash> -R seabbs/BDBVHubRenewal \
        --latest=false --title "Chronos-2 fine-tuned checkpoint <hash>" \
        --notes "LoRA adapter of Chronos-2 for the _ft models." \
        output/finetune/ckpt/n<N>/finetuned-ckpt/adapter_config.json \
        output/finetune/ckpt/n<N>/finetuned-ckpt/adapter_model.safetensors \
        output/finetune/ckpt/n<N>/settings.json
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import inspect
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from bdbvhubrenewal.forecasters import foundation
from bdbvhubrenewal.forecasters.base import NATIONAL
from bdbvhubrenewal.forecasters.finetune import training_block, weekly_cutoffs
from bdbvhubrenewal.forecasters.foundation import (
    CHRONOS_MODEL,
    CHRONOS_REVISION,
)

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
"""Cut-offs of each source (weekly, daily) per simulation, shared among
its observation draws."""

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

TRAIN_SEEDS = (5001, 5901)
"""Seeds of the training outbreaks' simulations, from and to (excluded)."""

HELD_OUT_SEED = 5901
"""First seed of the held-out outbreaks, simulated apart from the
training outbreaks."""

PREFIXES = (1000, 2000, 4000, 6000, 8000)
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


LEVELS = {"three": 0.5, "places": 0.25, "strata": 0.25}
"""Chance a block of places by strata (`P-<stratum>`) has the country,
the places' totals and their strata (three levels), the country and the
places' totals, or the country and strata."""


def choose_places(
    provinces: list[str], budget: int, rng: np.random.Generator
) -> tuple[str, ...]:
    """Return a block's places below the country, at most `budget` of them.

    Strata of places (`P-<stratum>`) give blocks of `LEVELS`; other
    provinces give a random subset.
    """
    if budget < 1:
        return ()
    if not any("-" in p for p in provinces):
        k = min(budget, len(provinces))
        return tuple(str(p) for p in sorted(rng.choice(provinces, k, False)))
    groups: dict[str, list[str]] = {}
    for p in provinces:
        groups.setdefault(p.split("-")[0], []).append(p)
    level = rng.choice(list(LEVELS), p=list(LEVELS.values()))
    if level == "strata":
        k = min(budget, len(provinces))
        return tuple(str(p) for p in sorted(rng.choice(provinces, k, False)))
    parents = list(rng.permutation(sorted(groups)))
    if level == "places" or budget < 2:
        return tuple(str(p) for p in sorted(parents[:budget]))
    chosen: list[str] = []
    strata: list[str] = []
    for parent in parents:
        room = budget - len(chosen) - len(strata) - 1
        if room < 1:
            break
        chosen.append(parent)
        strata.extend(sorted(groups[parent])[:room])
        if room < len(groups[parent]):
            break
    return tuple(str(p) for p in (*sorted(chosen), *sorted(strata)))


def outbreak_blocks(
    counts: pd.DataFrame,
    provinces: list[str],
    rng: np.random.Generator,
    delay_weeks: int = SIM_DELAY_WEEKS,
    cuts: int = CUTS,
) -> list[dict]:
    """Return the training blocks of one outbreak.

    Each source is cut at `cuts` forecast origins spread over the
    outbreak's course, one in each of `cuts` equal stretches of its
    cut-offs, so they fall in its rising, peak and falling phases. A
    block's series count is drawn log-uniformly from 1 to `MAX_SERIES`,
    and its places (`choose_places`) are about that many series at its
    lag, with the country first.
    """
    # Cut-offs fall Wednesday to Saturday, as the hub's releases do.
    weekday = int(rng.integers(2, 6))
    first = str(counts["report_date"].min())
    end = pd.Timestamp(counts["report_date"].max()).date()
    last = end - dt.timedelta(weeks=delay_weeks + 5)
    cutoffs = weekly_cutoffs(first, last.isoformat(), weekday)
    out = []
    for source in ("weekly", "daily"):
        for j in range(cuts):
            lag = int(rng.choice(MAX_LAGS[source]))
            width = 7 if source == "weekly" else 1
            # At least two steps of context before the lags.
            least = -(-(lag + 2) * width // 7)
            if len(cutoffs) <= least:
                continue
            share = (j + rng.random()) / cuts
            cut = least + int(share * (len(cutoffs) - least))
            places: tuple[str, ...] = (NATIONAL,)
            if len(provinces) > 1:
                size = np.exp(rng.uniform(0, np.log(MAX_SERIES)))
                fits = MAX_SERIES // (lag + 1) - 1
                budget = min(round(size / (lag + 1)) - 1, fits)
                places = (NATIONAL, *choose_places(provinces, budget, rng))
            block = training_block(
                counts,
                source,
                places,
                cutoffs[: cut + 1],
                max_lag=lag,
                totals=any("-" in p for p in places),
            )
            if block["start"] < 2:
                continue
            out.append(_pad(block, CONTEXT, HORIZON))
    return out


def blocks(sim_dirs: list[Path], out: Path, seed: int) -> None:
    """Write the training blocks of every outbreak in `sim_dirs` to one file.

    Each outbreak's blocks are cut with a generator seeded by `seed` and
    its id, so an outbreak gives the same blocks in any set of outbreaks.
    The file holds the blocks and the outbreak of each.
    """
    import zlib

    import torch

    counts = pd.concat(
        [
            pd.read_csv(p)
            for d in sim_dirs
            for p in sorted(d.glob("counts-*.csv.gz"))
        ]
    )
    settings = pd.concat(
        [
            pd.read_csv(p)
            for d in sim_dirs
            for p in sorted(d.glob("settings-*.csv"))
        ]
    ).set_index("outbreak")
    t0 = time.perf_counter()
    written, sources = [], []
    for outbreak, rows in counts.groupby("outbreak"):
        provinces = strata(settings.loc[outbreak])
        rng = np.random.default_rng([seed, zlib.crc32(outbreak.encode())])
        days = settings.loc[outbreak].get("max_delay", np.nan)
        weeks = SIM_DELAY_WEEKS if pd.isna(days) else int(days) // 7
        cut = outbreak_blocks(rows, provinces, rng, weeks)
        written.extend(cut)
        sources.extend([outbreak] * len(cut))
    seconds = time.perf_counter() - t0
    torch.save({"blocks": written, "outbreaks": sources}, out)
    series = [int(b["context"].shape[0]) for b in written]
    print(
        f"{len(written)} blocks from {counts['outbreak'].nunique()} outbreaks "
        f"in {seconds:.1f} s ({len(written) / seconds:.1f} blocks/s); "
        f"series per block median {int(np.median(series))}, max {max(series)}"
    )


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


def use_planned_batches(plan: list[list[int]]) -> None:
    """Make Chronos-2's training batches the planned ones, in order."""
    from chronos.chronos2.dataset import Chronos2Dataset

    def generate(self):
        for indices in plan:
            yield self._build_batch(indices)

    Chronos2Dataset._generate_train_batches = generate


def _order(outbreak: str) -> tuple[int, int]:
    """Sort key of an outbreak id, `<seed>-<index>`."""
    seed, index = outbreak.split("-")
    return int(seed), int(index)


def training_order(
    outbreaks: list[str], n: int, held_out: int, seed: int
) -> tuple[list[str], list[str]]:
    """Return the training outbreaks in training order, and the held-out.

    Training is the first `n` outbreaks of `TRAIN_SEEDS` in order of seed
    and index, shuffled once with `seed`, so its first N are the outbreaks
    of the checkpoint after N. The held-out outbreaks are the first
    `held_out` from seeds `HELD_OUT_SEED` on.
    """
    ids = sorted(set(outbreaks), key=_order)
    first, last = TRAIN_SEEDS
    pool = [o for o in ids if first <= _order(o)[0] < last]
    rest = [o for o in ids if _order(o)[0] >= HELD_OUT_SEED]
    if len(pool) < n or len(rest) < held_out:
        raise ValueError(
            f"{len(pool)} training and {len(rest)} held-out outbreaks, "
            f"fewer than {n} and {held_out}."
        )
    pool = pool[:n]
    order = np.random.default_rng(seed).permutation(len(pool))
    return [pool[i] for i in order], rest[:held_out]


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


def _plan(
    saved: dict,
    rank: dict[str, int],
    prefixes: list[int],
    batch_size: int,
    seed: int,
) -> tuple[list[dict], list[list[int]], dict[int, int]]:
    """Return the training blocks, their batches and the prefix steps.

    The blocks of each prefix's new outbreaks are batched apart, so the
    step that ends a prefix has seen exactly its outbreaks.
    """
    rng = np.random.default_rng(seed)
    fit: list[dict] = []
    plan: list[list[int]] = []
    ends: dict[int, int] = {}
    pairs = list(zip(saved["blocks"], saved["outbreaks"], strict=True))
    for lo, hi in zip([0, *prefixes[:-1]], prefixes, strict=True):
        segment = [b for b, o in pairs if lo <= rank.get(o, -1) < hi]
        sizes = np.array([int(b["context"].shape[0]) for b in segment])
        for batch in plan_batches(sizes, batch_size, rng):
            plan.append([len(fit) + i for i in batch])
        fit.extend(segment)
        ends[len(plan)] = hi
    return fit, plan, ends


def train(
    blocks_path: Path,
    out: Path,
    n: int,
    learning_rate: float,
    batch_size: int,
    holdout: int,
    seed: int,
) -> None:
    """Fine-tune Chronos-2 with LoRA in one pass, saving prefix checkpoints.

    The training outbreaks are seen once, in a fixed shuffled order. The
    checkpoint after the first N outbreaks, for each N of `PREFIXES`, is
    saved in `out/n<N>` with its outbreak ids, settings and hash, so each
    is Chronos-2 fine-tuned on N outbreaks in one pass. The learning rate
    is constant, so a checkpoint does not depend on how long the run goes
    on after it.
    """
    import torch
    from chronos import Chronos2Pipeline
    from transformers import TrainerCallback

    checkpoint_large_batches(checkpoint_above(MEMORY_GB))
    torch.manual_seed(seed)
    # Chronos-2 slices its training windows with numpy's global generator.
    np.random.seed(seed)  # noqa: NPY002
    saved = torch.load(blocks_path, weights_only=False)
    order, held_ids = training_order(saved["outbreaks"], n, holdout, seed)
    rank = {o: i for i, o in enumerate(order)}
    held = set(held_ids)
    val = [
        b
        for b, o in zip(saved["blocks"], saved["outbreaks"], strict=True)
        if o in held
    ]
    prefixes = [p for p in PREFIXES if p < n] + [n]
    fit, plan, ends = _plan(saved, rank, prefixes, batch_size, seed)
    del saved
    use_planned_batches(plan)
    series = sum(int(b["context"].shape[0]) for b in fit)
    print(
        f"{len(fit)} blocks, {series} series, {len(plan)} steps; "
        f"checkpoints at steps {sorted(ends)}",
        flush=True,
    )
    out.mkdir(parents=True, exist_ok=True)
    (out / "outbreaks-order.txt").write_text("\n".join(order) + "\n")
    (out / "outbreaks-held-out.txt").write_text("\n".join(held_ids) + "\n")
    base = {
        "chronos_revision": CHRONOS_REVISION,
        "blocks_sha256": _sha256(blocks_path),
        "code_sha256": {
            **{p: _sha256(Path(p)) for p in CODE},
            **{
                f"foundation.{f.__name__}": hashlib.sha256(
                    inspect.getsource(f).encode()
                ).hexdigest()
                for f in FOUNDATION_CODE
            },
        },
        "max_lags": {k: [min(v), max(v)] for k, v in MAX_LAGS.items()},
        "held_out_sha256": _sha256(out / "outbreaks-held-out.txt"),
        "n_validation": len(val),
        "learning_rate": learning_rate,
        "batch_size": batch_size,
        "context": CONTEXT,
        "horizon": HORIZON,
        "seed": seed,
    }
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    pipeline = Chronos2Pipeline.from_pretrained(
        CHRONOS_MODEL, revision=CHRONOS_REVISION, device_map=device
    )
    start_loss = held_out_loss(pipeline.model, val, batch_size)
    history: list[dict] = [{"step": 0, "held_out_loss": start_loss}]
    print(history[-1], flush=True)

    class Checkpoints(TrainerCallback):
        """Keeps the trainer's logs and saves the prefix checkpoints."""

        def __init__(self) -> None:
            self.t0 = time.perf_counter()

        def on_log(self, _args, state, _control, logs=None, **_kwargs):
            if logs:
                row = {"step": state.global_step, **logs}
                row["seconds"] = time.perf_counter() - self.t0
                history.append(row)
                print(row, flush=True)

        def on_step_end(self, _args, state, _control, model=None, **_kw):
            if state.global_step not in ends:
                return
            m = ends[state.global_step]
            where = out / f"n{m}"
            where.mkdir(parents=True, exist_ok=True)
            (where / "outbreaks.txt").write_text("\n".join(order[:m]) + "\n")
            model.save_pretrained(where / "finetuned-ckpt")
            settings = {
                **base,
                "n_outbreaks": m,
                "outbreaks_sha256": _sha256(where / "outbreaks.txt"),
                "steps": state.global_step,
            }
            settings["hash"] = _settings_hash(settings)
            settings["device"] = device
            settings["held_out_loss_start"] = start_loss
            settings["held_out_loss"] = held_out_loss(model, val, batch_size)
            settings["seconds"] = time.perf_counter() - self.t0
            (where / "settings.json").write_text(json.dumps(settings, indent=1))
            history.append(
                {
                    "step": state.global_step,
                    "n_outbreaks": m,
                    "held_out_loss": settings["held_out_loss"],
                    "hash": settings["hash"],
                }
            )
            print(history[-1], flush=True)
            (out / "history.json").write_text(json.dumps(history, indent=1))

    pipeline.fit(
        fit,
        prediction_length=HORIZON,
        validation_inputs=val[:EVAL_BLOCKS] or None,
        finetune_mode="lora",
        context_length=CONTEXT,
        min_past=CONTEXT,
        learning_rate=learning_rate,
        num_steps=len(plan),
        batch_size=batch_size,
        output_dir=out,
        callbacks=[Checkpoints()],
        logging_steps=10,
        eval_steps=EVAL_EVERY,
        save_strategy="no",
        load_best_model_at_end=False,
        lr_scheduler_type="constant",
        optim="adamw_torch",
        seed=seed,
    )
    (out / "history.json").write_text(json.dumps(history, indent=1))


def main() -> None:
    """Run `blocks` or `train`."""
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    b = sub.add_parser("blocks")
    b.add_argument("out", type=Path)
    b.add_argument("sim_dirs", type=Path, nargs="+")
    b.add_argument("--seed", type=int, default=1)
    t = sub.add_parser("train")
    t.add_argument("blocks", type=Path)
    t.add_argument("out", type=Path)
    t.add_argument("--outbreaks", type=int, default=PREFIXES[-1])
    t.add_argument("--learning-rate", type=float, default=1e-4)
    t.add_argument("--batch-size", type=int, default=256)
    t.add_argument("--holdout", type=int, default=150)
    t.add_argument("--seed", type=int, default=1)
    args = parser.parse_args()
    if args.command == "blocks":
        blocks(args.sim_dirs, args.out, args.seed)
    else:
        train(
            args.blocks,
            args.out,
            args.outbreaks,
            args.learning_rate,
            args.batch_size,
            args.holdout,
            args.seed,
        )


if __name__ == "__main__":
    main()
