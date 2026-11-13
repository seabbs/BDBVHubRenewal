# Copyright (c) 2026 Sam Abbott. MIT licence, see LICENSE.
"""Fine-tune Chronos-2 with LoRA on simulated outbreaks.

`blocks` cuts the outbreaks from `scripts/finetune_simulate.R` into
training blocks (`forecasters.finetune.training_block`), and `train`
fine-tunes Chronos-2 on them and saves one checkpoint with its settings
hash. Run from the repo root:

    uv run --group chronos python scripts/finetune_chronos.py blocks \
        output/finetune/blocks.pt output/finetune/sims...
    uv run --group chronos python scripts/finetune_chronos.py train \
        output/finetune/blocks.pt output/finetune/ckpt

The `_ft` models download the checkpoint from the release
`chronos-ft-<hash>` (`forecasters.foundation.finetuned_adapter`).
Publish it, then set its hash and adapter SHA-256 in
`config/finetune.toml`:

    gh release create chronos-ft-<hash> -R seabbs/BDBVHubRenewal \
        --latest=false --title "Chronos-2 fine-tuned checkpoint <hash>" \
        --notes "LoRA adapter of Chronos-2 for the _ft models." \
        output/finetune/ckpt/finetuned-ckpt/adapter_config.json \
        output/finetune/ckpt/finetuned-ckpt/adapter_model.safetensors \
        output/finetune/ckpt/settings.json
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

MAX_SERIES = 2000
"""Most series in a block. A block's series count is drawn log-uniformly
up to it. With gradient checkpointing a block of 2,400 series fits the
laptop's GPU memory (64 GB) but peaks at about 56 GB in training, so
2,000 leaves headroom."""

CUTS = {"weekly": 1, "daily": 2}
"""Blocks of each source per outbreak."""

CHECKPOINT_SERIES = 900
"""Batches with more series than this recompute each encoder block's
activations in the backward pass (gradient checkpointing), to fit
memory."""

CODE = (
    "scripts/finetune_simulate.R",
    "scripts/finetune_chronos.py",
    "src/bdbvhubrenewal/forecasters/finetune.py",
)
"""Files the checkpoint depends on, hashed into its settings."""

FOUNDATION_CODE = (foundation.hub_triangle, foundation.layout_block)
"""The functions of `foundation` the blocks are laid out with, hashed into
the settings."""

ROUND2_SEED = 3001
"""First seed of the round 2 grid's simulations."""

HELD_OUT_SEED = 3900
"""First seed of the held-out outbreaks, simulated apart from every
training set."""

EVAL_BLOCKS = 100
"""Held-out blocks whose loss is logged during training."""

EVAL_EVERY = 100
"""Steps between logged held-out losses."""

SAVE_EVERY = 1000
"""Steps between saved checkpoints."""

SIM_DELAY_WEEKS = 8
"""The simulations' longest delay, after which a count is complete."""


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
        "future_covariates": torch.from_numpy(out[:, context:].copy()),
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


def outbreak_blocks(
    counts: pd.DataFrame, provinces: list[str], rng: np.random.Generator
) -> list[dict]:
    """Return the training blocks of one outbreak.

    A block's series count is drawn log-uniformly from 1 to `MAX_SERIES`,
    and its strata are a random subset of `provinces` of about that many
    series at its lag, with the total over strata first.
    """
    # Cut-offs fall Wednesday to Saturday, as the hub's releases do.
    weekday = int(rng.integers(2, 6))
    first = str(counts["report_date"].min())
    end = pd.Timestamp(counts["report_date"].max()).date()
    last = end - dt.timedelta(weeks=SIM_DELAY_WEEKS + 5)
    cutoffs = weekly_cutoffs(first, last.isoformat(), weekday)
    out = []
    for source in ("weekly", "daily"):
        for _ in range(CUTS[source]):
            lag = int(rng.choice(MAX_LAGS[source]))
            width = 7 if source == "weekly" else 1
            # At least two steps of context before the lags.
            least = -(-(lag + 2) * width // 7)
            if len(cutoffs) <= least:
                continue
            cut = int(rng.integers(least, len(cutoffs)))
            places: tuple[str, ...] = (NATIONAL,)
            if len(provinces) > 1:
                size = np.exp(rng.uniform(0, np.log(MAX_SERIES)))
                fits = MAX_SERIES // (lag + 1) - 1
                k = min(round(size / (lag + 1)) - 1, len(provinces), fits)
                if k > 0:
                    chosen = rng.choice(provinces, size=k, replace=False)
                    places = (NATIONAL, *sorted(chosen))
            block = training_block(
                counts, source, places, cutoffs[: cut + 1], max_lag=lag
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
        cut = outbreak_blocks(rows, provinces, rng)
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


def checkpoint_large_batches() -> None:
    """Recompute encoder activations in backward for large batches.

    Chronos-2's group attention runs over every series of a batch, so its
    memory grows with their square. Above `CHECKPOINT_SERIES` series each
    encoder block is checkpointed (`torch.utils.checkpoint`).
    """
    import functools

    import torch
    from chronos.chronos2.model import Chronos2EncoderBlock
    from torch.utils.checkpoint import checkpoint

    forward = Chronos2EncoderBlock.forward

    def checkpointed(self, hidden_states, **kwargs):
        large = hidden_states.shape[0] > CHECKPOINT_SERIES
        if large and self.training and torch.is_grad_enabled():
            return checkpoint(
                functools.partial(forward, self, **kwargs),
                hidden_states,
                use_reentrant=False,
            )
        return forward(self, hidden_states, **kwargs)

    Chronos2EncoderBlock.forward = checkpointed


def size_buckets(sizes: np.ndarray) -> list[np.ndarray]:
    """Return the blocks of each size class, by powers of two of series."""
    classes = np.floor(np.log2(sizes)).astype(int)
    return [np.flatnonzero(classes == c) for c in np.unique(classes)]


def bucketed_batches(
    sizes: np.ndarray, batch_size: int, rng: np.random.Generator
):
    """Yield the block indices of each training batch, packed by size.

    A batch's size class is drawn with probability its share of the
    series, and its blocks are drawn at random from that class until the
    batch holds `batch_size` series, so batches of small blocks pack
    tightly, a large block is a batch on its own, and every series is as
    likely to be seen.
    """
    buckets = size_buckets(sizes)
    share = np.array([sizes[b].sum() for b in buckets], dtype=float)
    share /= share.sum()
    while True:
        bucket = buckets[rng.choice(len(buckets), p=share)]
        held, out = 0, []
        while held < batch_size:
            i = int(bucket[rng.integers(len(bucket))])
            out.append(i)
            held += int(sizes[i])
        yield out


def use_bucketed_batches(sizes: np.ndarray, seed: int) -> None:
    """Make Chronos-2's training batches `bucketed_batches`."""
    from chronos.chronos2.dataset import Chronos2Dataset

    def generate(self):
        rng = np.random.default_rng(seed)
        for indices in bucketed_batches(sizes, self.batch_size, rng):
            yield self._build_batch(indices)

    Chronos2Dataset._generate_train_batches = generate


def steps_for(sizes: np.ndarray, batch_size: int, epochs: float) -> int:
    """Return the steps that pass over every series `epochs` times.

    The mean series of a `bucketed_batches` batch is estimated from 5,000
    simulated batches.
    """
    batches = bucketed_batches(sizes, batch_size, np.random.default_rng(0))
    held = [sizes[next(batches)].sum() for _ in range(5000)]
    return int(np.ceil(epochs * sizes.sum() / np.mean(held)))


def _order(outbreak: str) -> tuple[int, int]:
    """Sort key of an outbreak id, `<seed>-<index>`."""
    seed, index = outbreak.split("-")
    return int(seed), int(index)


def training_sets(
    outbreaks: list[str], round1: bool, round2: int, held_out: int
) -> tuple[list[str], list[str]]:
    """Return the training and held-out outbreaks.

    Training is every round 1 outbreak (seeds below `ROUND2_SEED`) if
    `round1`, and the first `round2` round 2 outbreaks in order of seed and
    index, from seeds `ROUND2_SEED` to `HELD_OUT_SEED`. The held-out
    outbreaks are the first `held_out` from seeds `HELD_OUT_SEED` on, so
    nested training sets all hold out the same outbreaks and blocks.
    """
    ids = sorted(set(outbreaks), key=_order)
    first = [o for o in ids if _order(o)[0] < ROUND2_SEED]
    second = [o for o in ids if ROUND2_SEED <= _order(o)[0] < HELD_OUT_SEED]
    rest = [o for o in ids if _order(o)[0] >= HELD_OUT_SEED]
    if len(second) < round2 or len(rest) < held_out:
        raise ValueError(
            f"{len(second)} round 2 and {len(rest)} held-out outbreaks, "
            f"fewer than {round2} and {held_out}."
        )
    train = (first if round1 else []) + second[:round2]
    return train, rest[:held_out]


def held_out_loss(pipeline, blocks: list[dict], batch_size: int) -> float:
    """Return the mean quantile loss of a pipeline on held-out blocks."""
    import torch
    from chronos.chronos2.dataset import Chronos2Dataset, DatasetMode
    from chronos.chronos2.model import Chronos2Model

    data = Chronos2Dataset(
        inputs=blocks,
        context_length=CONTEXT,
        prediction_length=HORIZON,
        batch_size=batch_size,
        output_patch_size=pipeline.model_output_patch_size,
        mode=DatasetMode.VALIDATION,
    )
    model = pipeline.model.eval()
    accepted = inspect.signature(Chronos2Model.forward).parameters
    losses, weights = [], []
    with torch.no_grad():
        for batch in data:
            batch = {
                k: v.to(model.device) if torch.is_tensor(v) else v
                for k, v in batch.items()
                if k in accepted
            }
            loss = model(**batch).loss
            losses.append(float(loss))
            weights.append(len(batch["context"]))
    return float(np.average(losses, weights=weights))


def train(
    blocks_path: Path,
    out: Path,
    epochs: float,
    learning_rate: float,
    batch_size: int,
    holdout: int,
    seed: int,
    round1: bool,
    round2: int,
) -> None:
    """Fine-tune Chronos-2 with LoRA and save the checkpoint."""
    import torch
    from chronos import Chronos2Pipeline
    from transformers import TrainerCallback

    class History(TrainerCallback):
        """Keeps the trainer's logs."""

        def __init__(self) -> None:
            self.logs: list[dict] = []
            self.t0 = time.perf_counter()

        def on_log(self, _args, state, _control, logs=None, **_kwargs):
            if logs:
                self.logs.append(
                    {
                        "step": state.global_step,
                        "seconds": time.perf_counter() - self.t0,
                        **logs,
                    }
                )
                print(self.logs[-1], flush=True)

    checkpoint_large_batches()
    torch.manual_seed(seed)
    # Chronos-2 cuts its training batches with numpy's global generator.
    np.random.seed(seed)  # noqa: NPY002
    saved = torch.load(blocks_path, weights_only=False)
    train_ids, held_ids = training_sets(
        saved["outbreaks"], round1, round2, holdout
    )
    out.mkdir(parents=True, exist_ok=True)
    (out / "outbreaks-train.txt").write_text("\n".join(train_ids) + "\n")
    (out / "outbreaks-held-out.txt").write_text("\n".join(held_ids) + "\n")
    use, held = set(train_ids), set(held_ids)
    val, fit = [], []
    for block, outbreak in zip(
        saved["blocks"], saved["outbreaks"], strict=True
    ):
        if outbreak in held:
            val.append(block)
        elif outbreak in use:
            fit.append(block)
    # The loss logged during training is on a fixed sample of the held-out
    # blocks, and the whole held-out set is scored at the start and end.
    monitor = val[:EVAL_BLOCKS]
    sizes = np.array([int(b["context"].shape[0]) for b in fit])
    use_bucketed_batches(sizes, seed)
    steps = steps_for(sizes, batch_size, epochs)
    print(f"{len(fit)} blocks, {sizes.sum()} series: {steps} steps")
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    pipeline = Chronos2Pipeline.from_pretrained(
        CHRONOS_MODEL, revision=CHRONOS_REVISION, device_map=device
    )
    history = History()
    before = held_out_loss(pipeline, val, batch_size)
    print({"held_out_loss": before, "step": 0}, flush=True)
    tuned = pipeline.fit(
        fit,
        prediction_length=HORIZON,
        validation_inputs=monitor or None,
        finetune_mode="lora",
        context_length=CONTEXT,
        min_past=CONTEXT,
        learning_rate=learning_rate,
        num_steps=steps,
        batch_size=batch_size,
        output_dir=out,
        callbacks=[history],
        logging_steps=10,
        eval_steps=EVAL_EVERY,
        save_steps=SAVE_EVERY,
        save_total_limit=None,
        eval_on_start=bool(val),
        optim="adamw_torch",
        seed=seed,
    )
    after = held_out_loss(tuned, val, batch_size)
    print({"held_out_loss": after, "step": steps}, flush=True)
    history.logs.append(
        {"held_out_loss_start": before, "held_out_loss_end": after}
    )
    (out / "history.json").write_text(json.dumps(history.logs, indent=1))
    settings = {
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
        "n_fit": len(fit),
        "n_validation": len(val),
        "n_round1_outbreaks": sum(
            _order(o)[0] < ROUND2_SEED for o in train_ids
        ),
        "n_round2_outbreaks": round2,
        "n_held_out_outbreaks": len(held_ids),
        "outbreaks_sha256": {
            f: _sha256(out / f)
            for f in ("outbreaks-train.txt", "outbreaks-held-out.txt")
        },
        "epochs": epochs,
        "steps": steps,
        "learning_rate": learning_rate,
        "batch_size": batch_size,
        "context": CONTEXT,
        "horizon": HORIZON,
        "seed": seed,
    }
    key = hashlib.sha256(
        json.dumps(settings, sort_keys=True).encode()
    ).hexdigest()[:12]
    settings["hash"] = key
    settings["device"] = device
    (out / "settings.json").write_text(json.dumps(settings, indent=1))
    print(f"settings hash {key}")


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
    t.add_argument("--epochs", type=float, default=1.0)
    t.add_argument("--learning-rate", type=float, default=1e-4)
    t.add_argument("--batch-size", type=int, default=256)
    t.add_argument("--holdout", type=int, default=150)
    t.add_argument(
        "--round1", action=argparse.BooleanOptionalAction, default=False
    )
    t.add_argument("--round2", type=int, required=True)
    t.add_argument("--seed", type=int, default=1)
    args = parser.parse_args()
    if args.command == "blocks":
        blocks(args.sim_dirs, args.out, args.seed)
    else:
        train(
            args.blocks,
            args.out,
            args.epochs,
            args.learning_rate,
            args.batch_size,
            args.holdout,
            args.seed,
            args.round1,
            args.round2,
        )


if __name__ == "__main__":
    main()
