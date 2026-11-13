# Copyright (c) 2026 Sam Abbott. MIT licence, see LICENSE.
"""Fine-tune Chronos-2 with LoRA on simulated outbreaks.

`blocks` cuts the outbreaks from `scripts/finetune_simulate.R` into
training blocks (`forecasters.finetune.training_block`), and `train`
fine-tunes Chronos-2 on them and saves one checkpoint with its settings
hash. Run from the repo root:

    uv run --group chronos python scripts/finetune_chronos.py blocks \
        output/finetune/sims output/finetune/blocks.pt
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

MAX_PLACES = {"weekly": 26, "daily": 5}
"""Most provinces in a block, so a daily block fits GPU memory."""

CUTS = 4
"""Blocks of each source per outbreak."""

CODE = (
    "scripts/finetune_simulate.R",
    "scripts/finetune_chronos.py",
    "src/bdbvhubrenewal/forecasters/finetune.py",
)
"""Files the checkpoint depends on, hashed into its settings."""

FOUNDATION_CODE = (foundation.hub_triangle, foundation.layout_block)
"""The functions of `foundation` the blocks are laid out with, hashed into
the settings."""

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


def outbreak_blocks(
    counts: pd.DataFrame, rng: np.random.Generator
) -> list[dict]:
    """Return the training blocks of one outbreak."""
    # Cut-offs fall Wednesday to Saturday, as the hub's releases do.
    weekday = int(rng.integers(2, 6))
    first = str(counts["report_date"].min())
    end = pd.Timestamp(counts["report_date"].max()).date()
    last = end - dt.timedelta(weeks=SIM_DELAY_WEEKS + 5)
    cutoffs = weekly_cutoffs(first, last.isoformat(), weekday)
    provinces = sorted(counts["location"].unique())
    out = []
    for source in ("weekly", "daily"):
        for _ in range(CUTS):
            lag = int(rng.choice(MAX_LAGS[source]))
            width = 7 if source == "weekly" else 1
            # At least two steps of context before the lags.
            least = -(-(lag + 2) * width // 7)
            if len(cutoffs) <= least:
                continue
            cut = int(rng.integers(least, len(cutoffs)))
            places: tuple[str, ...] = (NATIONAL,)
            if len(provinces) > 1:
                k = int(
                    rng.integers(1, min(len(provinces), MAX_PLACES[source]) + 1)
                )
                chosen = rng.choice(provinces, size=k, replace=False)
                places = (NATIONAL, *sorted(chosen))
            block = training_block(
                counts, source, places, cutoffs[: cut + 1], max_lag=lag
            )
            if block["start"] < 2:
                continue
            out.append(_pad(block, CONTEXT, HORIZON))
    return out


def blocks(sim_dir: Path, out: Path, seed: int) -> None:
    """Write every outbreak's training blocks to one file."""
    import torch

    rng = np.random.default_rng(seed)
    counts = pd.concat(
        [pd.read_csv(p) for p in sorted(sim_dir.glob("counts-*.csv.gz"))]
    )
    t0 = time.perf_counter()
    written = []
    for _, rows in counts.groupby("outbreak"):
        written.extend(outbreak_blocks(rows, rng))
    seconds = time.perf_counter() - t0
    torch.save(written, out)
    series = [int(b["context"].shape[0]) for b in written]
    print(
        f"{len(written)} blocks from {counts['outbreak'].nunique()} outbreaks "
        f"in {seconds:.1f} s ({len(written) / seconds:.1f} blocks/s); "
        f"series per block median {int(np.median(series))}, max {max(series)}"
    )


def train(
    blocks_path: Path,
    out: Path,
    steps: int,
    learning_rate: float,
    batch_size: int,
    holdout: float,
    seed: int,
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

    torch.manual_seed(seed)
    # Chronos-2 cuts its training batches with numpy's global generator.
    np.random.seed(seed)  # noqa: NPY002
    data = torch.load(blocks_path, weights_only=False)
    order = np.random.default_rng(seed).permutation(len(data))
    n_val = round(holdout * len(data))
    val = [data[i] for i in order[:n_val]]
    fit = [data[i] for i in order[n_val:]]
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    pipeline = Chronos2Pipeline.from_pretrained(
        CHRONOS_MODEL, revision=CHRONOS_REVISION, device_map=device
    )
    history = History()
    pipeline.fit(
        fit,
        prediction_length=HORIZON,
        validation_inputs=val or None,
        finetune_mode="lora",
        context_length=CONTEXT,
        min_past=CONTEXT,
        learning_rate=learning_rate,
        num_steps=steps,
        batch_size=batch_size,
        output_dir=out,
        callbacks=[history],
        logging_steps=10,
        eval_steps=50,
        save_steps=50,
        eval_on_start=bool(val),
        optim="adamw_torch",
        seed=seed,
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
    b.add_argument("sim_dir", type=Path)
    b.add_argument("out", type=Path)
    b.add_argument("--seed", type=int, default=1)
    t = sub.add_parser("train")
    t.add_argument("blocks", type=Path)
    t.add_argument("out", type=Path)
    t.add_argument("--steps", type=int, default=1000)
    t.add_argument("--learning-rate", type=float, default=1e-4)
    t.add_argument("--batch-size", type=int, default=256)
    t.add_argument("--holdout", type=float, default=0.05)
    t.add_argument("--seed", type=int, default=1)
    args = parser.parse_args()
    if args.command == "blocks":
        blocks(args.sim_dir, args.out, args.seed)
    else:
        train(
            args.blocks,
            args.out,
            args.steps,
            args.learning_rate,
            args.batch_size,
            args.holdout,
            args.seed,
        )


if __name__ == "__main__":
    main()
