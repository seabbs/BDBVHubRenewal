# Copyright (c) 2026 Sam Abbott. MIT licence, see LICENSE.
r"""Fine-tune Chronos-2 on the German data's development half alone.

For the SI's comparison of Chronos-2 fine-tuned on real data with the
model fine-tuned on simulations. `blocks` cuts the German hub triangle,
as known on `AS_OF` (the last day of the development half), into
training blocks with `finetune_chronos.german_blocks`: every nowcast
date whose truth, four weeks ahead, is at least `TRUTH_DELAY` days old
by `AS_OF`, weekly and daily sources, Germany alone and with its states
or age groups, lags of `finetune_chronos.GERMAN_LAGS`. `train` trains a
LoRA adapter from base Chronos-2 with Chronos-2's own recipe as
`scripts/finetune_stream.py` uses it (LoRA r 8, alpha 16, its own loss,
the learning rate falling linearly to zero, prediction length the
block's horizon in whole patches) for Chronos-2's default 1,000 steps,
and publishes it as `germany-ft-<hash>`, a tag the `_ft` models'
newest-release lookup does not read.

    python scripts/finetune_germany.py blocks \
        data/raw/germany/COVID-19_hospitalizations.csv OUT/blocks.pkl
    python scripts/finetune_germany.py train OUT/blocks.pkl OUT --publish

The last `HELD_OUT_DAYS` of cuts are held out: their loss is logged, and
they are not trained on.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import inspect
import json
import pickle
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import finetune_chronos as ft
import finetune_stream as fs
import finetune_verify

from bdbvhubrenewal.forecasters.foundation import (
    CHRONOS_MODEL,
    CHRONOS_REVISION,
)

AS_OF = "2022-02-09"
"""The last report date used: the last nowcast date of the German
development half (`config/germany.toml`, `split_date` less a day)."""

TRUTH_DELAY = 40
"""Days a count must have had to be complete by `AS_OF`, as the German
benchmark's truth."""

FIRST_CUT = "2021-04-20"
"""The first nowcast date cut, two weeks after the triangle's first
Meldedatum; cuts with under two steps of context are skipped."""

HELD_OUT_DAYS = 14
"""The latest cuts held out for the logged loss."""

STEPS = 1000
"""Training steps: Chronos-2's default (`Chronos2Pipeline.fit`)."""

PREFIX = "germany-ft"
"""The checkpoint's release tag prefix."""

LOG_EVERY = 10
"""Steps between logged training losses."""


def log(message: str) -> None:
    """Print a timestamped line."""
    stamp = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"{stamp} [germany] {message}", flush=True)


def truth_until(as_of: str = AS_OF, delay: int = TRUTH_DELAY) -> str:
    """Return the last Meldedatum complete for `delay` days by `as_of`."""
    return (dt.date.fromisoformat(as_of) - dt.timedelta(delay)).isoformat()


def cut_dates(first: str, last_truth: str) -> list[str]:
    """Return every nowcast date whose horizon ends by `last_truth`."""
    last = dt.date.fromisoformat(last_truth) - dt.timedelta(
        weeks=ft.AHEAD_WEEKS
    )
    day, out = dt.date.fromisoformat(first), []
    while day <= last:
        out.append(day.isoformat())
        day += dt.timedelta(days=1)
    return out


def _rotated(i: int) -> dict[str, tuple[int, ...]]:
    """Return `GERMAN_LAGS` rotated as `german_blocks` takes them at cut i."""
    return {
        s: v[i % len(v) :] + v[: i % len(v)] for s, v in ft.GERMAN_LAGS.items()
    }


def _cut_one(args: tuple) -> list[dict]:
    counts, cut, i, last_truth = args
    return ft.german_blocks(counts, [cut], last_truth, lags=_rotated(i))


def make_blocks(path: Path, out: Path, workers: int) -> None:
    """Cut the German blocks and pickle them in cut order."""
    counts = ft.german_counts(pd.read_csv(path), AS_OF)
    last_truth = truth_until()
    cuts = cut_dates(FIRST_CUT, last_truth)
    log(f"{len(cuts)} cuts {cuts[0]} to {cuts[-1]}, truth to {last_truth}")
    jobs = [(counts, c, i, last_truth) for i, c in enumerate(cuts)]
    with ProcessPoolExecutor(workers) as pool:
        blocks = [b for part in pool.map(_cut_one, jobs) for b in part]
    with out.open("wb") as f:
        pickle.dump(blocks, f)
    log(f"{len(blocks)} blocks written to {out}")


def block_id(block: dict) -> str:
    """Return a block's id: cut, source, places and lag."""
    return f"{block['cut']}:{block['source']}:{block['places']}:{block['lag']}"


def split_held_out(blocks: list[dict]) -> tuple[list[dict], list[dict]]:
    """Return the training blocks and the last `HELD_OUT_DAYS` of cuts."""
    last = max(dt.date.fromisoformat(b["cut"]) for b in blocks)
    first_held = (last - dt.timedelta(HELD_OUT_DAYS - 1)).isoformat()
    train = [b for b in blocks if b["cut"] < first_held]
    held = [b for b in blocks if b["cut"] >= first_held]
    return train, held


def train(args: argparse.Namespace) -> None:
    """Train the adapter on the German blocks and save, verify, publish."""
    import torch
    from chronos import Chronos2Pipeline
    from chronos.chronos2.dataset import DatasetMode
    from chronos.chronos2.model import Chronos2Model
    from peft import LoraConfig

    torch.set_num_threads(args.threads)
    out: Path = args.out
    out.mkdir(parents=True, exist_ok=True)
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
    ).stdout.strip()
    with args.blocks.open("rb") as f:
        blocks, held = split_held_out(pickle.load(f))
    log(f"{len(blocks)} training blocks, {len(held)} held out")
    fs.checkpoint_large_batches(ft.checkpoint_above(args.memory_gb))
    most = ft.max_series(args.memory_gb)
    torch.manual_seed(args.seed)
    pipeline = Chronos2Pipeline.from_pretrained(
        CHRONOS_MODEL, revision=CHRONOS_REVISION, device_map="cpu"
    )
    lora = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=[
            "self_attention.q",
            "self_attention.v",
            "self_attention.k",
            "self_attention.o",
            "output_patch_embedding.output_layer",
        ],
    )
    model = fs.lora_model(pipeline.model, lora, None)
    model.train()
    params = [p for p in model.parameters() if p.requires_grad]
    optimiser = torch.optim.AdamW(params, lr=args.learning_rate)
    accepted = inspect.signature(Chronos2Model.forward).parameters
    ids = [block_id(b) for b in blocks]
    (out / "blocks.txt").write_text("\n".join(ids) + "\n")
    data_file = Path(args.data)
    base = {
        "grid": "germany",
        "chronos_revision": CHRONOS_REVISION,
        "code_sha256": {
            **{p: ft._sha256(Path(p)) for p in ft.CODE},
            "scripts/finetune_stream.py": ft._sha256(Path(fs.__file__)),
            "scripts/finetune_germany.py": ft._sha256(Path(__file__)),
            **{
                f"foundation.{f.__name__}": hashlib.sha256(
                    inspect.getsource(f).encode()
                ).hexdigest()
                for f in ft.FOUNDATION_CODE
            },
        },
        "data": {
            "triangle_sha256": ft._sha256(data_file),
            "as_of": AS_OF,
            "truth_until": truth_until(),
            "first_cut": FIRST_CUT,
            "held_out_days": HELD_OUT_DAYS,
        },
        "blocks_sha256": ft._sha256(args.blocks),
        "blocks_list_sha256": ft._sha256(out / "blocks.txt"),
        "max_lags": {k: list(v) for k, v in ft.GERMAN_LAGS.items()},
        "n_blocks": len(blocks),
        "lora": {"r": 8, "alpha": 16},
        "learning_rate": args.learning_rate,
        "lr_steps": args.steps,
        "steps": args.steps,
        "batch_size": args.batch_size,
        "memory_gb": args.memory_gb,
        "context": ft.CONTEXT,
        "prediction_length": (
            f"each block's horizon in whole patches of {ft.OUTPUT_PATCH} steps"
        ),
        "seed": args.seed,
        "note": args.note,
    }
    sizes = np.array([int(b["context"].shape[0]) for b in blocks])
    lengths = np.array([b["prediction_length"] for b in blocks])
    data = fs.ByLength(blocks, args.batch_size, DatasetMode.TRAIN)
    consumed = out / "consumed.txt"
    state_dir = out / "state"
    state = fs.load_state(state_dir, model, optimiser)
    if state is None:
        consumed.write_text("")
        start_loss = fs.held_out_loss(model, held, args.batch_size)
        state = {
            "step": 0,
            "epoch": 0,
            "position": 0,
            "start_loss": start_loss,
            "history": [{"step": 0, "held_out_loss": start_loss}],
            "skipped": [],
        }
        log(f"held-out loss at the start {start_loss:.4f}")
    else:
        lines = consumed.read_text().splitlines()[: state["step"]]
        consumed.write_text("".join(f"{line}\n" for line in lines))
        log(f"resumed at step {state['step']}")

    def backward(part: list[int], share: float) -> float:
        batch = {
            k: v for k, v in data._build_batch(part).items() if k in accepted
        }
        loss = model(**batch).loss
        (loss * share).backward()
        value = float(loss.detach())
        del batch, loss
        return value

    def skip(i: int) -> None:
        state["skipped"].append({"block": ids[i], "step": state["step"] + 1})
        log(f"skipped {ids[i]}, {sizes[i]} series, out of memory")

    def zero() -> None:
        optimiser.zero_grad()
        fs.free_memory()

    t0 = time.perf_counter()
    plan: list = []
    while state["step"] < args.steps:
        rng = np.random.default_rng([args.seed, state["epoch"]])
        plan = ft.plan_by_length(lengths, sizes, args.batch_size, rng, most)
        while state["position"] < len(plan) and state["step"] < args.steps:
            _, indices = plan[state["position"]]
            for group in optimiser.param_groups:
                group["lr"] = fs.learning_rate(
                    state["step"], args.learning_rate, args.steps
                )
            value = fs.accumulate(list(indices), backward, zero, skip)
            if not fs.finite_step(params, optimiser, value):
                log(f"step {state['step'] + 1}: no update (loss {value})")
            state["step"] += 1
            state["position"] += 1
            with consumed.open("a") as f:
                f.write(
                    f"{state['step']}\t"
                    + ",".join(ids[i] for i in indices)
                    + "\n"
                )
            if state["step"] % LOG_EVERY == 0:
                state["history"].append({"step": state["step"], "loss": value})
                elapsed = time.perf_counter() - t0
                log(
                    f"step {state['step']}: loss {value:.4f}, "
                    f"epoch {state['epoch']}, {elapsed:.0f} s"
                )
            if state["step"] % fs.SAVE_EVERY == 0:
                fs.save_state(state_dir, model, optimiser, state)
        if state["position"] >= len(plan):
            state["epoch"] += 1
            state["position"] = 0
    fs.save_state(state_dir, model, optimiser, state)
    model.save_pretrained(out / "finetuned-ckpt")
    settings = {
        **base,
        "epochs": state["epoch"] + state["position"] / max(1, len(plan)),
        "consumed_sha256": ft._sha256(consumed),
        "skipped_blocks": state["skipped"],
    }
    key = ft._settings_hash(settings)
    settings["hash"] = key
    settings["code_commit"] = commit
    settings["device"] = "cpu"
    settings["held_out_loss_start"] = state["start_loss"]
    settings["held_out_loss"] = fs.held_out_loss(model, held, args.batch_size)
    if args.real_validation:
        with args.real_validation.open("rb") as f:
            real = pickle.load(f)
        settings["real_validation_sha256"] = ft._sha256(args.real_validation)
        settings["real_validation_loss"] = fs.held_out_loss(
            model, real, args.batch_size
        )
    settings["seconds"] = time.perf_counter() - t0
    (out / "settings.json").write_text(json.dumps(settings, indent=1))
    (out / "history.json").write_text(json.dumps(state["history"]))
    log(
        f"checkpoint {key}: held-out loss {settings['held_out_loss']:.4f} "
        f"(start {state['start_loss']:.4f})"
    )
    mismatch = finetune_verify.problems(out)
    if mismatch:
        log(f"not published: {'; '.join(mismatch)}")
        sys.exit(1)
    if args.publish:
        fs.publish(
            out,
            key,
            prefix=PREFIX,
            listing="blocks.txt",
            describe=(
                f"Chronos-2 fine-tuned on German data {key} (SI only)",
                f"LoRA adapter of Chronos-2 fine-tuned from base Chronos-2 "
                f"on {len(blocks)} blocks of the German hospitalisation "
                f"data as known on {AS_OF} (the development half), with "
                f"truth at least {TRUTH_DELAY} days old "
                f"(scripts/finetune_germany.py). For the SI comparison "
                f"only; the _ft models never load it.",
            ),
        )
    log("finished")


def main() -> None:
    """Cut the blocks or train."""
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    cut = sub.add_parser("blocks")
    cut.add_argument("triangle", type=Path)
    cut.add_argument("out", type=Path)
    cut.add_argument("--workers", type=int, default=4)
    run = sub.add_parser("train")
    run.add_argument("blocks", type=Path)
    run.add_argument("out", type=Path)
    run.add_argument(
        "--data",
        default="data/raw/germany/COVID-19_hospitalizations.csv",
    )
    run.add_argument("--steps", type=int, default=STEPS)
    run.add_argument("--learning-rate", type=float, default=1e-5)
    run.add_argument("--batch-size", type=int, default=256)
    run.add_argument("--memory-gb", type=float, default=10.0)
    run.add_argument("--threads", type=int, default=6)
    run.add_argument("--seed", type=int, default=1)
    run.add_argument("--real-validation", type=Path)
    run.add_argument("--note", default="")
    run.add_argument("--publish", action="store_true")
    args = parser.parse_args()
    if args.command == "blocks":
        make_blocks(args.triangle, args.out, args.workers)
    else:
        train(args)


if __name__ == "__main__":
    main()
