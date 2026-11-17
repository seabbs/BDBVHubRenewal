# Copyright (c) 2026 Sam Abbott. MIT licence, see LICENSE.
"""Time the parts of a fine-tuning step of `scripts/finetune_stream.py`.

Cuts the blocks of the first `--sims` simulations of the training seeds
as the trainer does, then trains `--steps` steps of their plan with
Chronos-2 and the LoRA adapter (and the series bias with
`--series-bias`), timing each part of a step with the device
synchronised: batch building, forward, backward, the optimiser step and
the memory release. It also times one held-out evaluation of
`ft.EVAL_BLOCKS` blocks and one save of the trainer's state, and writes
the times and block sizes as JSON.

    uv run --group chronos python scripts/finetune_speed.py SIM_DIR \
        --train-seeds 31000 31001 --held-out-seeds 30900 30901 \
        --sims 10 --steps 40 --series-bias --out speed.json

It changes nothing it reads; the state goes to a temporary folder.
"""

from __future__ import annotations

import argparse
import inspect
import json
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import finetune_chronos as ft
import finetune_stream as fs

from bdbvhubrenewal.forecasters import series_bias


def build_parser() -> argparse.ArgumentParser:
    """Return the command-line parser."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("sim_dir", type=Path)
    p.add_argument("--train-seeds", type=int, nargs=2, required=True)
    p.add_argument("--held-out-seeds", type=int, nargs=2, required=True)
    # Folder of the held-out batches, if not SIM_DIR.
    p.add_argument("--held-out-dir", type=Path)
    p.add_argument("--sims", type=int, default=10)
    p.add_argument("--steps", type=int, default=40)
    p.add_argument("--warmup", type=int, default=3)
    p.add_argument("--device", default="auto")
    p.add_argument("--threads", type=int)
    p.add_argument("--memory-gb", type=float, default=ft.MEMORY_GB)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--blocks-per-simulation", type=int, default=32)
    p.add_argument("--draws-per-simulation", type=int, default=1)
    p.add_argument("--subset-seed", type=int, default=1)
    p.add_argument("--sources", nargs="+", default=list(ft.SOURCES))
    p.add_argument("--univariate-share", type=float, default=0.2)
    p.add_argument("--series-bias", action="store_true")
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--out", type=Path)
    return p


def main() -> None:
    """Cut, train a few steps and report where the time goes."""
    import torch
    from chronos import Chronos2Pipeline
    from chronos.chronos2.dataset import DatasetMode
    from chronos.chronos2.model import Chronos2Model

    args = build_parser().parse_args()
    if args.threads:
        torch.set_num_threads(args.threads)
    ft.SOURCES = tuple(args.sources)
    ft.UNIVARIATE_SHARE = args.univariate_share
    fs.DRAWS_PER_PASS = args.draws_per_simulation
    fs.SUBSET_SEED = args.subset_seed
    fs.checkpoint_large_batches(ft.checkpoint_above(args.memory_gb))
    ft.MAX_SERIES = min(ft.MAX_SERIES, ft.max_series(args.memory_gb))
    k = args.blocks_per_simulation
    device = fs.pick_device(args.device)

    def sync() -> None:
        if device == "cuda":
            torch.cuda.synchronize()
        elif device == "mps":
            torch.mps.synchronize()

    seeds = range(args.train_seeds[0], args.train_seeds[1] + 1)
    ids = fs.ready_outbreaks(args.sim_dir, seeds)[: args.sims]
    held_seeds = range(args.held_out_seeds[0], args.held_out_seeds[1] + 1)
    held_dir = args.held_out_dir or args.sim_dir
    held = fs.ready_outbreaks(held_dir, held_seeds)[:10]
    tick = time.perf_counter()
    pairs = fs.cut(args.sim_dir, ids, args.seed, k)
    cut_s = time.perf_counter() - tick
    val = [b for _, b in fs.cut(held_dir, held, args.seed, k)]
    blocks = [b for _, b in pairs]
    sizes = np.array([int(b["context"].shape[0]) for b in blocks])
    lengths = np.array([b["prediction_length"] for b in blocks])
    rng = np.random.default_rng([args.seed, 0])
    plan = ft.plan_by_length(
        lengths, sizes, args.batch_size, rng, ft.MAX_SERIES
    )

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)  # noqa: NPY002
    pipeline = Chronos2Pipeline.from_pretrained(
        fs.CHRONOS_MODEL, revision=fs.CHRONOS_REVISION, device_map=device
    )
    if args.series_bias:
        series_bias.attach(pipeline.model)
    model = fs.lora_model(
        pipeline.model, fs.lora_config(args.series_bias), None
    )
    model.train()
    params = [p for p in model.parameters() if p.requires_grad]
    optimiser = fs.make_optimiser(model, 1e-5, 1e-3)
    accepted = inspect.signature(Chronos2Model.forward).parameters
    tick = time.perf_counter()
    data = fs.ByLength(
        blocks, args.batch_size, DatasetMode.TRAIN, args.series_bias
    )
    dataset_s = time.perf_counter() - tick

    parts = ("build", "forward", "backward", "optimiser", "free")
    steps = []
    for position, (_, indices) in enumerate(plan[: args.steps]):
        times = {}
        sync()
        tick = time.perf_counter()
        batch = data._build_batch(list(indices))
        times["build"] = time.perf_counter() - tick
        tick = time.perf_counter()
        with series_bias.assigned(model, batch.get("series_rows")):
            loss = fs.forward(model, batch, accepted, device).loss
            sync()
            times["forward"] = time.perf_counter() - tick
            tick = time.perf_counter()
            loss.backward()
            sync()
            times["backward"] = time.perf_counter() - tick
        value = float(loss.detach())
        del batch, loss
        tick = time.perf_counter()
        fs.finite_step(params, optimiser, value)
        sync()
        times["optimiser"] = time.perf_counter() - tick
        tick = time.perf_counter()
        fs.free_memory()
        sync()
        times["free"] = time.perf_counter() - tick
        times["total"] = sum(times[p] for p in parts)
        times["series"] = int(sizes[indices].sum())
        times["blocks"] = len(indices)
        times["length"] = int(lengths[indices[0]])
        times["loss"] = value
        steps.append(times)
        fs.log(
            f"step {position + 1}: {times['series']} series in "
            f"{times['blocks']} blocks, length {times['length']}: "
            + ", ".join(f"{p} {times[p]:.3f}" for p in (*parts, "total"))
        )

    sync()
    tick = time.perf_counter()
    fs.held_out_loss(model, val[: ft.EVAL_BLOCKS], args.batch_size)
    sync()
    eval_s = time.perf_counter() - tick
    with tempfile.TemporaryDirectory() as tmp:
        tick = time.perf_counter()
        fs.save_state(Path(tmp) / "state", model, optimiser, {"step": 0})
        save_s = time.perf_counter() - tick

    timed = steps[args.warmup :]
    mean = {p: float(np.mean([s[p] for s in timed])) for p in (*parts, "total")}
    per_sim = len(plan) / len(ids)
    summary = {
        "device": device,
        "threads": torch.get_num_threads(),
        "memory_gb": args.memory_gb,
        "max_series": ft.MAX_SERIES,
        "checkpoint_above": ft.checkpoint_above(args.memory_gb),
        "series_bias": args.series_bias,
        "sims": len(ids),
        "blocks": len(blocks),
        "steps_planned": len(plan),
        "steps_per_simulation": per_sim,
        "block_series": {
            "mean": float(sizes.mean()),
            "median": float(np.median(sizes)),
            "max": int(sizes.max()),
            "quantiles_10_50_90": np.quantile(sizes, [0.1, 0.5, 0.9]).tolist(),
        },
        "cut_s_per_simulation": cut_s / len(ids),
        "dataset_s": dataset_s,
        "step_s": mean,
        "s_per_simulation": mean["total"] * per_sim,
        "eval_s": eval_s,
        "eval_blocks": min(ft.EVAL_BLOCKS, len(val)),
        "save_s": save_s,
        "steps": steps,
    }
    fs.log(
        f"{device}: {mean['total']:.3f} s a step "
        f"({', '.join(f'{p} {mean[p]:.3f}' for p in parts)}), "
        f"{per_sim:.1f} steps and {summary['s_per_simulation']:.1f} s a "
        f"simulation; held-out eval {eval_s:.1f} s, save {save_s:.2f} s"
    )
    if args.out:
        args.out.write_text(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
