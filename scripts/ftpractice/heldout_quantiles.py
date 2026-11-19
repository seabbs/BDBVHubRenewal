"""Quantiles of Chronos-2 checkpoints on held-out simulated blocks.

Cuts blocks from the held-out round's simulations with the trainer's own
code (`finetune_stream.cut`, grid r6 design), then for each checkpoint
writes every target step's quantiles and final count, with the block's
simulation, source and step relative to the cut-off, so widths can be
joined to the simulation's grid factors.

    python scripts/ftpractice/heldout_quantiles.py SIM_DIR OUT_DIR \
        --seeds 32000 32039 --per-simulation 8 \
        --checkpoint zero-shot \
        --checkpoint r6-n500=~/.cache/bdbvhubrenewal/chronos-slots-6bb954c7ffe7

A checkpoint is `zero-shot` or `NAME=ADAPTER_DIR`. Blocks are cut once and
cached in OUT_DIR/blocks.pkl; each checkpoint's rows go to
OUT_DIR/<name>.csv.gz and are skipped if present. Before each checkpoint
it waits while a checkpoint-scoring unit runs, so scoring has the GPU.
"""

from __future__ import annotations

import argparse
import inspect
import pickle
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parents[1]))
import finetune_chronos as ft
import finetune_stream as fs

from bdbvhubrenewal.forecasters import series_bias
from bdbvhubrenewal.forecasters.foundation import (
    CHRONOS_MODEL,
    CHRONOS_REVISION,
    torch_device,
)

R6_SOURCES = (
    "daily",
    "weekly",
    "3day",
    "daily+weekly",
    "daily+rolling7",
    "daily+3day",
    "daily+rolling3",
)
LEVELS = (0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95)
SCORING = "bdbvhubrenewal.evaluation.checkpoint"


def log(msg: str) -> None:
    """Print a timestamped line."""
    print(f"{time.strftime('%H:%M:%S')} {msg}", flush=True)


def wait_for_scoring() -> None:
    """Wait while a checkpoint-scoring unit is running (scoring first)."""
    while True:
        busy = subprocess.run(
            ["pgrep", "-f", SCORING], capture_output=True, text=True
        ).stdout.split()
        if not busy:
            return
        log(f"scoring running ({len(busy)} procs); waiting")
        time.sleep(60)


def blocks(sim_dir: Path, seeds: range, k: int, memory_gb: float, out: Path):
    """Return the held-out (simulation, block) pairs, cut once and cached."""
    cache = out / "blocks.pkl"
    if cache.exists():
        with cache.open("rb") as f:
            return pickle.load(f)
    ft.use_design("r6")
    ft.SOURCES = R6_SOURCES
    ft.UNIVARIATE_SHARE = 0.2
    fs.DRAWS_PER_PASS, fs.SUBSET_SEED = 1, 1
    ft.MAX_SERIES = min(ft.MAX_SERIES, ft.max_series(memory_gb))
    ids = fs.ready_outbreaks(sim_dir, seeds)
    log(f"cutting {len(ids)} simulations, {k} blocks each")
    pairs = fs.cut(sim_dir, ids, 1, k, [])
    with cache.open("wb") as f:
        pickle.dump(pairs, f)
    return pairs


def load(spec: str, device: str):
    """Return Chronos-2, zero-shot or with an adapter merged."""
    from chronos import Chronos2Pipeline

    pipeline = Chronos2Pipeline.from_pretrained(
        CHRONOS_MODEL, revision=CHRONOS_REVISION, device_map=device
    )
    model = pipeline.model
    if spec != "zero-shot":
        model = series_bias.load_adapter(model, Path(spec).expanduser())
    return model.to(device).eval()


def predict(model, pairs, batch_size: int) -> pd.DataFrame:
    """Return each target step's quantiles and final count."""
    import torch
    from chronos.chronos2.dataset import DatasetMode
    from chronos.chronos2.model import Chronos2Model

    blks = [b for _, b in pairs]
    data = fs.ByLength(
        blks, batch_size, DatasetMode.VALIDATION, series_bias.has_bias(model)
    )
    accepted = inspect.signature(Chronos2Model.forward).parameters
    q = list(model.chronos_config.quantiles)
    cols = [q.index(x) for x in LEVELS]
    rows, skipped = [], 0
    with torch.no_grad():
        for indices in data.indices():
            batch = data._build_batch(indices)
            try:
                out = fs.forward(model, batch, accepted)
            except torch.OutOfMemoryError:
                skipped += len(indices)
                torch.cuda.empty_cache()
                continue
            preds = out.quantile_preds.float().cpu().numpy()
            target = batch["future_target"].float().cpu().numpy()
            for i, (lo, hi) in zip(
                indices, batch["target_idx_ranges"], strict=True
            ):
                sim, block = pairs[i]
                d = fs.nowcast_steps(block)
                n_steps = int(block["prediction_length"])
                levels = block.get("levels", ())
                for r in range(lo, hi):
                    y = target[r, :n_steps]
                    ok = ~np.isnan(y)
                    steps = np.arange(n_steps)[ok]
                    rec = {
                        "simulation": "-".join(sim.split("-")[:2]),
                        "block": i,
                        "source": block.get("source", ""),
                        "level": levels[r - lo] if r - lo < len(levels) else "",
                        "row": r - lo,
                        "step": steps - d,
                        "final": y[ok],
                    }
                    for c, x in zip(cols, LEVELS, strict=True):
                        rec[f"q{x}"] = preds[r, c, :n_steps][ok]
                    rows.append(pd.DataFrame(rec))
    if skipped:
        log(f"skipped {skipped} blocks out of memory")
    return pd.concat(rows, ignore_index=True)


def main() -> None:
    """Cut the blocks, then write each checkpoint's rows."""
    p = argparse.ArgumentParser()
    p.add_argument("sim_dir", type=Path)
    p.add_argument("out", type=Path)
    p.add_argument("--seeds", type=int, nargs=2, required=True)
    p.add_argument("--per-simulation", type=int, default=8)
    p.add_argument("--memory-gb", type=float, default=5.0)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--checkpoint", action="append", required=True)
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    pairs = blocks(
        args.sim_dir,
        range(args.seeds[0], args.seeds[1] + 1),
        args.per_simulation,
        args.memory_gb,
        args.out,
    )
    log(f"{len(pairs)} blocks")
    device = torch_device()
    for item in args.checkpoint:
        name, _, spec = item.partition("=")
        spec = spec or name
        path = args.out / f"{name}.csv.gz"
        if path.exists():
            log(f"{name} done already")
            continue
        wait_for_scoring()
        start = time.monotonic()
        model = load(spec, device)
        rows = predict(model, pairs, args.batch_size)
        rows.insert(0, "checkpoint", name)
        tmp = path.with_name(path.name + ".tmp")
        rows.to_csv(tmp, index=False, compression="gzip")
        tmp.rename(path)
        log(f"{name}: {len(rows)} rows, {time.monotonic() - start:.0f} s")
        del model
        fs.free_memory()


if __name__ == "__main__":
    main()
