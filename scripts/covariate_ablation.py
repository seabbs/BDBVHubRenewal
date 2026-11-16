# Copyright (c) 2026 Sam Abbott. MIT licence, see LICENSE.
r"""Score Chronos-2 with the lag covariates given, removed or shuffled.

Blocks are held-out simulations cut as the trainer cuts them
(`finetune_stream.cut`, seed 1, two draws of each of 16 snapshots) and
the BDBV pseudo-origins (`finetune_real_validation`). Each block is
scored under each condition with the zero-shot model and a fine-tuned
adapter. The loss is Chronos-2's scaled quantile loss (twice the pinball
loss, summed over its quantile levels, on the target scaled by its own
context) on each target row and step, averaged over the nowcast steps
(the release lag's D steps to the cut-off) and the forecast steps (after
it) separately, then over the block's targets.

Conditions:
  given            the block as trained.
  removed          no covariate rows (target only, same cut).
  swap_places      each place's whole lag series moved to another place.
  shuffle_places   at each step, the places' lags assigned at random.
  shuffle_time     history columns of the lag rows permuted jointly.
  shuffle_time_all history and horizon columns permuted jointly.
  no_horizon       lag rows NaN over the horizon (history kept).

    python scripts/covariate_ablation.py SIM_DIR HELD_SEEDS DELAYS ADAPTER \\
        OUT.csv [--sims N] [--per-source N] [--max-rows N]
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import finetune_chronos as ft
import finetune_real_validation as frv
import finetune_stream as fs

CONDITIONS = (
    "given",
    "removed",
    "swap_places",
    "shuffle_places",
    "shuffle_time",
    "shuffle_time_all",
    "no_horizon",
)


def sim_blocks(sim_dir: Path, seeds: list[int], n_sims: int) -> list[dict]:
    """Cut held-out simulations' blocks as the trainer does."""
    fs.init_worker(
        min(ft.MAX_SERIES, ft.max_series(ft.MEMORY_GB)),
        ft.MAX_LAGS,
        ("weekly", "daily", "3day"),
        0.2,
        2,
        1,
    )
    ids = fs.ready_outbreaks(sim_dir, range(seeds[0], seeds[1] + 1))
    ids = ids[:n_sims]
    out = []
    for sim in ids:
        # The cut and horizon of each block, recorded as it is padded.
        meta = []
        orig_pad = ft._pad

        def pad(block, context, meta=meta, orig_pad=orig_pad):
            meta.append((block["start"], block["horizon"]))
            return orig_pad(block, context)

        ft._pad = pad
        try:
            pairs = fs.cut(sim_dir, [sim], 1, 32)
        finally:
            ft._pad = orig_pad
        for (outbreak, block), (start, h) in zip(pairs, meta, strict=True):
            out.append({**block, "start": start, "horizon": h, "id": outbreak})
    return out


def real_blocks(delays: Path) -> list[dict]:
    """Return the BDBV pseudo-origin blocks with their start and horizon."""
    d = pd.read_csv(delays)
    line_list = d[d["vintage"] == frv.VINTAGE]
    metas = []
    orig_pad = ft._pad

    def pad(block, context):
        metas.append((block["start"], block["horizon"]))
        return orig_pad(block, context)

    ft._pad = pad
    try:
        blocks = ft.real_blocks(line_list, frv.CUTS, frv.LAST_WEEK, frv.LAG)
    finally:
        ft._pad = orig_pad
    return [
        {**b, "start": s, "horizon": h, "id": f"bdbv-{c}"}
        for b, (s, h), c in zip(blocks, metas, frv.CUTS, strict=True)
    ]


def lag_of(block: dict) -> int:
    """Return the block's release lag D."""
    n_targets = block["n_targets"]
    return block["n_covariates"] // n_targets


def condition(block: dict, name: str, rng: np.random.Generator) -> dict | None:
    """Return the block under a condition, or None where it does not apply."""
    x = block["context"].copy()
    n, d = block["n_targets"], lag_of(block)
    c, h = ft.CONTEXT, block["horizon"]
    first = c - min(block["start"], c)
    cov = slice(n, n + n * d)
    if name == "given":
        pass
    elif name == "removed":
        x = x[:n]
    elif name in ("swap_places", "shuffle_places"):
        if n < 2:
            return None
        rows = x[cov].reshape(n, d, -1)
        if name == "swap_places":
            # A derangement of whole series: every place takes another's
            # lags. Chronos-2's group attention has no variate identity,
            # so this should leave the output unchanged.
            while True:
                perm = rng.permutation(n)
                if (perm != np.arange(n)).all():
                    break
            rows = rows[perm]
        else:
            # Each step's lags go to a random place, drawn afresh at every
            # step, so a lag series mixes places over time.
            for t in range(rows.shape[2]):
                rows[:, :, t] = rows[rng.permutation(n), :, t]
        x[cov] = rows.reshape(n * d, -1)
    elif name == "shuffle_time":
        cols = np.arange(first, c)
        sub = x[cov]
        sub[:, cols] = block["context"][cov][:, rng.permutation(cols)]
        x[cov] = sub
    elif name == "shuffle_time_all":
        cols = np.arange(first, c + h)
        sub = x[cov]
        sub[:, cols] = block["context"][cov][:, rng.permutation(cols)]
        x[cov] = sub
    elif name == "no_horizon":
        x[cov, c:] = np.nan
    else:
        raise ValueError(name)
    k = 0 if name == "removed" else n * d
    return {
        "context": x,
        "future_covariates": x[:, c:],
        "n_targets": n,
        "n_covariates": k,
        "n_future_covariates": k,
        "prediction_length": block["prediction_length"],
    }


def losses(model, block: dict):
    """Return the loss of each target row and step, `[n, P]`."""
    import torch
    from chronos.chronos2.dataset import Chronos2Dataset, DatasetMode

    data = Chronos2Dataset(
        inputs=[fs.tensors(block)],
        context_length=ft.CONTEXT,
        prediction_length=block["prediction_length"],
        batch_size=10**6,
        output_patch_size=ft.OUTPUT_PATCH,
        mode=DatasetMode.VALIDATION,
    )
    batch = next(iter(data))
    n = block["n_targets"]
    target = batch["future_target"][:n].clone()
    batch = {**batch, "future_target": None}
    with torch.no_grad():
        out = model(
            context=batch["context"],
            group_ids=batch["group_ids"],
            future_covariates=batch["future_covariates"],
            num_output_patches=batch["num_output_patches"],
        )
    preds = out.quantile_preds[:n].float()  # [n, Q, P]
    norm = model.instance_norm
    ctx = batch["context"][:n].float()
    _, loc_scale = norm(ctx)
    t, _ = norm(target, loc_scale)
    p, _ = norm(preds, (loc_scale[0][:, :, None], loc_scale[1][:, :, None]))
    q = model.quantiles.float()[None, :, None]
    t = t[:, None, :]
    ql = 2 * torch.abs((t - p) * ((t <= p).float() - q))
    return ql.sum(dim=1)  # [n, P]


def main() -> None:
    """Score every block under every condition and model."""
    import torch
    from chronos import Chronos2Pipeline
    from peft import PeftModel

    from bdbvhubrenewal.forecasters.foundation import (
        CHRONOS_MODEL,
        CHRONOS_REVISION,
    )

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sim_dir", type=Path)
    parser.add_argument("held", type=int, nargs=2)
    parser.add_argument("delays", type=Path)
    parser.add_argument("adapter", type=Path)
    parser.add_argument("out", type=Path)
    parser.add_argument("--sims", type=int, default=40)
    parser.add_argument("--per-source", type=int, default=40)
    parser.add_argument("--max-rows", type=int, default=300)
    parser.add_argument("--label", default="covab")
    parser.add_argument("--shard", type=int, nargs=2, default=(0, 1))
    args = parser.parse_args()
    torch.set_num_threads(1)
    blocks = real_blocks(args.delays)
    for b in blocks:
        b["set"] = "bdbv"
    sims = sim_blocks(args.sim_dir, args.held, args.sims)
    rng = np.random.default_rng(1)
    for source in ("weekly", "daily", "3day"):
        mine = [
            b
            for b in sims
            if b["source"] == source and b["context"].shape[0] <= args.max_rows
        ]
        take = rng.choice(
            len(mine), min(args.per_source, len(mine)), replace=False
        )
        for i in sorted(take):
            blocks.append({**mine[i], "set": "sim"})
    i, k = args.shard
    blocks = blocks[i::k]
    print(
        f"{len(blocks)} blocks of {len(sims)} cut; "
        f"rows {[b['context'].shape[0] for b in blocks]}",
        flush=True,
    )
    pipeline = Chronos2Pipeline.from_pretrained(
        CHRONOS_MODEL, revision=CHRONOS_REVISION, device_map="cpu"
    )
    base = pipeline.model.eval()
    tuned = (
        PeftModel.from_pretrained(
            Chronos2Pipeline.from_pretrained(
                CHRONOS_MODEL, revision=CHRONOS_REVISION, device_map="cpu"
            ).model,
            args.adapter,
        )
        .merge_and_unload()
        .eval()
    )
    rows = []
    t0 = time.time()
    for i, block in enumerate(blocks):
        d = block["n_covariates"] // block["n_targets"]
        h = block["horizon"]
        for cond in CONDITIONS:
            crng = np.random.default_rng([i, CONDITIONS.index(cond)])
            x = condition(block, cond, crng)
            if x is None:
                continue
            for name, model in (("zero-shot", base), ("n1000", tuned)):
                per_step = losses(model, x).numpy()
                rows.append(
                    {
                        "block": f"{args.shard[0]}-{i}",
                        "id": block["id"],
                        "set": block["set"],
                        "source": block.get("source", "weekly"),
                        "mix": block.get("mix", ""),
                        "n_targets": block["n_targets"],
                        "lag": d,
                        "horizon": h,
                        "model": name,
                        "condition": cond,
                        "nowcast": float(per_step[:, :d].mean()),
                        "forecast": float(per_step[:, d:h].mean()),
                        "all": float(per_step[:, :h].mean()),
                    }
                )
        if i % 10 == 0:
            print(
                f"{args.label} block {i + 1}/{len(blocks)} "
                f"{time.time() - t0:.0f}s",
                flush=True,
            )
            pd.DataFrame(rows).to_csv(args.out, index=False)
    pd.DataFrame(rows).to_csv(args.out, index=False)
    print(f"{args.label} done {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
