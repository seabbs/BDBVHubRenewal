# Copyright (c) 2026 Sam Abbott. MIT licence, see LICENSE.
"""Held-out interval coverage and upper tails of each checkpoint, on CPU.

Scores zero-shot Chronos-2 and the checkpoints n1000, n2000 and n4000 on
a seeded sample of held-out blocks, optionally with a floor on the scale
Chronos-2 divides each series by, and writes one row per target series:
coverage of the 50%, 90% and 1-99% intervals, the 99% quantile over the
truth (median and largest over the horizon, both plus one), and the
quantile loss over the series' mean plus one. Chronos-2 has no 2.5% or
97.5% quantile, so the widest interval is 1-99%. One thread.

    python scripts/finetune_tails.py SIM_DIR CKPT_DIR N_BLOCKS FLOOR OUT
"""

from __future__ import annotations

import inspect
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).parent))
import finetune_chronos as ft
import finetune_stream as fs

from bdbvhubrenewal.forecasters.foundation import (
    CHRONOS_MODEL,
    CHRONOS_REVISION,
    floor_scale,
)

MODELS = ("zero-shot", "n1000", "n2000", "n4000")


def series_rows(model, blocks: list[dict], qs: np.ndarray) -> list[dict]:
    """Return one row of scores per target series of each block."""
    from chronos.chronos2.dataset import Chronos2Dataset, DatasetMode
    from chronos.chronos2.model import Chronos2Model

    accepted = inspect.signature(Chronos2Model.forward).parameters
    at = {q: int(np.argmin(np.abs(qs - q))) for q in (0.01, 0.05, 0.25)}
    at |= {1 - q: len(qs) - 1 - i for q, i in at.items()}
    rows = []
    model.eval()
    with torch.no_grad():
        for b, block in enumerate(blocks):
            batch = next(
                iter(
                    Chronos2Dataset(
                        inputs=[fs.tensors(block)],
                        context_length=ft.CONTEXT,
                        prediction_length=ft.HORIZON,
                        batch_size=1,
                        output_patch_size=ft.OUTPUT_PATCH,
                        mode=DatasetMode.VALIDATION,
                    )
                )
            )
            inputs = {
                k: v
                for k, v in batch.items()
                if k in accepted and k != "future_target"
            }
            pred = model(**inputs).quantile_preds.float().numpy()
            truth = batch["future_target"].float().numpy()
            for r, level in enumerate(block["levels"]):
                ok = ~np.isnan(truth[r])
                if not ok.any():
                    continue
                y = truth[r][ok]
                p = pred[r][:, : len(truth[r])][:, ok]

                def inside(lo: float, hi: float, p=p, y=y) -> float:
                    return float(np.mean((y >= p[at[lo]]) & (y <= p[at[hi]])))

                loss = sum(
                    np.mean(2 * np.abs((y - p[i]) * ((y <= p[i]) - q)))
                    for i, q in enumerate(qs)
                )
                ratio = (p[at[0.99]] + 1) / (y + 1)
                rows.append(
                    {
                        "block": b,
                        "level": level,
                        "mean": float(y.mean()),
                        "coverage_50": inside(0.25, 0.75),
                        "coverage_90": inside(0.05, 0.95),
                        "coverage_98": inside(0.01, 0.99),
                        "q99_over_truth": float(np.median(ratio)),
                        "q99_over_truth_max": float(np.max(ratio)),
                        "loss": float(loss / (y.mean() + 1)),
                    }
                )
    return rows


def main() -> None:
    """Score the checkpoints in `sys.argv[2]` on blocks of `sys.argv[1]`."""
    from chronos import Chronos2Pipeline
    from peft import PeftModel

    sim, ckpt = Path(sys.argv[1]), Path(sys.argv[2])
    n_blocks, floor, out = int(sys.argv[3]), float(sys.argv[4]), sys.argv[5]
    torch.set_num_threads(1)
    ft.MAX_SERIES = min(ft.MAX_SERIES, ft.max_series(ft.MEMORY_GB))
    held = (ckpt / "outbreaks-held-out.txt").read_text().split()
    blocks = [b for _, b in fs.cut(sim, held, 1, 16)]
    pick = np.random.default_rng(235).choice(len(blocks), n_blocks, False)
    blocks = [blocks[i] for i in sorted(pick)]
    base = Chronos2Pipeline.from_pretrained(
        CHRONOS_MODEL, revision=CHRONOS_REVISION, device_map="cpu"
    ).model
    if floor > 0:
        floor_scale(base, floor)
    qs = np.array(base.chronos_config.quantiles)
    frames = []
    for name in MODELS:
        model = (
            base
            if name == "zero-shot"
            else PeftModel.from_pretrained(base, ckpt / name / "finetuned-ckpt")
        )
        frames.append(
            pd.DataFrame(series_rows(model, blocks, qs)).assign(model=name)
        )
        if name != "zero-shot":
            base = model.unload()
    res = pd.concat(frames)
    res["size"] = pd.cut(
        res["mean"],
        [-1, 5, 50, np.inf],
        labels=["small (<=5)", "medium (5-50)", "large (>50)"],
    )
    res.assign(floor=floor).to_csv(out, index=False)


if __name__ == "__main__":
    main()
