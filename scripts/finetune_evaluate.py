# Copyright (c) 2026 Sam Abbott. MIT licence, see LICENSE.
r"""Score an adapter snapshot of `scripts/finetune_stream.py --eval offload`.

The trainer leaves each evaluation to this script: at step 0 and every
`--eval-every` steps it saves the adapter and a request
(`<train>/evals/step-<k>/`), and at each checkpoint its adapter
(`<train>/n<N>/finetuned-ckpt`). This script loads base Chronos-2 with
that adapter and scores the held-out blocks the trainer saved
(`<train>/held-out-blocks.pkl`) with the trainer's own functions, so
the losses are the ones the trainer would have logged, on whatever
device runs it.

    uv run --group chronos python scripts/finetune_evaluate.py \
        TRAIN/evals/step-0000100 --train TRAIN --out result.json

Kinds, from the request or `--kinds`:

- `start`: all held-out blocks, by level and by source (step 0);
- `monitor`: the first `ft.EVAL_BLOCKS` blocks (`eval_loss`);
- `by_source`: each source's first `ft.EVAL_BLOCKS` blocks;
- `checkpoint`: all blocks, by level, by source and the real validation
  blocks, as the trainer adds them to a checkpoint's settings.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).parent))
import finetune_chronos as ft
import finetune_stream as fs

from bdbvhubrenewal.forecasters import series_bias
from bdbvhubrenewal.forecasters.foundation import torch_device

KINDS = ("start", "monitor", "by_source", "checkpoint")


def load_blocks(path: Path, n: int | None = None) -> list[dict]:
    """Return the trainer's held-out blocks, checking their count."""
    blocks = fs.load_blocks(Path(path))
    if n is not None and len(blocks) != n:
        raise ValueError(
            f"{path} holds {len(blocks)} held-out blocks, the run {n}"
        )
    return blocks


def evaluate(
    adapter: Path,
    base,
    blocks: Sequence[dict],
    kinds: Sequence[str],
    batch_size: int,
    real: Sequence[dict] = (),
) -> dict[str, Any]:
    """Return the losses `kinds` ask for of `base` with `adapter`.

    `base` is a base Chronos-2 model, with the series bias attached if
    the adapter holds one.
    """
    from peft import PeftModel

    unknown = set(kinds) - set(KINDS)
    if unknown:
        raise ValueError(f"unknown kinds: {sorted(unknown)}")
    model = PeftModel.from_pretrained(base, adapter)
    model.eval()
    val = list(blocks)
    out: dict[str, Any] = {}
    if "start" in kinds or "checkpoint" in kinds:
        parts = fs.held_out_parts(model, val, batch_size)
        out["held_out_loss"] = parts["loss"]
        out |= fs.loss_parts("held_out_loss", parts)
    if "start" in kinds:
        out["by_level"] = fs.held_out_by_level(model, val, fs.LEVEL_BLOCKS)
        out["by_source"] = fs.held_out_by_source(
            model, val, batch_size, ft.EVAL_BLOCKS
        )
    if "monitor" in kinds:
        parts = fs.held_out_parts(model, val[: ft.EVAL_BLOCKS], batch_size)
        out["eval_loss"] = parts["loss"]
        out |= fs.loss_parts("eval_loss", parts)
    if "by_source" in kinds:
        out["by_source"] = fs.held_out_by_source(
            model, val, batch_size, ft.EVAL_BLOCKS
        )
    if "checkpoint" in kinds:
        if real:
            parts = fs.held_out_parts(model, list(real), batch_size)
            out["real_validation_loss"] = parts["loss"]
            out |= fs.loss_parts("real_validation_loss", parts)
        out["held_out_loss_by_level"] = fs.held_out_by_level(
            model, val, fs.LEVEL_BLOCKS
        )
        out["held_out_loss_by_source"] = fs.held_out_by_source(
            model, val, batch_size
        )
    return out


def build_parser() -> argparse.ArgumentParser:
    """Return the command-line parser."""
    p = argparse.ArgumentParser(description=__doc__)
    # A snapshot folder (adapter/ and request.json) or a checkpoint's
    # finetuned-ckpt folder.
    p.add_argument("snapshot", type=Path)
    p.add_argument("--train", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--kinds", nargs="+", choices=KINDS)
    p.add_argument("--real-validation", type=Path)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--device", default="auto")
    return p


def main() -> None:
    """Score a snapshot and write its losses as JSON."""
    from chronos import Chronos2Pipeline

    args = build_parser().parse_args()
    snapshot = args.snapshot
    request_file = snapshot / "request.json"
    request = (
        json.loads(request_file.read_text()) if request_file.exists() else {}
    )
    adapter = snapshot / "adapter" if request else snapshot
    kinds = args.kinds or request.get("kinds")
    if not kinds:
        raise SystemExit(f"no kinds asked for {snapshot}")
    batch_size = request.get("batch_size", args.batch_size)
    blocks = load_blocks(
        args.train / fs.HELD_OUT_BLOCKS, request.get("n_validation")
    )
    real = fs.load_blocks(args.real_validation) if args.real_validation else []
    device = torch_device(args.device)
    pipeline = Chronos2Pipeline.from_pretrained(
        fs.CHRONOS_MODEL, revision=fs.CHRONOS_REVISION, device_map=device
    )
    if series_bias.adapter_has_bias(adapter):
        series_bias.attach(pipeline.model, series_bias.adapter_pairs(adapter))
    tick = time.perf_counter()
    losses = evaluate(adapter, pipeline.model, blocks, kinds, batch_size, real)
    result = {
        **{k: request[k] for k in ("step", "n_outbreaks") if k in request},
        **losses,
        "kinds": list(kinds),
        "device": device,
        "seconds": time.perf_counter() - tick,
    }
    tmp = args.out.with_name(args.out.name + ".tmp")
    tmp.write_text(json.dumps(result, indent=1))
    tmp.rename(args.out)
    fs.log(f"{snapshot.name}: {', '.join(kinds)} in {result['seconds']:.0f} s")


if __name__ == "__main__":
    main()
