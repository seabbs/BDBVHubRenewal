"""Score an offloaded run's snapshots and checkpoints (rules/evaluate.smk).

    python workflow/scripts/evaluate_offloaded.py steps TRAIN
    python workflow/scripts/evaluate_offloaded.py checkpoint TRAIN/n<N>

`steps` scores every adapter snapshot under TRAIN/evals that has no
result yet (`flowlib.evaluations.pending`) and writes each one's losses to
`TRAIN/evals/step-<k>.json`. `checkpoint` scores a checkpoint's adapter as
the trainer would have at that checkpoint and writes its sidecar,
`n<N>/held-out.json`, with base Chronos-2's held-out loss
(`flowlib.evaluations.start_loss`). Both use the trainer's own loss
functions through scripts/finetune_evaluate.py, on the trainer's saved
held-out blocks, so the losses are those `--eval local` would log. Base
Chronos-2 is loaded once and copied for each adapter.
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

ROOT = Path(__file__).parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "workflow"))

import finetune_evaluate as fe  # noqa: E402
import finetune_stream as fs  # noqa: E402
from flowlib import evaluations  # noqa: E402

from bdbvhubrenewal.forecasters import series_bias  # noqa: E402
from bdbvhubrenewal.forecasters.foundation import torch_device  # noqa: E402


def fresh(base, device: str):
    """Return a copy of base Chronos-2 on `device`, for one adapter."""
    return copy.deepcopy(base).to(device)


def write(path: Path, data: dict[str, Any]) -> None:
    """Write JSON beside `path` and move it into place."""
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(data, indent=1))
    tmp.rename(path)


def score(base, adapter: Path, blocks, kinds, batch_size, real, device):
    """Return the losses `kinds` ask for of `adapter`, and the seconds."""
    model = fresh(base, device)
    series_bias.attach_for(model, adapter)
    tick = time.perf_counter()
    losses = fe.evaluate(adapter, model, blocks, kinds, batch_size, real)
    del model
    fs.free_memory()
    return losses, time.perf_counter() - tick


def steps(train: Path, device: str, real: Sequence[dict]) -> None:
    """Score each step snapshot with no result yet."""
    todo = evaluations.pending(train)
    if not todo:
        fs.log(f"{train}: no step snapshots to score")
        return
    base = fe.load_base("cpu")
    blocks: dict[Any, list[dict]] = {}
    for snap in todo:
        request = json.loads((snap / "request.json").read_text())
        n = request.get("n_validation")
        if n not in blocks:
            blocks[n] = fe.load_blocks(train / fs.HELD_OUT_BLOCKS, n)
        kinds = request["kinds"]
        losses, seconds = score(
            base,
            snap / "adapter",
            blocks[n],
            kinds,
            request.get("batch_size", 256),
            real,
            device,
        )
        result = {
            **{k: request[k] for k in ("step", "n_outbreaks") if k in request},
            **losses,
            "kinds": list(kinds),
            "device": device,
            "seconds": seconds,
        }
        write(snap.with_name(snap.name + ".json"), result)
        fs.log(f"{snap.name}: {', '.join(kinds)} in {seconds:.0f} s")


def checkpoint(where: Path, device: str, real: Sequence[dict]) -> None:
    """Score a checkpoint's adapter and write its held-out sidecar."""
    train = where.parent
    settings = json.loads((where / "settings.json").read_text())
    blocks = fe.load_blocks(
        train / fs.HELD_OUT_BLOCKS, settings.get("n_validation")
    )
    losses, seconds = score(
        fe.load_base("cpu"),
        where / "finetuned-ckpt",
        blocks,
        ["checkpoint"],
        settings.get("batch_size", 256),
        real,
        device,
    )
    start = evaluations.start_loss(train)
    sidecar = {
        **({"held_out_loss_start": start} if start is not None else {}),
        **losses,
        "device": device,
        "seconds": seconds,
    }
    write(where / evaluations.SIDECAR, sidecar)
    fs.log(f"{where.name}: held-out loss {losses['held_out_loss']:.4f}")


def main(argv: Sequence[str] | None = None) -> None:
    """Run `steps` or `checkpoint`."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("what", choices=("steps", "checkpoint"))
    p.add_argument("where", type=Path)
    p.add_argument("--device", default="auto")
    p.add_argument("--real-validation", type=Path)
    args = p.parse_args(argv)
    device = torch_device(args.device)
    real = fs.load_blocks(args.real_validation) if args.real_validation else []
    if args.what == "steps":
        steps(args.where, device, real)
    else:
        checkpoint(args.where, device, real)


if __name__ == "__main__":
    main()
