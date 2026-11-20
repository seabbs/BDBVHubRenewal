"""Score an offloaded run's snapshots and checkpoints (rules/evaluate.smk).

    python workflow/scripts/evaluate_offloaded.py steps TRAIN/n<N> \
        --eval-every K [--prev TRAIN/n<M>]
    python workflow/scripts/evaluate_offloaded.py checkpoint TRAIN/n<N>

`steps` scores segment n<N>'s step snapshots, `checkpoint` its adapter, with
the trainer's own losses (scripts/finetune_evaluate.py); the files are as
`flowlib.evaluations` says.
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

from nowcastfoundation import series_bias  # noqa: E402
from nowcastfoundation.backends.chronos2 import torch_device  # noqa: E402


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


def missing(train: Path, after: int, upto: int, every: int) -> list[str]:
    """Return the segment's due steps with neither snapshot nor result.

    Each step after `after` up to `upto` that is a multiple of `every` is
    due.
    """
    evals = train / "evals"

    def found(step: int) -> bool:
        name = f"step-{step:07d}"
        return (evals / name / "request.json").exists() or (
            evals / f"{name}.json"
        ).exists()

    due = range((after // every + 1) * every, upto + 1, every)
    return [f"step-{k:07d}" for k in due if not found(k)]


def steps(
    where: Path,
    prev: Path | None,
    device: str,
    real,
    every: int,
) -> None:
    """Score segment `where`'s step snapshots that have no result yet.

    The segment's steps are those after checkpoint `prev`'s (from step 0
    for the first) up to its own. A due step with neither snapshot nor
    result fails the job: its snapshots were not copied in, and scoring
    nothing must not mark the segment done.
    """
    train = where.parent
    upto = json.loads((where / "settings.json").read_text())["steps"]
    after = (
        json.loads((prev / "settings.json").read_text())["steps"]
        if prev is not None
        else -1
    )
    gone = missing(train, after, upto, every)
    if gone:
        raise SystemExit(
            f"{where}: no step snapshot or result for {', '.join(gone[:5])}"
            f"{' ...' if len(gone) > 5 else ''} in {train / 'evals'}"
            " (were the trainer's snapshots copied in?)"
        )
    todo = evaluations.pending(train, after, upto)
    if not todo:
        fs.log(f"{where.name}: no step snapshots to score")
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
    if start is None:
        raise SystemExit(
            f"{where}: no base Chronos-2 loss: neither evals/step-0000000.json "
            "nor a locally evaluated checkpoint's held_out_loss_start"
        )
    sidecar = {
        "held_out_loss_start": start,
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
    p.add_argument("--prev", type=Path)
    p.add_argument("--device", default="auto")
    p.add_argument(
        "--eval-every",
        type=int,
        help="the trainer's --eval-every: each multiple must be scored",
    )
    p.add_argument("--real-validation", type=Path)
    args = p.parse_args(argv)
    device = torch_device(args.device)
    real = fs.load_blocks(args.real_validation) if args.real_validation else []
    if args.what == "steps":
        if args.eval_every is None:
            p.error("steps needs --eval-every")
        steps(args.where, args.prev, device, real, args.eval_every)
    else:
        checkpoint(args.where, device, real)


if __name__ == "__main__":
    main()
