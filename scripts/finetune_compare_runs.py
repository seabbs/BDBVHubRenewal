# Copyright (c) 2026 Sam Abbott. MIT licence, see LICENSE.
r"""Compare two continuous fine-tuning runs' held-out loss by source.

Each checkpoint `nK` of either run is scored on the held-out blocks of
the given sources, cut as the trainer cuts them
(`scripts/finetune_stream.py`), so the two runs are compared on the same
blocks at the same number of simulations. Weekly and daily blocks are cut
alike by both runs; 3-day blocks are not compared. Each run's logged
curve (`history.json`, by source where the trainer logged it) is added
by step. Rows are appended to the CSV once per run, checkpoint and source.

    python scripts/finetune_compare_runs.py \\
        slots=output/finetune/continuous-slots \\
        no-slots=../fm-finetune-r4/output/finetune/continuous \\
        --out output/finetune/slots-vs-noslots.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import finetune_chronos as ft
import finetune_stream as fs

FIELDS = (
    "run",
    "kind",
    "n_outbreaks",
    "step",
    "source",
    "loss",
    "blocks",
    "checkpoint_hash",
)


def held_out_blocks(run: Path, sources: tuple[str, ...]) -> list[dict]:
    """Return a run's held-out blocks of `sources`, as its trainer cuts them.

    32 blocks per simulation from 2 draws, subset seed 1 and seed 1.
    """
    held = (run / "train" / "outbreaks-held-out.txt").read_text().split()
    fs.init_worker(
        min(ft.MAX_SERIES, ft.max_series(ft.MEMORY_GB)),
        ft.MAX_LAGS,
        ("weekly", "daily", "3day"),
        0.2,
        2,
        1,
    )
    blocks = [b for _, b in fs.cut(run / "sims", held, 1, 32)]
    return [b for b in blocks if b["source"] in sources]


def load(checkpoint: Path):
    """Return base Chronos-2 with a checkpoint's adapter (and slots)."""
    from chronos import Chronos2Pipeline
    from peft import PeftModel

    from bdbvhubrenewal.forecasters import slots
    from bdbvhubrenewal.forecasters.foundation import (
        CHRONOS_MODEL,
        CHRONOS_REVISION,
    )

    settings = json.loads((checkpoint / "settings.json").read_text())
    pipeline = Chronos2Pipeline.from_pretrained(
        CHRONOS_MODEL, revision=CHRONOS_REVISION, device_map="cpu"
    )
    n_slots = settings.get("series_slots", {}).get("n", 0)
    if n_slots:
        slots.attach(pipeline.model, n_slots)
    model = PeftModel.from_pretrained(
        pipeline.model, checkpoint / "finetuned-ckpt"
    )
    return model.eval(), settings


def done(out: Path) -> set[tuple[str, str, str, str]]:
    """Return the (run, kind, n_outbreaks or step, source) rows written."""
    if not out.exists():
        return set()
    with out.open() as f:
        return {
            (r["run"], r["kind"], r["n_outbreaks"] or r["step"], r["source"])
            for r in csv.DictReader(f)
        }


def curve_rows(name: str, run: Path, sources: tuple[str, ...]) -> list[dict]:
    """Return a run's logged held-out curve by source, by step."""
    history = run / "train" / "history.json"
    if not history.exists():
        return []
    rows = []
    for entry in json.loads(history.read_text()):
        for source, loss in entry.get("by_source", {}).items():
            if source in sources:
                rows.append(
                    {
                        "run": name,
                        "kind": "curve",
                        "n_outbreaks": "",
                        "step": entry["step"],
                        "source": source,
                        "loss": loss,
                        "blocks": f"first {ft.EVAL_BLOCKS}",
                        "checkpoint_hash": "",
                    }
                )
    return rows


def main() -> None:
    """Score each run's checkpoints by source and append the rows."""
    parser = argparse.ArgumentParser()
    parser.add_argument("runs", nargs="+", help="name=run folder")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--sources", nargs="+", default=["weekly", "daily"])
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()
    runs = dict(r.split("=", 1) for r in args.runs)
    sources = tuple(args.sources)
    have = done(args.out)
    rows: list[dict] = []
    # Every run is scored on the first run's held-out blocks.
    blocks = None
    for name, folder in runs.items():
        run = Path(folder)
        for row in curve_rows(name, run, sources):
            key = (name, "curve", str(row["step"]), row["source"])
            if key not in have:
                rows.append(row)
        for checkpoint in sorted(
            (run / "train").glob("n*/settings.json"),
            key=lambda p: int(p.parent.name[1:]),
        ):
            n = checkpoint.parent.name[1:]
            if all((name, "checkpoint", n, s) in have for s in sources):
                continue
            if blocks is None:
                first = Path(next(iter(runs.values())))
                blocks = held_out_blocks(first, sources)
            model, settings = load(checkpoint.parent)
            losses = fs.held_out_by_source(model, blocks, args.batch_size)
            counts = {s: sum(b["source"] == s for b in blocks) for s in sources}
            for source, loss in losses.items():
                rows.append(
                    {
                        "run": name,
                        "kind": "checkpoint",
                        "n_outbreaks": n,
                        "step": settings.get("steps", ""),
                        "source": source,
                        "loss": loss,
                        "blocks": counts[source],
                        "checkpoint_hash": settings["hash"],
                    }
                )
            print(name, f"n{n}", {s: round(v, 4) for s, v in losses.items()})
    new = not args.out.exists()
    with args.out.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        if new:
            writer.writeheader()
        writer.writerows(rows)
    print(f"{len(rows)} rows appended to {args.out}")


if __name__ == "__main__":
    main()
