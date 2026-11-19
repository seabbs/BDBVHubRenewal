"""Offloaded held-out evaluation: where its results are and who reads them.

With `finetune.eval: offload` the trainer saves an adapter snapshot and a
request at step 0 and each evaluation step (`train/evals/step-<k>/`) and
writes checkpoints without their held-out losses
(scripts/finetune_stream.py). The `evaluate` rules (rules/evaluate.smk,
run by workflow/scripts/evaluate_offloaded.py) score them: each step's
result is `train/evals/step-<k>.json` and a checkpoint's is the sidecar
`n<N>/held-out.json`, beside its `settings.json`, which is a protected
output of its segment. Readers take both through `checkpoint_settings`
and `merged_history`, so a run evaluated either way reads the same.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from bdbvhubrenewal.evaluation.sidecar import RESULT_ONLY, SIDECAR, with_sidecar

__all__ = [
    "HELD_OUT_BLOCKS",
    "RESULT_ONLY",
    "SIDECAR",
    "checkpoint_settings",
    "merged_history",
    "pending",
    "start_loss",
    "step_results",
]

HELD_OUT_BLOCKS = "held-out-blocks.pkl"
"""The held-out blocks an offloaded run saves for its evaluator
(`finetune_stream.HELD_OUT_BLOCKS`)."""


def _read(path: Path) -> Any:
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return None


def checkpoint_settings(where: Path | str) -> dict[str, Any]:
    """Return a checkpoint's settings with its sidecar's losses added."""
    where = Path(where)
    settings = _read(where / "settings.json")
    sidecar = _read(where / SIDECAR)
    return with_sidecar(
        settings if isinstance(settings, dict) else {},
        sidecar if isinstance(sidecar, dict) else None,
    )


def step_results(train: Path | str) -> dict[int, dict[str, Any]]:
    """Return each evaluated step's losses, by step."""
    out = {}
    for path in sorted(Path(train).glob("evals/step-*.json")):
        got = _read(path)
        if not isinstance(got, dict):
            continue
        step = got.get("step", int(path.stem.removeprefix("step-")))
        out[int(step)] = {k: v for k, v in got.items() if k not in RESULT_ONLY}
    return out


def merged_history(
    history: Sequence[Mapping[str, Any]], train: Path | str
) -> list[dict[str, Any]]:
    """Return the trainer's history with each step's offloaded losses.

    A step's result fills that step's evaluation entry (not a checkpoint's,
    which carries a `hash`), or is added as an entry if the trainer logged
    none; the trainer's own fields are kept.
    """
    results = step_results(train)
    out = []
    for entry in history:
        entry = dict(entry)
        if "hash" not in entry and entry.get("step") in results:
            entry = {**results.pop(entry["step"]), **entry}
        out.append(entry)
    out += [{**v, "step": k} for k, v in results.items()]
    return sorted(out, key=lambda h: h.get("step", 0))


def step_of(snapshot: Path) -> int:
    """Return a step snapshot's step, from its name `step-<k>`."""
    return int(snapshot.name.removeprefix("step-").removesuffix(".json"))


def pending(
    train: Path | str, after: int = -1, upto: int | None = None
) -> list[Path]:
    """Return the step snapshots with no result yet, in step order.

    Only those with `after` < step <= `upto`: one segment's, so two
    segments' evaluations never score the same snapshot.
    """
    evals = Path(train) / "evals"
    return [
        d
        for d in sorted(evals.glob("step-*"))
        if d.is_dir()
        and not d.name.endswith(".tmp")
        and (d / "request.json").exists()
        and not d.with_name(d.name + ".json").exists()
        and after < step_of(d) <= (upto if upto is not None else step_of(d))
    ]


def start_loss(train: Path | str) -> float | None:
    """Return base Chronos-2's held-out loss, step 0's or a checkpoint's."""
    train = Path(train)
    got = _read(train / "evals" / "step-0000000.json")
    if isinstance(got, dict) and "held_out_loss" in got:
        return float(got["held_out_loss"])
    for settings in sorted(train.glob("n*/settings.json")):
        s = _read(settings)
        if isinstance(s, dict) and s.get("held_out_loss_start") is not None:
            return float(s["held_out_loss_start"])
    return None
