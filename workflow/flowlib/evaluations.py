"""Offloaded held-out evaluation: where its results are and who reads them.

With `finetune.eval: offload` the trainer saves an adapter snapshot and a
request at step 0 and each evaluation step (`train/evals/step-<k>/`) and
writes checkpoints without their held-out losses
(scripts/finetune_stream.py). The `evaluate` rules (rules/evaluate.smk)
score them elsewhere: each step's result is `train/evals/step-<k>.json`
and a checkpoint's is the sidecar `n<N>/held-out.json`, beside its
`settings.json`, which is a protected output of its segment. Readers take
both through `checkpoint_settings` and `merged_history`, so a run
evaluated either way reads the same.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

HELD_OUT_BLOCKS = "held-out-blocks.pkl"
"""The held-out blocks an offloaded run saves for its evaluator
(`finetune_stream.HELD_OUT_BLOCKS`)."""

SIDECAR = "held-out.json"
"""A checkpoint's offloaded held-out losses, beside its settings."""

RESULT_ONLY = ("kinds", "device", "seconds", "n_outbreaks")
"""Fields of a step's result that are about the evaluation, not losses."""


def _read(path: Path) -> Any:
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return None


def checkpoint_settings(where: Path | str) -> dict[str, Any]:
    """Return a checkpoint's settings with its sidecar's losses added.

    The sidecar only adds fields: one the trainer wrote is kept as it is.
    """
    where = Path(where)
    settings = _read(where / "settings.json")
    settings = settings if isinstance(settings, dict) else {}
    sidecar = _read(where / SIDECAR)
    if not isinstance(sidecar, dict):
        return settings
    losses = {k: v for k, v in sidecar.items() if k not in RESULT_ONLY}
    return {**losses, **settings}


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


def pending(train: Path | str) -> list[Path]:
    """Return the step snapshots that have no result yet, in step order."""
    evals = Path(train) / "evals"
    return [
        d
        for d in sorted(evals.glob("step-*"))
        if d.is_dir()
        and not d.name.endswith(".tmp")
        and (d / "request.json").exists()
        and not d.with_name(d.name + ".json").exists()
    ]


def start_loss(train: Path | str) -> float | None:
    """Return base Chronos-2's held-out loss for the run, if known.

    From the step-0 snapshot's result, else from a checkpoint the trainer
    evaluated itself (a run switched to offload at a later segment).
    """
    train = Path(train)
    got = _read(train / "evals" / "step-0000000.json")
    if isinstance(got, dict) and "held_out_loss" in got:
        return float(got["held_out_loss"])
    for settings in sorted(train.glob("n*/settings.json")):
        s = _read(settings)
        if isinstance(s, dict) and s.get("held_out_loss_start") is not None:
            return float(s["held_out_loss_start"])
    return None
