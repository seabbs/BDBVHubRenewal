"""A training run's schedule and losses, for the status and loss plots.

The schedule comes from the trainer's command line (`--learning-rate`,
`--lr-steps`, `--<group>-lr`, `--prefixes`), its saved settings and the
learning rates its history logs; the losses from its history
(`train/history.json`), log rows and checkpoint settings.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from .logs import CACHE, prefixes
from .runs import Run


def _size(name: str) -> int:
    digits = re.sub(r"\D", "", name)
    return int(digits) if digits else 0


def run_params(
    folder: Path | str, hist: Any, step: int | None, args: str
) -> dict[str, Any]:
    """Return a run's schedule.

    Returns
    -------
    dict
        `lr_steps`, the optimiser `groups` (name -> base rate), each
        group's `current` rate, the rates its history `logged` (name ->
        [(step, rate)]) and the checkpoint `prefixes`.
    """
    folder = Path(folder)
    saved = sorted(
        (
            (_size(d.name), CACHE.get(d / "settings.json"))
            for d in folder.glob("train/n*")
        ),
        key=lambda x: x[0],
    )
    settings = [s for _, s in saved if isinstance(s, dict)]
    last = settings[-1] if settings else {}
    m = re.search(r"--lr-steps\s+(\d+)", args)
    lr_steps = int(m.group(1)) if m else int(last.get("lr_steps") or 0)
    groups: dict[str, float] = {}
    base_name = (
        "LoRA" if last.get("lora") or "--series-bias" in args else "adapter"
    )
    m = re.search(r"--learning-rate\s+(\S+)", args)
    if m:
        groups[base_name] = float(m.group(1))
    elif last.get("learning_rate"):
        groups[base_name] = float(last["learning_rate"])
    for m in re.finditer(r"--([a-z][a-z-]*)-lr\s+(\S+)", args):
        groups[m.group(1)] = float(m.group(2))
    for k, v in last.items():
        named = any(
            re.fullmatch(r"\w*" + k.split("_")[-1] + r"\w*", g)
            for g in groups
            if g != base_name
        )
        if isinstance(v, dict) and v.get("learning_rate") and not named:
            groups.setdefault(
                k.replace("series_", ""), float(v["learning_rate"])
            )
    logged: dict[str, list[tuple[int, float]]] = {}
    for h in hist if isinstance(hist, list) else []:
        v = h.get("lr") if isinstance(h, dict) else None
        if v is None or "step" not in h:
            continue
        if isinstance(v, list):
            v = {f"group {i}": x for i, x in enumerate(v)}
        elif not isinstance(v, dict):
            v = {"lr": v}
        for g, x in v.items():
            logged.setdefault(g, []).append((h["step"], float(x)))
    # One group per name, ignoring case; the history's spelling wins.
    spelling = {g.lower(): g for g in logged}
    for g in groups:
        spelling.setdefault(g.lower(), g)
    merged: dict[str, float] = {}
    for g, v in groups.items():
        merged.setdefault(spelling[g.lower()], v)
    groups = merged
    logged = {spelling[g.lower()]: v for g, v in logged.items()}
    for g in logged:
        groups.setdefault(g, logged[g][0][1])
    current = {}
    for g, base in groups.items():
        if g in logged:
            current[g] = logged[g][-1][1]
        elif step is not None:
            current[g] = (
                base * max(0.0, 1 - step / lr_steps) if lr_steps else base
            )
    return {
        "lr_steps": lr_steps,
        "groups": groups,
        "current": current,
        "logged": logged,
        "prefixes": prefixes(folder, args),
    }


def steps_per_simulation(
    lr_steps: int,
    sizes: Sequence[int],
    checkpoints: Sequence[tuple[str, Mapping, Any]],
) -> float:
    """Return the trainer's steps per simulated outbreak.

    The learning-rate schedule spans the final prefix; without one, the
    last checkpoint's own ratio, else 1.
    """
    if lr_steps and sizes:
        return lr_steps / max(sizes)
    for _, s, _ in reversed(checkpoints):
        if s.get("steps") and s.get("n_outbreaks"):
            return s["steps"] / s["n_outbreaks"]
    return 1.0


def history(run: Run) -> list:
    """Return a run's logged history (`train/history.json`), if any."""
    if run.folder is None:
        return []
    got = CACHE.get(run.folder / "train" / "history.json")
    return got if isinstance(got, list) else []


def loss_run(
    run: Run,
    rows: Sequence[Mapping[str, Any]],
    checkpoints: Sequence[tuple[str, Mapping, Any]],
    spo: float,
    base: Mapping[str, Any] | None,
    colour: str,
) -> dict[str, Any]:
    """Return one run's loss plot data.

    Training losses from the history, else the log rows; held-out losses
    from the log rows and the checkpoints; held-out loss by source from
    the history's evaluations, else the log rows. `base` is the step-0
    references (`full0`, `src0`) of the run that stands for base
    Chronos-2, or None for the run's own.
    """
    hist = history(run)
    h0 = next(
        (
            h
            for h in hist
            if isinstance(h, dict)
            and h.get("step") == 0
            and "held_out_loss" in h
        ),
        {},
    )
    full0 = h0.get("held_out_loss") or next(
        (
            s.get("held_out_loss_start")
            for _, s, _ in checkpoints
            if s.get("held_out_loss_start")
        ),
        None,
    )
    src0 = h0.get("by_source") or {}
    train = [
        (h["step"], h["loss"])
        for h in hist
        if isinstance(h, dict) and "loss" in h and "step" in h
    ] or [(r["step"], r["loss"]) for r in rows]
    evals = [
        (h["step"], h["by_source"])
        for h in hist
        if isinstance(h, dict)
        and h.get("step")
        and isinstance(h.get("by_source"), dict)
    ] or [(r["step"], r["src"]) for r in rows if r.get("src")]
    by: dict[str, list[list[float]]] = {}
    for st, src in sorted(evals, key=lambda e: e[0]):
        for k, v in src.items():
            by.setdefault(k, []).append([st, round(v, 5)])
    base = base or {"full0": full0, "src0": src0}
    return {
        "key": run.key,
        "label": run.label,
        "color": colour,
        "live": run.live,
        "train": [[x, round(y, 4)] for x, y in train],
        "tr0": sum(v for _, v in train[:20]) / len(train[:20])
        if train
        else None,
        "held": [[r["step"], r["held"]] for r in rows],
        "ck": [
            [s.get("steps") or round(_size(nm) * spo), s["held_out_loss"]]
            for nm, s, _ in checkpoints
            if s.get("held_out_loss")
        ],
        "full0": full0,
        "bfull": base["full0"],
        "src0": src0,
        "bsrc": base["src0"],
        "bysrc": by,
    }
