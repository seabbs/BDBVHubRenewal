"""A training run's schedule and losses, for the status and loss plots.

The schedule comes from the trainer's command line (`--learning-rate`,
`--outbreaks`, `--<group>-lr`, `--prefixes`, or `--lr-steps` of runs
that decayed over steps), its saved settings and the
learning rates its history logs; the losses from its history
(`train/history.json`), log rows and checkpoint settings.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Mapping, Sequence
from datetime import datetime
from itertools import pairwise
from pathlib import Path
from typing import Any

from .logs import CACHE, prefixes
from .runs import Run


def _size(name: str) -> int:
    digits = re.sub(r"\D", "", name)
    return int(digits) if digits else 0


def _falls(hist: Any) -> bool:
    """Return whether any optimiser group's logged rate fell over steps."""
    seen: dict[str, float] = {}
    for h in hist if isinstance(hist, list) else []:
        v = h.get("lr") if isinstance(h, dict) else None
        if isinstance(v, list):
            v = {str(i): x for i, x in enumerate(v)}
        elif not isinstance(v, dict):
            v = {"lr": v} if v is not None else {}
        for g, x in v.items():
            if g in seen and float(x) < seen[g]:
                return True
            seen[g] = float(x)
    return False


def run_params(
    folder: Path | str,
    hist: Any,
    step: int | None,
    args: str,
    default: Sequence[int],
    sims: float | None = None,
) -> dict[str, Any]:
    """Return a run's schedule.

    `default` are the run's checkpoint sizes, the last its simulations,
    unless the trainer's command line `args` names them; `sims` is the
    simulations trained by `step`, for the current rate of a run that
    decays over simulations.

    Returns
    -------
    dict
        `lr_steps` (runs that decayed over steps), `lr_outbreaks` (the
        simulations the rate falls to zero over, otherwise), the
        optimiser `groups` (name -> base rate), each
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
    # Runs without a step count decay over `--outbreaks`, as their
    # settings record (`lr_decay`); older runs with settings but no
    # `lr_decay` kept a constant rate.
    # A trainer the workflow starts leaves no command line: its run's
    # simulations are the last of `default`.
    m = re.search(r"--outbreaks\s+(\d+)", args)
    # A rate the history logs falling is a decay whatever the settings say.
    decays = last.get("lr_decay") or not settings or _falls(hist)
    total = int(m.group(1)) if m else max(default, default=0)
    lr_outbreaks = total if decays and not lr_steps else 0
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
        elif lr_steps and step is not None:
            current[g] = base * max(0.0, 1 - step / lr_steps)
        elif lr_outbreaks and sims is not None:
            current[g] = base * max(0.0, 1 - sims / lr_outbreaks)
        elif step is not None:
            current[g] = base
    return {
        "lr_steps": lr_steps,
        "lr_outbreaks": lr_outbreaks,
        "groups": groups,
        "current": current,
        "logged": logged,
        "prefixes": prefixes(folder, args, default),
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


def anchors(
    checkpoints: Sequence[tuple[str, Mapping, Any]],
    chunks: Mapping[int, tuple[int, int]],
) -> list[tuple[int, int]]:
    """Return a run's known (step, simulations trained) pairs, by step.

    Step 0 is no simulations; each checkpoint records its steps and
    simulations (`settings.json`); and a planned chunk (`logs.Log.chunks`)
    that starts where a known pair ends adds its steps there. Only the
    run's own files count.
    """
    known = {0: 0}
    for _, s, _ in checkpoints:
        if s.get("steps") is not None and s.get("n_outbreaks"):
            known[int(s["n_outbreaks"])] = int(s["steps"])
    for before in sorted(chunks):
        last, steps = chunks[before]
        if before in known and last not in known:
            known[last] = known[before] + steps
    return sorted((st, n) for n, st in known.items())


RATE_WINDOW = 3600.0
"""Seconds of timestamped steps the speed is taken over."""


def recent_rate(
    points: Sequence[tuple[float, float]], window: float = RATE_WINDOW
) -> tuple[float | None, float | None]:
    """Return the steps a second and the seconds they span, or Nones.

    `points` are (time, step), from every logged step and the trainer's
    status file. The rate is over the last `window` seconds; with fewer
    than two points there (evaluations logged only every few hundred
    steps), over the last two points, however far apart.
    """
    pts = sorted(set(points))
    if len(pts) < 2:
        return None, None
    last_t, last_s = pts[-1]
    inside = [p for p in pts if p[0] >= last_t - window]
    first_t, first_s = inside[0] if len(inside) >= 2 else pts[-2]
    span = last_t - first_t
    if span <= 0 or last_s <= first_s:
        return None, None
    return (last_s - first_s) / span, span


def status_point(folder: Path | str) -> tuple[float, float] | None:
    """Return the trainer's status time and step while it trains, or None.

    `train/stream-status.json` holds the step every few steps with the
    local time it was written.
    """
    s = CACHE.get(Path(folder) / "train" / "stream-status.json")
    if not isinstance(s, dict) or s.get("stage") != "training":
        return None
    try:
        t = datetime.strptime(str(s["time"]), "%Y-%m-%d %H:%M:%S").timestamp()
        return t, float(s["step"])
    except (KeyError, TypeError, ValueError):
        return None


def step_at(
    pairs: Sequence[tuple[int, int]], n: int, spo: float = 1.0
) -> tuple[int, bool]:
    """Return the step at which `n` simulations are trained, from `anchors`.

    With whether it is planned: a checkpoint or a planned chunk ends at
    `n`. Otherwise past the last pair at the run's mean rate so far, or
    `spo` steps a simulation if no pair is known past step 0.
    """
    pairs = list(pairs) or [(0, 0)]
    for s, k in pairs:
        if k == n:
            return s, True
    for (s0, n0), (s1, n1) in pairwise(pairs):
        if n0 < n < n1:
            return round(s0 + (s1 - s0) * (n - n0) / (n1 - n0)), False
    s1, n1 = pairs[-1]
    rate = s1 / n1 if n1 else spo
    return round(s1 + (n - n1) * rate), False


def sims_at(
    pairs: Sequence[tuple[int, int]], step: float, spo: float = 1.0
) -> float:
    """Return the simulations trained by `step`, from `anchors`.

    Linear between known pairs, as the trainer's own schedule is within
    a chunk (`sims_done`); past the last, at the run's mean rate so far,
    or `spo` steps a simulation if no pair is known past step 0.
    """
    pairs = list(pairs) or [(0, 0)]
    for (s0, n0), (s1, n1) in pairwise(pairs):
        if step <= s1:
            return n0 + (n1 - n0) * (step - s0) / ((s1 - s0) or 1)
    s1, n1 = pairs[-1]
    return n1 * step / s1 if s1 else step / spo


def history(run: Run) -> list:
    """Return a run's logged history (`train/history.json`), if any."""
    if run.folder is None:
        return []
    got = CACHE.get(run.folder / "train" / "history.json")
    return got if isinstance(got, list) else []


PARTS = ("nowcast", "forecast")
"""The parts of held-out loss: the steps to each block's cut-off and
those after it."""


def _parts(entry: Mapping[str, Any], name: str) -> dict[str, float]:
    """Return the nowcast and forecast parts of loss `name`, if both."""
    got = {p: entry.get(f"{name}_{p}") for p in PARTS}
    return got if all(v is not None for v in got.values()) else {}


def loss_run(
    run: Run,
    rows: Sequence[Mapping[str, Any]],
    checkpoints: Sequence[tuple[str, Mapping, Any]],
    sims: Callable[[float], float],
    base: Mapping[str, Any] | None,
    colour: str,
) -> dict[str, Any]:
    """Return one run's loss plot data, against simulations trained.

    Training losses from the history, else the log rows; held-out losses
    from the log rows and the checkpoints; held-out loss by source from
    the history's evaluations, else the log rows. `sims` gives the
    simulations trained by a step (`sims_at`); checkpoints are at their
    own simulations. `base` is the step-0
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
    parts0 = _parts(h0, "held_out_loss")
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

    def x(step: float) -> float:
        return round(sims(step), 2)

    def at(name: str, s: Mapping[str, Any]) -> int:
        return int(s.get("n_outbreaks") or _size(name))

    by: dict[str, list[list[float]]] = {}
    for st, src in sorted(evals, key=lambda e: e[0]):
        for k, v in src.items():
            by.setdefault(k, []).append([x(st), round(v, 5)])
    lvl0 = h0.get("by_level") or next(
        (
            s.get("held_out_loss_by_level_start")
            for _, s, _ in checkpoints
            if isinstance(s.get("held_out_loss_by_level_start"), dict)
        ),
        {},
    )

    def by_key(field: str) -> dict[str, list[list[float]]]:
        out: dict[str, list[list[float]]] = {}
        for nm, s, _ in checkpoints:
            for k, v in (s.get(field) or {}).items():
                out.setdefault(k, []).append([at(nm, s), round(float(v), 5)])
        return out

    base = base or {
        "full0": full0,
        "src0": src0,
        "parts0": parts0,
        "lvl0": lvl0,
    }
    return {
        "key": run.key,
        "label": run.label,
        "color": colour,
        "live": run.live,
        "train": [[x(st), round(y, 4)] for st, y in train],
        "tr0": sum(v for _, v in train[:20]) / len(train[:20])
        if train
        else None,
        "held": [[x(r["step"]), r["held"]] for r in rows],
        "ck": [
            [at(nm, s), s["held_out_loss"]]
            for nm, s, _ in checkpoints
            if s.get("held_out_loss")
        ],
        "held_parts": {
            p: [[x(r["step"]), r["parts"][p]] for r in rows if r.get("parts")]
            for p in PARTS
        },
        "ck_parts": {
            p: [
                [at(nm, s), parts[p]]
                for nm, s, _ in checkpoints
                if (parts := _parts(s, "held_out_loss"))
            ]
            for p in PARTS
        },
        "full0": full0,
        "parts0": parts0,
        "bparts0": base.get("parts0") or {},
        "bfull": base["full0"],
        "src0": src0,
        "bsrc": base["src0"],
        "bysrc": by,
        # Checkpoints' full-set loss by layout and by level, and the level
        # losses at step 0; the trainer logs no level losses in between.
        "ck_src": by_key("held_out_loss_by_source"),
        "ck_lvl": by_key("held_out_loss_by_level"),
        "lvl0": lvl0,
        "blvl0": base.get("lvl0") or {},
    }
