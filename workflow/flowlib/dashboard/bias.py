"""Attention bias by relation, for the dashboard's bias row.

A run with series attention bias holds one [layers, heads] tensor per
relation: `same` (a place's counts, lags and totals), `up` (a place to
its country) and `down` (a country to its places). Each run shows the
relations it holds. The values come
from the run's history (`train/history.json` entries with `bias`), its
logged steps' means and ranges, its checkpoints' adapters
(`train/n<N>/finetuned-ckpt/*.safetensors`, read with the standard
library, or `*.pt`, read with torch when installed) and, for the live
run, its current state (`train/state/`).
"""

from __future__ import annotations

import json
import re
import struct
from array import array
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

from .logs import CACHE, _size, checkpoints
from .runs import Run

PAIRS = (
    ("same", "same"),
    ("up", "up (child to parent)"),
    ("down", "down (parent to child)"),
)
"""The trainer's relations and their labels, in its order. A run shows
the relations it logged or saved, these first."""

COLOURS = {
    "same": "#2563eb",
    "up": "#16a34a",
    "down": "#ea580c",
}

OTHER = "#6b7280"
"""The colour of a relation a run holds beyond `PAIRS`."""

NAME = re.compile(r"series_bias(?:\.\w+)*?\.([a-z]+)_?$")
"""A relation's tensor of the bias module, by the end of its name."""


def relation(name: str) -> str | None:
    """Return the relation a tensor or logged key names, if any."""
    found = NAME.search(name) or re.fullmatch(r"([a-z]+)_?", name)
    return found.group(1) if found else None


_SEEN: dict[tuple[str, int], dict[str, list[list[float]]]] = {}


def _safetensors(path: Path) -> dict[str, list[list[float]]]:
    """Return a safetensors file's 2-D float tensors that name a relation."""
    out = {}
    with path.open("rb") as f:
        n = int.from_bytes(f.read(8), "little")
        head = json.loads(f.read(n))
        for name, info in head.items():
            shape = info.get("shape", []) if isinstance(info, dict) else []
            if (
                name == "__metadata__"
                or len(shape) != 2
                or not NAME.search(name)
            ):
                continue
            a, b = info["data_offsets"]
            f.seek(8 + n + a)
            raw = f.read(b - a)
            kind = info["dtype"]
            if kind == "F32":
                values: Sequence[float] = array("f", raw)
            elif kind == "F64":
                values = array("d", raw)
            elif kind == "F16":
                values = struct.unpack(f"<{len(raw) // 2}e", raw)
            elif kind == "BF16":
                values = [
                    struct.unpack("<f", b"\0\0" + raw[i : i + 2])[0]
                    for i in range(0, len(raw), 2)
                ]
            else:
                continue
            rows, cols = shape
            out[name] = [
                [float(x) for x in values[i * cols : (i + 1) * cols]]
                for i in range(rows)
            ]
    return out


def _torch():
    """Return torch, or None if it is not installed."""
    try:
        import torch
    except ImportError:
        return None
    return torch


def _pt(path: Path) -> dict[str, list[list[float]]]:
    """Return a torch file's 2-D tensors that name a relation."""
    torch = _torch()
    if torch is None:
        return {}
    state = torch.load(path, map_location="cpu", weights_only=True)
    out = {}

    def walk(obj: Any, prefix: str) -> None:
        if isinstance(obj, dict):
            for k, v in obj.items():
                walk(v, f"{prefix}.{k}" if prefix else str(k))
        elif (
            hasattr(obj, "shape")
            and len(obj.shape) == 2
            and NAME.search(prefix)
        ):
            out[prefix] = obj.float().tolist()

    walk(state, "")
    return out


def read_bias(path: Path | str) -> dict[str, list[list[float]]]:
    """Return an adapter file's bias by relation (cached by time).

    A relation may have several tensors; the saved module's is preferred
    over an original copy.
    """
    path = Path(path)
    try:
        key = (str(path), path.stat().st_mtime_ns)
    except OSError:
        return {}
    if key not in _SEEN:
        try:
            tensors = (
                _pt(path)
                if path.suffix in (".pt", ".bin")
                else (_safetensors(path))
            )
        except Exception:
            tensors = {}
        got = {}
        by: dict[str, list[str]] = {}
        for n in tensors:
            if NAME.search(n):
                by.setdefault(NAME.search(n).group(1), []).append(n)
        for rel, names in by.items():
            names.sort(
                key=lambda n: ("modules_to_save" not in n, "original" in n, n)
            )
            got[rel] = tensors[names[0]]
        _SEEN[key] = got
    return _SEEN[key]


def summary(rows: Sequence[Sequence[float]]) -> dict[str, float]:
    """Return the mean, min and max of a [layers, heads] matrix."""
    flat = [x for r in rows for x in r]
    return {"mean": sum(flat) / len(flat), "min": min(flat), "max": max(flat)}


def run_bias(
    run: Run,
    step: int | None,
    rows: Sequence[Mapping[str, Any]] = (),
    sims: Callable[[float], float] | None = None,
) -> dict[str, Any] | None:
    """Return a run's bias statistics over simulations and its latest matrices.

    Only the run's own files count, in this order: its history's
    evaluations, its logged steps' means and ranges (`rows`, from
    `logs.RunLog`, for steps the history lacks), its checkpoints'
    adapters (at their own simulations) and, for the live run, its
    current state at `step` (None for other runs). `sims` gives the
    simulations trained by a step (`page.Context.sims`); without it, x
    is the step. Returns None if the run has no bias anywhere.
    """
    if run.folder is None:
        return None
    sims = sims or float
    pts: dict[str, dict[float, dict[str, float]]] = {}
    latest: dict[str, Any] | None = None
    seen: set[int] = set()
    hist = CACHE.get(run.folder / "train" / "history.json")
    for h in hist if isinstance(hist, list) else []:
        if not isinstance(h, dict) or "step" not in h:
            continue
        if not isinstance(h.get("bias"), dict):
            continue
        at = round(sims(h["step"]), 2)
        for key, v in h["bias"].items():
            rel = relation(key)
            if rel is None:
                continue
            values = v.get("values") if isinstance(v, dict) else v
            if not (
                isinstance(values, list)
                and values
                and isinstance(values[0], list)
            ):
                continue
            seen.add(h["step"])
            given = isinstance(v, dict) and {"mean", "min", "max"} <= set(v)
            pts.setdefault(rel, {})[at] = (
                {x: v[x] for x in ("mean", "min", "max")}
                if given
                else summary(values)
            )
            if not latest or at > latest["n"]:
                latest = {"n": at}
            if latest["n"] == at:
                latest[rel] = values
    for r in rows:
        if r.get("step") in seen or not r.get("bias"):
            continue
        at = round(sims(r["step"]), 2)
        for key, v in r["bias"].items():
            rel = relation(key)
            if rel is not None:
                pts.setdefault(rel, {}).setdefault(at, dict(v))
    files = []
    for name, s, _ in checkpoints(run.folder):
        n = s.get("n_outbreaks") or _size(name)
        ckpt = run.folder / "train" / name / "finetuned-ckpt"
        for f in sorted(ckpt.glob("*.safetensors")) + sorted(ckpt.glob("*.pt")):
            files.append((float(n), f))
    state = run.folder / "train" / "state"
    if step is not None:
        current = (
            [state / "adapter.pt"]
            if (state / "adapter.pt").is_file()
            else sorted(state.glob("adapter*.safetensors"))
        )
        files += [(round(sims(step), 2), f) for f in current]
    for at, f in sorted(files, key=lambda x: x[0]):
        got = read_bias(f)
        if not got:
            continue
        for rel, values in got.items():
            pts.setdefault(rel, {}).setdefault(at, summary(values))
        if not latest or at >= latest.get("n", -1):
            latest = {"n": at, **got}
    if not any(pts.values()) and not latest:
        return None
    return {
        "pts": {k: sorted(v.items()) for k, v in pts.items()},
        "latest": latest,
    }


def payload(
    run_list: Sequence[Run],
    live_step: int | None,
    rows: Mapping[str, Sequence[Mapping[str, Any]]] | None = None,
    sims: Mapping[str, Callable[[float], float]] | None = None,
) -> dict[str, Any]:
    """Return the bias row's data for the runs that have bias.

    `rows` and `sims` are each run's logged steps and step-to-simulations
    map, by run key (`run_bias`).
    """
    out, missing = [], []
    for r in run_list:
        data = run_bias(
            r,
            live_step if r.live else None,
            (rows or {}).get(r.key, ()),
            (sims or {}).get(r.key),
        )
        if not data:
            missing.append({"key": r.key, "label": r.label})
            continue
        last = data["latest"] or {}
        out.append(
            {
                "key": r.key,
                "label": r.label,
                "color": r.colour,
                "pts": {
                    k: [[s, v["mean"], v["min"], v["max"]] for s, v in pts]
                    for k, pts in data["pts"].items()
                },
                "latest": {
                    "n": last.get("n"),
                    **{k: v for k, v in last.items() if k != "n"},
                },
            }
        )
    # The trainer's relations first, then any other a run holds.
    held = {k for r in out for k, v in r["pts"].items() if v}
    labels = dict(PAIRS)
    keys = [k for k, _ in PAIRS if k in held] + sorted(held - set(labels))
    return {
        "runs": out,
        "missing": missing,
        "col": {k: COLOURS.get(k, OTHER) for k in keys},
        "pairs": [{"key": k, "label": labels.get(k, k)} for k in keys],
    }
