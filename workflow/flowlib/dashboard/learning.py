"""What fine-tuning learns, from what the trainer writes.

Held-out loss by level at each checkpoint (`settings.json`), relative
to its loss at the start (`history.json`); and the size of the LoRA
update, the Frobenius norm of the scaled `B @ A`, by encoder layer at
each checkpoint and by kind of module over checkpoints. The loss by
source and the attention bias have their own sections. No scoring: only
the trainer's own files are read.
"""

from __future__ import annotations

import html
import json
import math
import re
from array import array
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

from bdbvhubrenewal.evaluation import labels
from flowlib.dashboard import logs, svg, training
from flowlib.dashboard.runs import Run

LORA = re.compile(r"\.lora_([AB])\.weight$")
"""A LoRA matrix, by side."""

BLOCK = re.compile(r"encoder\.block\.(\d+)\.layer\.(\d+)\.")
"""An encoder block's sublayer: 0 time attention, 1 group attention."""

KINDS = {"0": "time attention", "1": "group attention"}
"""Encoder sublayers by index (`Chronos2EncoderBlock.layer`)."""

EXPLAINERS = ("learnlevel", "learnlayer", "learnkind")
"""The explainer keys (`page/explain.json`) of the three panels, in order."""


def colour_of(key: str, keys: Sequence[str]) -> str:
    """Return a key's colour from the shared map.

    A level's or a layout's own colour, else the categorical colour at its
    position among `keys`.
    """
    named = {**labels.LAYOUT_COLOURS, **labels.LEVEL_COLOURS}
    if key in named:
        return named[key]
    return labels.CATEGORICAL[keys.index(key) % len(labels.CATEGORICAL)]


def _relative(
    points: Sequence[tuple[int, Mapping[str, float]]],
) -> dict[str, list[tuple[int, float]]]:
    """Return each key's values over steps divided by its first value."""
    out: dict[str, list[tuple[int, float]]] = {}
    start = points[0][1] if points else {}
    for step, values in points:
        for key, value in values.items():
            if start.get(key) and value is not None:
                out.setdefault(key, []).append((step, value / start[key]))
    return out


def _start(history: Sequence[Mapping[str, Any]]) -> Mapping[str, Any] | None:
    return next((h for h in history if h.get("step") == 0), None)


def by_level(
    history: Sequence[Mapping[str, Any]],
    checkpoints: Sequence[tuple[str, Mapping[str, Any]]],
) -> dict[str, list[tuple[int, float]]]:
    """Return held-out loss by level at each checkpoint, relative to start.

    Against each checkpoint's simulations trained.
    """
    start = _start(history)
    if start is None or not start.get("by_level"):
        return {}
    points = [(0, start["by_level"])] + [
        (
            int(s.get("n_outbreaks") or logs._size(name)),
            s["held_out_loss_by_level"],
        )
        for name, s in checkpoints
        if s.get("held_out_loss_by_level")
    ]
    return _relative(points)


def _matrices(path: Path) -> dict[str, tuple[list[int], array]]:
    """Return a safetensors file's float32 LoRA matrices of encoder layers."""
    out = {}
    with path.open("rb") as f:
        n = int.from_bytes(f.read(8), "little")
        head = json.loads(f.read(n))
        for name, info in head.items():
            if name == "__metadata__" or not LORA.search(name):
                continue
            if info.get("dtype") != "F32":
                continue
            a, b = info["data_offsets"]
            f.seek(8 + n + a)
            out[name] = (info["shape"], array("f", f.read(b - a)))
    return out


def _gram(shape: list[int], values: array, rows: bool) -> list[list[float]]:
    """Return `M M^T` (`rows`) or `M^T M` of a rank-r LoRA matrix."""
    n, m = shape
    if rows:  # A is r x in: A A^T is r x r.
        vec = [values[i * m : (i + 1) * m] for i in range(n)]
    else:  # B is out x r: B^T B is r x r.
        vec = [values[j::m] for j in range(m)]
    return [
        [sum(x * y for x, y in zip(u, v, strict=True)) for v in vec]
        for u in vec
    ]


_SEEN: dict[tuple[str, float], dict[str, float]] = {}


def norms(ckpt: Path | str) -> dict[str, float]:
    """Return the squared size of each adapted module's update.

    The squared Frobenius norm of `(alpha / r) B @ A`, from
    `||B A||^2 = sum((B^T B) * (A A^T))`, by the module's name.
    """
    ckpt = Path(ckpt)
    key = (str(ckpt), (ckpt / "adapter_model.safetensors").stat().st_mtime)
    if key in _SEEN:
        return _SEEN[key]
    config = json.loads((ckpt / "adapter_config.json").read_text())
    scale = config["lora_alpha"] / config["r"]
    mats = _matrices(ckpt / "adapter_model.safetensors")
    out = {}
    for name, (shape, values) in mats.items():
        other = mats.get(name.replace("lora_A", "lora_B"))
        if "lora_A" not in name or other is None:
            continue
        gram_a = _gram(shape, values, rows=True)
        gram_b = _gram(other[0], other[1], rows=False)
        norm2 = sum(
            x * y
            for ra, rb in zip(gram_a, gram_b, strict=True)
            for x, y in zip(ra, rb, strict=True)
        )
        out[name.replace(".lora_A.weight", "")] = scale**2 * max(norm2, 0.0)
    _SEEN[key] = out
    return out


def lora_size(ckpt: Path | str) -> list[tuple[int, float]]:
    """Return the size of each encoder layer's LoRA update.

    A layer's modules (`norms`) add in quadrature.
    """
    squared: dict[int, float] = {}
    for name, value in norms(ckpt).items():
        match = BLOCK.search(name)
        if match:
            layer = int(match.group(1))
            squared[layer] = squared.get(layer, 0.0) + value
    return [(k, math.sqrt(v)) for k, v in sorted(squared.items())]


def _adapters(run: Run):
    """Yield each saved checkpoint's name, settings and adapter folder."""
    if run.folder is None:
        return
    for name, settings, _ in logs.checkpoints(run.folder):
        ckpt = Path(run.folder) / "train" / name / "finetuned-ckpt"
        if (ckpt / "adapter_model.safetensors").is_file():
            yield name, settings, ckpt


def run_lora(run: Run) -> dict[str, list[tuple[int, float]]]:
    """Return the LoRA update size by layer at each of a run's checkpoints."""
    return {name: lora_size(ckpt) for name, _, ckpt in _adapters(run)}


def run_kinds(run: Run) -> dict[str, list[tuple[int, float]]]:
    """Return the LoRA update size by kind of module over checkpoints.

    Time attention, group attention and the output layer, each the
    modules of that kind in quadrature, at each checkpoint's simulations.
    """
    out: dict[str, list[tuple[int, float]]] = {}
    for _, settings, ckpt in _adapters(run):
        squared: dict[str, float] = {}
        for name, value in norms(ckpt).items():
            match = BLOCK.search(name)
            kind = KINDS.get(match.group(2), "") if match else "output layer"
            if kind:
                squared[kind] = squared.get(kind, 0.0) + value
        n = int(settings.get("n_outbreaks") or 0)
        for kind, value in squared.items():
            out.setdefault(kind, []).append((n, math.sqrt(value)))
    return out


RUN_DASHES = (None, "dash", "dot", "dd")
"""Line styles of runs, in configured order, where colour is another key."""


def _dash(i: int) -> tuple[str, ...]:
    d = RUN_DASHES[i % len(RUN_DASHES)]
    return (d,) if d else ()


def _swatch(colour: str, dash: str | None = None) -> str:
    style = f' stroke-dasharray="{svg.DASHES[dash][0]}"' if dash else ""
    return (
        '<svg width=22 height=8 viewBox="0 0 22 8"><line x1=0 x2=22 y1=4 '
        f'y2=4 stroke="{colour}" stroke-width=2{style} /></svg>'
    )


def _legend(keys: Sequence[str], run_list: Sequence[tuple[int, Run]]) -> str:
    """Return one legend: colour by key, then line style by run."""
    esc = html.escape
    out = [
        f'<span><i style="background:{colour_of(k, keys)}"></i>{esc(k)}</span>'
        for k in keys
    ]
    if len(run_list) > 1:
        out += [
            f"<span>{_swatch('var(--fg)', RUN_DASHES[i % len(RUN_DASHES)])}"
            f" {esc(r.label)}</span>"
            for i, r in run_list
        ]
    return "".join(out)


def _by_run(
    curves: Mapping[str, Mapping[str, Sequence[tuple[float, float]]]],
    run_list: Sequence[Run],
) -> tuple[list[tuple[Any, ...]], str]:
    """Return series coloured by key and styled by run, and their legend."""
    keys = sorted({k for c in curves.values() for k in c})
    shown = [(i, r) for i, r in enumerate(run_list) if curves.get(r.key)]
    series = [
        (
            f"{k} ({r.label})",
            colour_of(k, keys),
            list(curves[r.key][k]),
            "pts",
            *_dash(i),
        )
        for i, r in shown
        for k in sorted(curves[r.key])
    ]
    return series, _legend(keys, shown)


def _checkpoints(run: Run) -> list[tuple[str, Mapping[str, Any]]]:
    if run.folder is None:
        return []
    return [(n, s) for n, s, _ in logs.checkpoints(run.folder)]


def section(
    run_list: Sequence[Run], tip: Callable[[str], str] = lambda _: ""
) -> str:
    """Return the three panels of what fine-tuning learns, for every run.

    Each run's own history and checkpoints only, against simulations
    trained. `tip` gives the info marker of an explainer key
    (`page.tip`).
    """
    run_list = list(run_list)
    levels, legend = _by_run(
        {
            r.key: by_level(training.history(r), _checkpoints(r))
            for r in run_list
        },
        run_list,
    )
    level_card = svg.chart(
        "Held-out loss by level at each checkpoint, relative to the start",
        levels,
        hlines=[1.0],
        xlabel="simulations trained",
        info=tip("learnlevel"),
        note="All held-out blocks, each level scored with the others masked.",
        legend=legend,
    )
    # Each run's last checkpoint, in the run's colour.
    layers = []
    for r in run_list:
        sizes = run_lora(r)
        if sizes:
            name = max(sizes, key=logs._size)
            layers.append((f"{r.label} {name}", r.colour, sizes[name], "pts"))
    lora = svg.chart(
        "Size of the LoRA update by encoder layer, at each run's last "
        "checkpoint",
        layers,
        xlabel="encoder layer",
        info=tip("learnlayer"),
        note="Frobenius norm of the scaled B·A, a layer's attention "
        "modules in quadrature.",
    )
    kinds, legend = _by_run({r.key: run_kinds(r) for r in run_list}, run_list)
    kind_card = svg.chart(
        "Size of the LoRA update by kind of module",
        kinds,
        xlabel="simulations trained",
        info=tip("learnkind"),
        note="Time attention, group attention and the output layer, "
        "each in quadrature over its modules.",
        legend=legend,
    )
    return f"<div class=plots3>{level_card}{lora}{kind_card}</div>"
