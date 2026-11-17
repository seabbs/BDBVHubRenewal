"""Checkpoint scores for the dashboard's score panels.

Rows are the checkpoint score table (`<SCORE_CACHE>/tables/
checkpoint-scores.csv`, or its copy on the evidence branch) under one
scoring rule: relative WIS for Germany, relative CRPS for the BDBV
backtest. Each row goes to the configured run or reference it belongs to
(`runs.Classifier`); rows no run or reference claims are left out.
"""

from __future__ import annotations

import csv
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

from .rows import num, source
from .runs import Classifier, Reference, Run, split_label

VALUE = {"germany": "rel_wis", "ebola": "rel_crps"}
"""The relative score each study's panel shows."""

TICKS = [
    (125, "zero-shot"),
    (250, "250"),
    (500, "500"),
    (1000, "1000"),
    (2000, "2000"),
    (4000, "4000"),
    (8000, "8000"),
]
"""The x positions and labels of the checkpoint axis; zero-shot at 125."""


def read(path: Path | str) -> list[dict[str, str]]:
    """Return a score table's rows, or none if there is no table yet.

    Any other error reading it, such as a folder in its place, raises.
    """
    try:
        with Path(path).open() as f:
            return list(csv.DictReader(f))
    except FileNotFoundError:
        return []


def payload(
    rows: Iterable[Mapping[str, str]],
    run_list: Sequence[Run],
    references: Sequence[Reference],
    rule: str,
    local: Iterable[tuple[str, int, str]],
) -> dict[str, Any]:
    """Return the score panels' data.

    Parameters
    ----------
    rows
        Score table rows.
    run_list, references
        The configured runs and references.
    rule
        The scoring rule whose rows are shown.
    local
        Each saved checkpoint as (run, size, hash), so those not yet
        scored are marked pending.

    Returns
    -------
    dict
        `rows` as [study, level, source, series, size, hash, value],
        `pending`, the axis `xt`, the `series` with their colours, `col`
        and the `defaults` shown first.
    """
    which = Classifier(run_list, references)
    out: list[list[Any]] = []
    seen_hash: set[str] = set()
    seen_n: dict[str, set[int]] = {}
    for r in rows:
        study = r.get("study", "")
        if r.get("rule") != rule or study not in VALUE:
            continue
        series = which.series(r)
        value = num(r.get(VALUE[study]))
        src = source(r.get("model", ""))
        if series is None or value is None or src is None:
            continue
        checkpoint = r.get("checkpoint", "")
        key = "" if checkpoint == "zero-shot" else split_label(checkpoint)[0]
        n = int(num(r.get("outbreaks")) or 0)
        out.append(
            [study, r.get("level", ""), src, series, n, key, round(value, 4)]
        )
        seen_hash.add(key)
        seen_n.setdefault(series, set()).add(n)
    pending = [
        {"run": run, "n": n, "hash": h}
        for run, n, h in local
        if not ((h in seen_hash) if h else (n in seen_n.get(run, set())))
    ]
    series_list = [
        {"key": r.key, "label": r.label, "run": True, "color": r.colour}
        for r in run_list
    ] + [
        {"key": r.key, "label": r.label, "color": r.colour} for r in references
    ]
    live = next((r.key for r in run_list if r.live), None)
    first = next((r.key for r in run_list if r.key in seen_n), live)
    return {
        "rows": out,
        "pending": pending,
        "series": series_list,
        "col": {s["key"]: s["color"] for s in series_list},
        "defaults": [k for k in [first, *(r.key for r in references)] if k],
        "roles": {r.key: r.role for r in references},
    }
