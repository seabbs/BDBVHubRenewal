"""Checkpoint scores for the dashboard's score panels.

Rows are the checkpoint score table (`<SCORE_CACHE>/tables/
checkpoint-scores.csv`, or its copy on the evidence branch) under one
scoring rule: relative WIS for Germany, relative CRPS for the BDBV
backtest, and the mean WIS with its components for the WIS bars. Each
row goes to the configured run or reference it belongs to
(`runs.Classifier`); rows no run or reference claims are left out.
"""

from __future__ import annotations

import csv
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

from .paperstyle import WIS_PARTS
from .rows import num, source
from .runs import Classifier, Reference, Run, split_label

VALUE = {"rel_wis": "wis", "rel_crps": "crps"}
"""The relative score columns and the metric each gives."""

AXIS0 = "zero-shot"
"""The label of the checkpoint axis' left end, before any training."""


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
    view: Mapping[str, Any] | None = None,
    excluded: Iterable[str] = (),
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
    view
        `studies` (study -> label, in selector order), `levels` (their
        order) and `defaults` (the target, metric, source and level shown
        first), from the `dashboard` config.
    excluded
        Checkpoint hashes no panel shows.

    Returns
    -------
    dict
        `rows` as [study, level, source, series, size, hash, value,
        target, metric]: the source is the row's layout when it has one
        (held-out synthetic scores), else daily or weekly from the model;
        a row with no target is the 7-day target. Also, by study, the
        `targets`, `metrics`, `levels` and run `variants` (sources), each
        source's base source (`vbase`) and what a study's sources are
        called (`vlabel`); `pending`, the `series`, their colours, the
        series shown first and the reference `roles`.
    """
    view = view or {}
    study_order = list((view.get("studies") or {}).keys())
    level_order = list(view.get("levels") or [])
    which = Classifier(run_list, references, excluded)
    run_keys = {r.key for r in run_list}
    out: list[list[Any]] = []
    seen_hash: set[str] = set()
    seen_n: dict[str, set[int]] = {}
    targets: dict[str, set[str]] = {}
    metrics: dict[str, set[str]] = {}
    levels: dict[str, set[str]] = {}
    variants: dict[str, set[str]] = {}
    vbase: dict[str, str] = {}
    vlabel: dict[str, str] = {}
    for r in rows:
        if r.get("rule") != rule:
            continue
        series = which.series(r)
        if series is None:
            continue
        layout = (r.get("layout") or "").strip()
        base = source(r.get("model", ""))
        src = layout or base
        if src is None:
            continue
        study = r.get("study", "")
        checkpoint = r.get("checkpoint", "")
        key = "" if checkpoint == "zero-shot" else split_label(checkpoint)[0]
        n = int(num(r.get("outbreaks")) or 0)
        # Rows scored before the target column hold the 7-day target.
        target = (r.get("target") or "").strip() or "7day"
        for column, metric in VALUE.items():
            value = num(r.get(column))
            if value is None:
                continue
            out.append(
                [
                    study,
                    r.get("level", ""),
                    src,
                    series,
                    n,
                    key,
                    round(value, 4),
                    target,
                    metric,
                ]
            )
            targets.setdefault(study, set()).add(target)
            metrics.setdefault(study, set()).add(metric)
            levels.setdefault(study, set()).add(r.get("level", ""))
            vlabel[study] = "layout" if layout else vlabel.get(study, "source")
            if series in run_keys:
                variants.setdefault(study, set()).add(src)
            vbase[src] = base or src
            seen_hash.add(key)
            seen_n.setdefault(series, set()).add(n)
    pending = [
        {"run": run, "n": n, "hash": h}
        for run, n, h in local
        if h not in which.excluded
        and not ((h in seen_hash) if h else (n in seen_n.get(run, set())))
    ]
    series_list = [
        {"key": r.key, "label": r.label, "run": True, "color": r.colour}
        for r in run_list
    ] + [
        {"key": r.key, "label": r.label, "color": r.colour} for r in references
    ]
    live = next((r.key for r in run_list if r.live), None)
    first = next((r.key for r in run_list if r.key in seen_n), live)

    def rank(order: list[str]):
        return lambda x: (order.index(x) if x in order else len(order), x)

    return {
        "rows": out,
        "pending": pending,
        "series": series_list,
        "col": {s["key"]: s["color"] for s in series_list},
        "defaults": [k for k in [first, *(r.key for r in references)] if k],
        "roles": {r.key: r.role for r in references},
        "targets": {k: sorted(v) for k, v in targets.items()},
        "metrics": {k: sorted(v) for k, v in metrics.items()},
        "levels": {
            k: sorted(v, key=rank(level_order)) for k, v in levels.items()
        },
        "variants": {k: sorted(v) for k, v in variants.items()},
        "vbase": vbase,
        "vlabel": vlabel,
        "studies": sorted({r[0] for r in out}, key=rank(study_order)),
        "studylab": dict(view.get("studies") or {}),
        "view_defaults": dict(view.get("defaults") or {}),
        "axis0": AXIS0,
    }


def wis_bars(
    rows: Iterable[Mapping[str, str]],
    run_list: Sequence[Run],
    references: Sequence[Reference],
    rule: str,
    source: str | None = None,
    excluded: Iterable[str] = (),
) -> dict[tuple[str, str], list[dict[str, Any]]]:
    """Return each study and level's mean WIS bars, with its components.

    Only the rule's 7-day rows with every component (`WIS_PARTS`) are
    drawn; rows of a held-out layout are left out, as are a run's rows
    of a source other than `source`. Bars are each run's checkpoints by
    size, then the references in their configured order.

    Returns
    -------
    dict
        (study, level) -> bars, each with `series`, `label`, `colour`,
        `n`, `wis` and `parts` (component -> value).
    """
    which = Classifier(run_list, references, excluded)
    order = [r.key for r in run_list] + [r.key for r in references]
    named: dict[str, tuple[str, str]] = {
        r.key: (r.label, r.colour) for r in run_list
    } | {r.key: (r.label, r.colour) for r in references}
    runs_ = {r.key for r in run_list}
    out: dict[tuple[str, str], dict[tuple[str, int], dict[str, Any]]] = {}
    for r in rows:
        if r.get("rule") != rule or (r.get("layout") or "").strip():
            continue
        if ((r.get("target") or "").strip() or "7day") != "7day":
            continue
        series = which.series(r)
        if series is None:
            continue
        if series in runs_ and source and _source_of(r) not in (None, source):
            continue
        parts = {p: num(r.get(p)) for p in WIS_PARTS}
        total = num(r.get("wis"))
        if total is None or any(v is None for v in parts.values()):
            continue
        n = int(num(r.get("outbreaks")) or 0) if series in runs_ else 0
        label, colour = named[series]
        out.setdefault((r.get("study", ""), r.get("level", "")), {})[
            (series, n)
        ] = {
            "series": series,
            "label": f"{label} n{n}" if series in runs_ else label,
            "colour": colour,
            "n": n,
            "wis": round(total, 4),
            "parts": {p: round(v or 0.0, 4) for p, v in parts.items()},
        }
    return {
        key: [
            bars[k]
            for k in sorted(bars, key=lambda k: (order.index(k[0]), k[1]))
        ]
        for key, bars in out.items()
    }


def _source_of(row: Mapping[str, str]) -> str | None:
    """Return a score row's source from its model (daily or weekly)."""
    return source(row.get("model", ""))
