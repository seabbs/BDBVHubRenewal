"""German nowcasts by checkpoint, for the dashboard.

The checkpoint scoring caches one file per nowcast date for each label
(`<SCORE_CACHE>/germany/<label>/<date>.csv.gz`: model, location,
target_end_date, quantile, value, truth, truth_vintage). `reference`
holds the zero-shot models and randomwalknowcast, each other folder one
checkpoint's fine-tuned models. Every row goes to the configured
reference it matches or the run its folder belongs to; others are left
out. The page draws the paper panels' levels (`paperstyle.LEVELS`), and
the mean WIS uses the seven levels the cache holds.
"""

from __future__ import annotations

import csv
import gzip
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from .runs import Classifier, Reference, Run, split_label

WIS_LEVELS = ("0.025", "0.1", "0.25", "0.5", "0.75", "0.9", "0.975")
"""The cached quantile levels: the median and the 50%, 80% and 95%
intervals the mean WIS is taken over."""

PAGE_LEVELS = (0.025, 0.25, 0.5, 0.75, 0.975)
"""The levels drawn when none are given; `paperstyle.LEVELS` in use."""

REFERENCE = "reference"
"""The cache folders of the references start with this (`reference`,
`reference-complete`)."""


def wis(q: Sequence[float], y: float) -> float:
    """Return the WIS of the seven `WIS_LEVELS` quantiles at truth `y`."""
    total = 0.5 * abs(y - q[3])
    for alpha, lo, hi in (
        (0.05, q[0], q[6]),
        (0.2, q[1], q[5]),
        (0.5, q[2], q[4]),
    ):
        total += (
            alpha
            / 2
            * (
                (hi - lo)
                + 2 / alpha * max(lo - y, 0)
                + 2 / alpha * max(y - hi, 0)
            )
        )
    return total / 3.5


def _num(x: Any) -> float | None:
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def _sources(model: str) -> list[str]:
    if "daily" in model:
        return ["daily"]
    if "weekly" in model:
        return ["weekly"]
    # A model with no source (randomwalknowcast) is drawn on both.
    return ["daily", "weekly"]


def read_file(
    path: Path, folder: str, which: Classifier
) -> dict[tuple[str, str, str], dict[str, dict[str, Any]]]:
    """Return a cached file's quantiles by (location, series, source).

    The series is the reference a row matches, else the checkpoint hash
    of a folder a run claims; rows of neither are left out.
    """
    is_ref = folder.startswith(REFERENCE)
    checkpoint = "zero-shot" if is_ref else folder
    run_hash = None if is_ref else split_label(folder)[0]
    out: dict[tuple[str, str, str], dict[str, dict[str, Any]]] = {}
    with gzip.open(path, "rt") as f:
        for r in csv.DictReader(f):
            row = {"model": r["model"], "checkpoint": checkpoint}
            series = which.series(row)
            if series is None:
                continue
            if series not in {ref.key for ref in which.references}:
                series = run_hash
            value = _num(r["value"])
            if r["quantile"] not in WIS_LEVELS or value is None:
                continue
            for source in _sources(r["model"]):
                entry = out.setdefault(
                    (r["location"], series, source), {}
                ).setdefault(
                    r["target_end_date"], {"q": {}, "y": None, "v": ""}
                )
                entry["q"][r["quantile"]] = value
                y = _num(r.get("truth"))
                if y is not None:
                    entry["y"], entry["v"] = y, r.get("truth_vintage") or ""
    return out


def _order(location: str) -> tuple[int, str]:
    rank = 0 if location == "DE" else 1 if location.startswith("DE-") else 2
    return rank, location


def build(
    cache: Path | str,
    run_list: Sequence[Run],
    references: Sequence[Reference],
    levels: Sequence[float] = PAGE_LEVELS,
) -> dict[str, Any] | None:
    """Return every location's nowcasts by series, source and origin.

    Returns None if the cache folder does not exist.
    """
    cache = Path(cache)
    if not cache.is_dir():
        return None
    which = Classifier(run_list, references)
    picks = [WIS_LEVELS.index(f"{lv:g}") for lv in levels]
    series: dict[str, Any] = {}
    scores: dict[str, Any] = {}
    truth: dict[str, dict[str, tuple[str, float]]] = {}
    origins: set[str] = set()
    for folder in sorted(p for p in cache.iterdir() if p.is_dir()):
        is_ref = folder.name.startswith(REFERENCE)
        if not is_ref and which.run_of(folder.name) is None:
            continue
        for path in sorted(folder.glob("*.csv.gz")):
            origin = path.name.removesuffix(".csv.gz")
            if is_ref:
                origins.add(origin)
            for (loc, sid, source), dates in read_file(
                path, folder.name, which
            ).items():
                times, quantiles, ws = [], [], []
                for day in sorted(dates):
                    e = dates[day]
                    if not all(k in e["q"] for k in WIS_LEVELS):
                        continue
                    seven = [e["q"][k] for k in WIS_LEVELS]
                    times.append(day)
                    quantiles.append([round(seven[i], 1) for i in picks])
                    if e["y"] is not None:
                        ws.append(wis(seven, e["y"]))
                        seen = truth.setdefault(loc, {})
                        if day not in seen or e["v"] >= seen[day][0]:
                            seen[day] = (e["v"], e["y"])
                if not times:
                    continue
                series.setdefault(loc, {}).setdefault(sid, {}).setdefault(
                    source, {}
                )[origin] = {"t": times, "q": quantiles}
                if ws:
                    scores.setdefault(loc, {}).setdefault(sid, {}).setdefault(
                        source, {}
                    )[origin] = round(sum(ws) / len(ws), 2)
    sets = sorted({s for loc in series.values() for s in loc})
    if not origins:
        origins = {
            o
            for loc in series.values()
            for s in loc.values()
            for src in s.values()
            for o in src
        }
    return {
        "series": series,
        "wis": scores,
        "truth": truth,
        "sets": sets,
        "locs": sorted(series, key=_order),
        "origins": sorted(origins),
        "folders": {
            split_label(p.name)[0]: p.name
            for p in cache.iterdir()
            if p.is_dir() and not p.name.startswith(REFERENCE)
        },
    }


def _signature(cache: Path) -> tuple:
    out = []
    for f in sorted(cache.glob("*/*.csv.gz")):
        try:
            st = f.stat()
        except OSError:
            continue
        out.append((str(f), st.st_mtime_ns, st.st_size))
    return tuple(out)


def cached_build(
    cache: Path | str,
    run_list: Sequence[Run],
    references: Sequence[Reference],
    store: dict,
    levels: Sequence[float] = PAGE_LEVELS,
) -> dict[str, Any] | None:
    """Return `build`'s result, reused while no cache file changes.

    `store` keeps the last build and the cache files' paths, times and
    sizes it was built from.
    """
    cache = Path(cache)
    sig = (_signature(cache) if cache.is_dir() else None, tuple(levels))
    if store.get("sig") != sig or "built" not in store:
        store["built"] = build(cache, run_list, references, levels)
        store["sig"] = sig
    return store["built"]


def location(built: Mapping[str, Any] | None, loc: str) -> dict[str, Any]:
    """Return one location's page data: series, mean WIS and truth."""
    if not built or loc not in built["series"]:
        return {}
    seen = sorted(built["truth"].get(loc, {}).items())
    return {
        "series": built["series"][loc],
        "wis": built["wis"].get(loc, {}),
        "truth": {"t": [d for d, _ in seen], "y": [v[1] for _, v in seen]},
    }


def meta(
    built: Mapping[str, Any],
    run_list: Sequence[Run],
    references: Sequence[Reference],
    sizes: Mapping[str, int],
) -> list[dict[str, Any]]:
    """Return each series' label, run, size and colour, in page order.

    References first, in their configured order, then each run's
    checkpoints by size. `sizes` gives checkpoint hashes' sizes the run
    configuration does not.
    """
    which = Classifier(run_list, references)
    refs = {r.key: r for r in references}
    by_key = {r.key: r for r in run_list}
    out = []
    for s in built["sets"]:
        if s in refs:
            out.append(
                {
                    "id": s,
                    "label": refs[s].label,
                    "run": "ref",
                    "n": 0,
                    "color": refs[s].colour,
                }
            )
            continue
        run = which.run_of(built["folders"].get(s, s))
        if run is None:
            continue
        cfg = by_key[run]
        n = sizes.get(s) or cfg.n.get(s) or 0
        label = f"{cfg.label} n{n} {s}" if n else f"{cfg.label} {s}"
        out.append(
            {"id": s, "label": label, "run": run, "n": n, "color": cfg.colour}
        )
    ref_rank = {k: i for i, k in enumerate(refs)}
    run_rank = {r.key: i for i, r in enumerate(run_list)}
    out.sort(
        key=lambda m: (
            (0, ref_rank[m["id"]], 0)
            if m["run"] == "ref"
            else (1, run_rank[m["run"]], m["n"])
        )
    )
    return out
