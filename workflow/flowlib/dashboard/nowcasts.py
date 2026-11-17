"""Nowcast panels by checkpoint, for any configured dataset.

A dataset's forecasts are one folder per label (`reference`,
`reference-complete`, or a checkpoint such as `slots-<hash>`), each with
one file per nowcast date. A `quantiles` dataset's rows carry
`quantile`, `value`, `truth` and optionally `target` and `score`; a
`samples` dataset's rows carry `sample` and `value`, and are summarised
to quantile files first (`summarise_samples`), with the truth and sample
CRPS the checkpoint scoring uses (`evaluation.nowcast.rolling_truth`,
`reporting_delay`, `evaluation.scoring.crps_sample`). A row with no
target is the 7-day target.

Every row goes to the configured reference it matches, else, in a
reference folder, to its model by name (`model:<name>`, such as a hub
team), else to the checkpoint of a folder a run claims; excluded and
unclaimed checkpoints are left out. The page draws `levels`; coverage,
bias and the WIS use the seven cached levels.
"""

from __future__ import annotations

import csv
import gzip
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from bdbvhubrenewal.evaluation.nowcast import reporting_delay, rolling_truth
from bdbvhubrenewal.evaluation.scoring import crps_sample, wis_quantile

from .rows import num, sources
from .runs import Classifier, Reference, Run, shade, split_label

SEVEN = ("0.025", "0.1", "0.25", "0.5", "0.75", "0.9", "0.975")
"""The cached quantile levels."""

LEVELS = tuple(float(q) for q in SEVEN)

PAGE_LEVELS = (0.025, 0.25, 0.5, 0.75, 0.975)
"""The levels drawn when none are given; `paperstyle.LEVELS` in use."""

REFERENCE = "reference"
"""Reference folders start with this (`reference`, `reference-complete`)."""

TARGET = "7day"
"""The target of rows that name none (files scored before the column)."""


def wis(q: Sequence[float], y: float) -> float:
    """Return the WIS of the seven cached quantiles at truth `y`."""
    return float(wis_quantile([y], [list(q)], list(LEVELS))["wis"][0])


def cdf_at(q: Sequence[float], y: float) -> float:
    """Return the forecast CDF at `y`, linear between the seven quantiles.

    Below the lowest quantile it is half its level, above the highest
    half the remaining tail.
    """
    if y <= q[0]:
        return LEVELS[0] / 2
    if y >= q[-1]:
        return 1 - (1 - LEVELS[-1]) / 2
    for i in range(len(q) - 1):
        if q[i] <= y <= q[i + 1]:
            if q[i + 1] == q[i]:
                return (LEVELS[i] + LEVELS[i + 1]) / 2
            return LEVELS[i] + (LEVELS[i + 1] - LEVELS[i]) * (y - q[i]) / (
                q[i + 1] - q[i]
            )
    return 0.5


def stats(pairs: Sequence[tuple[Sequence[float], float]]) -> list[float]:
    """Return 50% and 95% coverage and the mean bias, 1 - 2 F(y)."""
    n = len(pairs)
    return [
        round(sum(q[2] <= y <= q[4] for q, y in pairs) / n, 3),
        round(sum(q[0] <= y <= q[6] for q, y in pairs) / n, 3),
        round(sum(1 - 2 * cdf_at(q, y) for q, y in pairs) / n, 3),
    ]


def is_reference(folder: str) -> bool:
    """Return whether a cache folder holds reference forecasts."""
    return folder.startswith(REFERENCE)


def read_file(
    path: Path, folder: str, which: Classifier
) -> dict[tuple[str, str, str, str], dict[str, dict[str, Any]]]:
    """Return a quantile file's rows by (location, series, source, target)."""
    ref_folder = is_reference(folder)
    checkpoint = "zero-shot" if ref_folder else folder
    run_hash = None if ref_folder else split_label(folder)[0]
    refs = {r.key for r in which.references}
    by_model: dict[str, str | None] = {}
    out: dict[tuple[str, str, str, str], dict[str, dict[str, Any]]] = {}
    with gzip.open(path, "rt") as f:
        for r in csv.DictReader(f):
            model = r["model"]
            if model not in by_model:
                series = which.series(
                    {"model": model, "checkpoint": checkpoint}
                )
                if series is None and ref_folder:
                    series = f"model:{model}"
                elif series is not None and series not in refs:
                    series = run_hash
                by_model[model] = series
            series = by_model[model]
            value = num(r.get("value"))
            if (
                series is None
                or r.get("quantile") not in SEVEN
                or value is None
            ):
                continue
            target = (r.get("target") or "").strip() or TARGET
            for src in sources(model):
                entry = out.setdefault(
                    (r["location"], series, src, target), {}
                ).setdefault(
                    r["target_end_date"],
                    {"q": {}, "y": None, "v": "", "s": None, "l": None},
                )
                entry["q"][r["quantile"]] = value
                y = num(r.get("truth"))
                if y is not None:
                    entry["y"], entry["v"] = y, r.get("truth_vintage") or ""
                entry["s"] = num(r.get("score"))
                entry["l"] = num(r.get("latest"))
    return out


def digest(
    rows: Mapping[tuple[str, str, str, str], Mapping[str, Mapping[str, Any]]],
    picks: Sequence[int],
) -> list[tuple]:
    """Return one file's series, stats, score, truth and latest by key.

    Kept with the file (`build`'s `files`), so a rebuild after a new
    file only assembles these.
    """
    out = []
    for (loc, sid, src, tgt), dates in rows.items():
        times, drawn, scored = [], [], []
        seen: dict[str, tuple[str, float]] = {}
        last: dict[str, float] = {}
        for day in sorted(dates):
            e = dates[day]
            if not all(k in e["q"] for k in SEVEN):
                continue
            seven = [e["q"][k] for k in SEVEN]
            times.append(day)
            drawn.append([round(seven[i], 1) for i in picks])
            if e["y"] is not None:
                scored.append((seven, e["y"], e["s"]))
                seen[day] = (e["v"], e["y"])
            if e["l"] is not None:
                last[day] = e["l"]
        entry: dict[str, Any] | None = None
        if times:
            entry = {"series": {"t": times, "q": drawn}}
            if scored:
                entry["stats"] = stats([(q, y) for q, y, _ in scored])
                entry["score"] = round(
                    sum(s if s is not None else wis(q, y) for q, y, s in scored)
                    / len(scored),
                    2,
                )
        out.append((loc, sid, src, tgt, entry, seen, last))
    return out


def _put(d: dict, keys: Sequence[str], value: Any) -> None:
    for k in keys[:-1]:
        d = d.setdefault(k, {})
    d[keys[-1]] = value


def build(
    cache: Path | str,
    run_list: Sequence[Run],
    references: Sequence[Reference],
    levels: Sequence[float] = PAGE_LEVELS,
    excluded: Iterable[str] = (),
    files: dict | None = None,
    order: Sequence[tuple[str, str]] = (),
) -> dict[str, Any] | None:
    """Return every location's nowcasts by series, target, source, origin.

    `files` keeps each file's computed rows (`digest`) by its time and
    size, so a rebuild reads and computes only new or changed files.
    `order` is the location groups, (name, regex), locations are sorted
    by. Returns None if the folder does not exist.
    """
    import re

    cache = Path(cache)
    if not cache.is_dir():
        return None
    files = {} if files is None else files
    which = Classifier(run_list, references, excluded)
    picks = [SEVEN.index(f"{lv:g}") for lv in levels]
    series: dict = {}
    score: dict = {}
    stat: dict = {}
    truth: dict = {}
    latest: dict = {}
    origins: set[str] = set()
    targets: set[str] = set()
    for folder in sorted(p for p in cache.iterdir() if p.is_dir()):
        ref = is_reference(folder.name)
        if not ref and which.run_of(folder.name) is None:
            continue
        for path in sorted(folder.glob("*.csv.gz")):
            origin = path.name.removesuffix(".csv.gz")
            if ref:
                origins.add(origin)
            try:
                st = path.stat()
            except OSError:
                continue
            key = (st.st_mtime_ns, st.st_size, tuple(picks))
            if str(path) not in files or files[str(path)][0] != key:
                rows = read_file(path, folder.name, which)
                files[str(path)] = (key, digest(rows, picks))
            for loc, sid, src, tgt, entry, seen, last in files[str(path)][1]:
                targets.add(tgt)
                for day, (vintage, y) in seen.items():
                    known = truth.setdefault(loc, {}).setdefault(tgt, {})
                    if day not in known or vintage >= known[day][0]:
                        known[day] = (vintage, y)
                for day, v in last.items():
                    _put(latest, [loc, tgt, day], v)
                if entry is None:
                    continue
                keys = [loc, sid, tgt, src, origin]
                _put(series, keys, entry["series"])
                if "stats" in entry:
                    _put(stat, keys, entry["stats"])
                    _put(score, keys, entry["score"])

    def rank(loc: str) -> tuple[int, str]:
        hit = (i for i, (_, rx) in enumerate(order) if re.match(rx, loc))
        return next(hit, len(order)), loc

    if not origins:
        origins = {
            o
            for loc in series.values()
            for s in loc.values()
            for t in s.values()
            for src in t.values()
            for o in src
        }
    return {
        "series": series,
        "score": score,
        "stats": stat,
        "truth": truth,
        "latest": latest,
        "sets": sorted({s for loc in series.values() for s in loc}),
        "locs": sorted(series, key=rank),
        "targets": sorted(targets),
        "origins": sorted(origins),
        "folders": {
            split_label(p.name)[0]: p.name
            for p in cache.iterdir()
            if p.is_dir() and not is_reference(p.name)
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
    excluded: Iterable[str] = (),
    order: Sequence[tuple[str, str]] = (),
) -> dict[str, Any] | None:
    """Return `build`'s result, reused while no file of the folder changes."""
    cache = Path(cache)
    sig = (_signature(cache) if cache.is_dir() else None, tuple(levels))
    if store.get("sig") != sig or "built" not in store:
        store["built"] = build(
            cache,
            run_list,
            references,
            levels,
            excluded,
            store.setdefault("files", {}),
            order,
        )
        store["sig"] = sig
    return store["built"]


def location(built: Mapping[str, Any] | None, loc: str) -> dict[str, Any]:
    """Return one location's page data: series, scores, stats, truth."""
    if not built or loc not in built["series"]:
        return {}

    def line(by_target: Mapping[str, Mapping[str, Any]], pick=lambda v: v):
        return {
            tgt: {"t": sorted(d), "y": [pick(d[k]) for k in sorted(d)]}
            for tgt, d in by_target.items()
        }

    return {
        "series": built["series"][loc],
        "score": built["score"].get(loc, {}),
        "stats": built["stats"].get(loc, {}),
        "truth": line(built["truth"].get(loc, {}), lambda v: v[1]),
        "latest": line(built["latest"].get(loc, {})),
    }


def meta(
    built: Mapping[str, Any],
    run_list: Sequence[Run],
    references: Sequence[Reference],
    sizes: Mapping[str, int],
    excluded: Iterable[str] = (),
) -> list[dict[str, Any]]:
    """Return each series' label, run, size and colour, in page order.

    References first, in their configured order, then other reference
    models by name, then each run's checkpoints by size, shaded light to
    dark when the run has a hue.
    """
    which = Classifier(run_list, references, excluded)
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
        elif s.startswith("model:"):
            out.append(
                {
                    "id": s,
                    "label": s.removeprefix("model:"),
                    "run": "ref",
                    "n": 1,
                    "color": "#6b7280",
                }
            )
        else:
            run = which.run_of(built["folders"].get(s, s))
            if run is None:
                continue
            n = sizes.get(s) or by_key[run].n.get(s) or 0
            label = (
                f"{by_key[run].label} n{n} {s}"
                if n
                else (f"{by_key[run].label} {s}")
            )
            out.append({"id": s, "label": label, "run": run, "n": n})
    ref_rank = {k: i for i, k in enumerate(refs)}
    run_rank = {r.key: i for i, r in enumerate(run_list)}
    out.sort(
        key=lambda m: (
            (0, ref_rank.get(m["id"], len(ref_rank)), m["n"], m["id"])
            if m["run"] == "ref"
            else (1, run_rank[m["run"]], m["n"], m["id"])
        )
    )
    for run in run_list:
        group = [m for m in out if m["run"] == run.key]
        for i, m in enumerate(group):
            m["color"] = shade(run, i, len(group))
    return out


def summarise_samples(
    raw: Path | str,
    out: Path | str,
    truth_path: Path | str,
    ds: Mapping[str, Any],
) -> int | None:
    """Write quantile files from a samples dataset's files.

    Each `raw/<label>/<origin>.csv.gz` newer than its summary or the
    truth table becomes `out/<label>/<origin>.csv.gz`, with the seven
    quantiles, the truth (each week's count in the first release `D`
    days after it, `D` the age by which `truth_level` of the `national`
    count is reported, leaving out `truth_skip` releases), the sample
    CRPS and the latest release's count.

    Returns
    -------
    int or None
        The truth delay `D` in days, or None if none can be set.
    """
    raw, out, truth_path = Path(raw), Path(out), Path(truth_path)
    target = pd.read_csv(truth_path, dtype={"kind": str})
    real = target.loc[target["kind"] == "real"].copy()
    delay = reporting_delay(
        real,
        float(ds.get("truth_level", 0.98)),
        tuple(ds.get("truth_skip") or ()),
        ds.get("national", "CD"),
    )
    truth = (
        rolling_truth(real, delay).set_index(["location", "target_end_date"])
        if delay is not None
        else pd.DataFrame(columns=["truth", "truth_vintage"])
    )
    last = real.loc[real["vintage"] == real["vintage"].max()]
    latest = last.set_index(["location", "target_end_date"])["total"]
    t_mt = truth_path.stat().st_mtime
    wanted = set(ds.get("targets") or ())
    for f in sorted(raw.glob("*/*.csv.gz")):
        dst = out / f.parent.name / f.name
        if dst.exists() and dst.stat().st_mtime >= max(f.stat().st_mtime, t_mt):
            continue
        frame = pd.read_csv(f)
        if "target" not in frame:
            frame["target"] = TARGET
        frame["target"] = frame["target"].fillna(TARGET)
        if wanted:
            frame = frame.loc[frame["target"].isin(wanted)]
        rows = []
        keys = ["model", "origin", "location", "target_end_date", "target"]
        for (model, origin, loc, week, tgt), g in frame.groupby(keys):
            x = np.sort(g["value"].to_numpy(float))
            hit = (loc, week) in truth.index
            y = float(truth.loc[(loc, week), "truth"]) if hit else None
            vintage = truth.loc[(loc, week), "truth_vintage"] if hit else ""
            sc = "" if y is None else round(crps_sample(y, x), 3)
            lt = latest.get((loc, week), "")
            for q in SEVEN:
                rows.append(
                    [
                        model,
                        origin,
                        loc,
                        week,
                        q,
                        round(float(np.quantile(x, float(q))), 2),
                        "" if y is None else y,
                        vintage,
                        tgt,
                        sc,
                        lt,
                    ]
                )
        dst.parent.mkdir(parents=True, exist_ok=True)
        tmp = dst.with_suffix(".tmp")
        with gzip.open(tmp, "wt", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(
                [
                    "model",
                    "origin",
                    "location",
                    "target_end_date",
                    "quantile",
                    "value",
                    "truth",
                    "truth_vintage",
                    "target",
                    "score",
                    "latest",
                ]
            )
            w.writerows(rows)
        tmp.replace(dst)
    return delay
