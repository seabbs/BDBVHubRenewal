# Copyright (c) 2026 Sam Abbott. MIT licence, see LICENSE.
"""Backtest scores over time, one point per deploy of the site from main.

The gh-pages branch keeps every deploy. A deploy from main is the commit
`Deploying to gh-pages from @ <repo>@<sha>`, so each one maps to the
main commit it was built from. Pull request previews are left out. Each
deploy's `figures/scoresummary-backtest-<target>.json` holds the
per-forecast scores the Backtest page summarises. This script summarises
them as the page's overall view does (as `relative_skill`, over the
forecasts the baseline also made), by baseline, target week, pool and
model,
over all forecasts and over the common forecasts only (`_common`), and
writes one row per deploy and each of those.

A commit can deploy more than once: the fit store reruns the build when it
stores new fits, and every build refreshes the hub data. Scores can
therefore move without a code change. `break` is true where the origins,
releases or number of forecasts scored for a model differ from its
previous deploy, so a jump there is not a model effect. A change in the
baseline itself shows in `base_wis`. Deploys before 2026-11-09 have one summary,
`scoresummary-backtest.json`, scored on each later release as reported,
which is read as the `latest` target.

Usage::

    git fetch origin gh-pages
    uv run python scripts/score_history.py [--ref origin/gh-pages]
        [--out results/score-history.csv]

With `--docs` and `--figures` it writes the Score history page's snippet
(`score-history.md`) and chart data (`scorehistory-<target>.json`, drawn
by `docs/javascripts/checkschart.js`), as the Pages build does. With
`--current`, this build's own summaries are the last point. Without the
branch the snippet says the history is unavailable. It imports pandas
only, not the package, so the Pages step stays fast.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import re
import subprocess
import sys
from collections.abc import Iterable, Iterator, Mapping, Sequence
from pathlib import Path
from typing import Any

import pandas as pd

DEPLOY = re.compile(r"^Deploying to gh-pages from @ \S+@([0-9a-f]{40})\b")
"""A deploy from main, and the main commit it was built from."""

FILES: Mapping[str, str] = {
    "figures/scoresummary-backtest-latest.json": "latest",
    "figures/scoresummary-backtest-complete.json": "complete",
    "figures/scoresummary-backtest.json": "latest",
}
"""Summary file and the target it scores."""

SKIP_MODELS = frozenset({"as reported"})
"""Rows in older summaries that are not models."""

MATCH = ("origin", "release", "place", "week")

Deploy = tuple[str, str, str, str, Mapping[str, Any]]
"""Source commit, deploy time, gh-pages commit, target and summary."""


def source_sha(message: str) -> str | None:
    """Return the main commit a deploy was built from, or None."""
    m = DEPLOY.match(message)
    return None if m is None else m.group(1)


def forecasts(summary: Mapping[str, Any]) -> pd.DataFrame:
    """Return a summary's per-forecast scores with their labels.

    Origins, releases, places and weeks are named, not indexed, so
    forecasts match across deploys.
    """
    models = [m["name"] for m in summary["models"]]
    keys = [m["key"] for m in summary["metrics"]]
    width = len(MATCH) + 1 + len(keys)
    rows = pd.DataFrame(
        [r[:width] for r in summary["rows"]],
        columns=["model", *MATCH, *keys],
    )
    names = {
        "model": models,
        "origin": summary["origins"],
        "release": summary["releases"],
        "place": [p["id"] for p in summary["places"]],
        "week": summary["weeks"],
    }
    pools = [p["pool"] for p in summary["places"]]
    rows["pool"] = [pools[i] for i in rows["place"]]
    for col, labels in names.items():
        rows[col] = [labels[i] for i in rows[col]]
    return rows.loc[~rows["model"].isin(SKIP_MODELS)]


def _pairs(
    rows: pd.DataFrame, relative: Sequence[str], baselines: Sequence[str]
) -> pd.DataFrame:
    """Pair each forecast with the baseline's same forecast, per baseline.

    As `relative_skill`, without importing the package (and JAX).
    """
    on = ["deploy", "pool", *MATCH]
    frames = []
    for baseline in baselines:
        base = rows.loc[rows["model"] == baseline, [*on, *relative]]
        frames.append(
            rows.merge(base, on=on, suffixes=("", "_b")).assign(
                baseline=baseline
            )
        )
    return pd.concat(frames, ignore_index=True)


def _common(pairs: pd.DataFrame, by: Sequence[str] = ()) -> pd.Series:
    """Flag the pairs in each model and baseline's common set.

    With `by`, such as `("week",)`, each of its values has its own window
    and set.

    The window runs back from the newest deploy, as far as keeps the most
    forecast-deploys: the forecasts in every deploy of the window times
    its deploys. A deploy that drops most of the common set, such as one
    before a relabel or a refit that adds places, so ends it. The common
    set is the forecasts in every deploy of that window.
    """
    key = (
        pairs["origin"]
        + "|"
        + pairs["release"]
        + "|"
        + pairs["place"]
        + "|"
        + pairs["week"]
    )
    flag = pd.Series(False, index=pairs.index)
    for _, g in pairs.groupby(["baseline", "model", *by], sort=False):
        sets = key[g.index].groupby(g["deploy"]).agg(set)
        shared: set[str] | None = None
        best: tuple[int, list[int], set[str]] = (0, [], set())
        window: list[int] = []
        for deploy, keys in sets.sort_index(ascending=False).items():
            shared = keys if shared is None else shared & keys
            if not shared:
                break
            window = [*window, int(str(deploy))]
            if len(shared) * len(window) > best[0]:
                best = (len(shared) * len(window), window, shared)
        inside = g["deploy"].isin(best[1]) & key[g.index].isin(best[2])
        flag[g.index[inside.to_numpy()]] = True
    return flag


def _summarise(
    pairs: pd.DataFrame,
    by: Sequence[str],
    keys: Sequence[str],
    relative: Sequence[str],
) -> pd.DataFrame:
    """Mean of each metric and, for relative ones, over the baseline's."""
    grp = pairs.groupby([*by, "pool", "model"], sort=False)
    out = grp[list(keys)].mean()
    out.insert(0, "n", grp.size())
    for k in relative:
        b = grp[f"{k}_b"].mean()
        out[f"base_{k}"] = b
        out[f"rel_{k}"] = out[k] / b.where(b > 0)
    rel = [f"rel_{k}" for k in relative]
    base = [f"base_{k}" for k in relative]
    return out.reset_index()[[*by, "pool", "model", "n", *keys, *base, *rel]]


def _metrics(summaries: Iterable[Mapping[str, Any]]) -> tuple[list[str], ...]:
    keys: dict[str, bool] = {}
    for s in summaries:
        for m in s["metrics"]:
            keys[m["key"]] = keys.get(m["key"], False) or m["relative"]
    return list(keys), [k for k, r in keys.items() if r]


def summary_scores(summary: Mapping[str, Any]) -> pd.DataFrame:
    """Summarise one score summary by baseline, pool and model.

    Every model is a baseline, over all forecasts. Returns
    `baseline`, `pool`, `model`, `n` (forecasts the baseline also made),
    the mean of each metric and, for relative metrics, the baseline's mean
    on the same forecasts `base_<metric>` and `rel_<metric>`.
    """
    keys, relative = _metrics([summary])
    rows = forecasts(summary).assign(deploy=0)
    pairs = _pairs(rows, relative, list(dict.fromkeys(rows["model"])))
    return _summarise(pairs, ["baseline"], keys, relative)


def history(
    deploys: Iterable[Deploy], baselines: Sequence[str] | None = None
) -> pd.DataFrame:
    """Summarise each deploy's scores, oldest first, and flag breaks.

    Parameters
    ----------
    deploys : iterable of (str, str, str, str, mapping)
        Source commit, deploy time, gh-pages commit, target and summary,
        oldest first.
    baselines : sequence of str, optional
        Models to divide by. Defaults to every model.

    Returns
    -------
    pandas.DataFrame
        One row per deploy, target, baseline, `split` (`"all"` or
        `"week <target week>"`), `common` (only the
        forecasts in the common set, see `_common`, per target week for a
        week, or all), pool and model: `deploy_time`, `source_sha`,
        `pages_sha`, the `summary_scores` columns, `origins` and
        `releases` (`;`-joined) and `break`. `break` is true where the
        number of forecasts scored against the baseline differs from the
        model's previous deploy of that series, where the model first
        appears or, over all forecasts, where the origins or releases
        differ. A change in the model's or the baseline's forecasts
        changes that number.
    """
    deploys = list(deploys)
    keys, relative = _metrics(d[4] for d in deploys)
    meta = pd.DataFrame(
        [
            {
                "deploy": i,
                "deploy_time": time,
                "source_sha": sha,
                "pages_sha": pages,
                "target": target,
                "origins": ";".join(summary["origins"]),
                "releases": ";".join(summary["releases"]),
            }
            for i, (sha, time, pages, target, summary) in enumerate(deploys)
        ]
    )
    rows = pd.concat(
        [forecasts(d[4]).assign(deploy=i) for i, d in enumerate(deploys)],
        ignore_index=True,
    )
    rows = rows.merge(meta[["deploy", "target"]], on="deploy")
    seen = list(dict.fromkeys(rows["model"]))
    chosen = seen if baselines is None else [b for b in baselines if b in seen]
    pairs = _pairs(rows, relative, chosen)
    parts = []
    for _, g in pairs.groupby("target", sort=False):
        g = g.assign(common=False)
        overall = pd.concat([g, g.loc[_common(g)].assign(common=True)])
        weekly = pd.concat([g, g.loc[_common(g, ["week"])].assign(common=True)])
        g = pd.concat(
            [
                overall.assign(split="all"),
                weekly.assign(split="week " + weekly["week"]),
            ],
            ignore_index=True,
        )
        by = ["deploy", "baseline", "split", "common"]
        parts.append(_summarise(g, by, keys, relative))
    out = pd.concat(parts, ignore_index=True).merge(meta, on="deploy")
    out = out.sort_values("deploy", kind="stable").reset_index(drop=True)
    series = ["target", "baseline", "split", "common", "pool", "model"]
    prev = out.groupby(series)[["origins", "releases", "n"]].shift()
    scope = (prev[["origins", "releases"]] != out[["origins", "releases"]]).any(
        axis=1
    )
    out["break"] = (prev["n"] != out["n"]) | (scope & ~out["common"])
    first = ["deploy_time", "source_sha", "pages_sha", "target"]
    rest = [c for c in out.columns if c not in {*first, "deploy"}]
    return out[[*first, *rest]]


def _git(*args: str) -> str:
    return subprocess.run(
        ["git", *args], capture_output=True, text=True, check=True
    ).stdout


def walk(ref: str) -> list[Deploy]:
    """Return the score summaries of each deploy from main, oldest first.

    Identical summaries are read once, which keeps a partial clone
    (`git fetch --filter=blob:none`) to a few blob fetches.
    """
    log = _git("log", "--reverse", "--format=%H%x09%cI%x09%s", ref)
    found: list[tuple[str, str, str, str, str]] = []
    for line in log.splitlines():
        pages, time, message = line.split("\t", 2)
        sha = source_sha(message)
        if sha is None:
            continue
        listed = {}
        for entry in _git("ls-tree", pages, "figures/").splitlines():
            meta, path = entry.split("\t", 1)
            listed[path] = meta.split()[2]
        seen = set()
        for path, target in FILES.items():
            if path in listed and target not in seen:
                seen.add(target)
                found.append((sha, time, pages, target, listed[path]))
    blobs = {}
    for oid in dict.fromkeys(f[4] for f in found):
        try:
            blobs[oid] = json.loads(_git("cat-file", "blob", oid))
        except json.JSONDecodeError:
            continue
    return [(*f[:4], blobs[f[4]]) for f in found if f[4] in blobs]


def current_deploys(figures: Path, sha: str, time: str) -> Iterator[Deploy]:
    """Yield this build's summaries from its figures, as a deploy."""
    for target in ("latest", "complete"):
        path = figures / f"scoresummary-backtest-{target}.json"
        if path.exists():
            yield sha, time, "", target, json.loads(path.read_text())


CHART_METRICS: tuple[tuple[str, str, float | None, bool], ...] = (
    ("rel_wis", "Relative WIS", 1.0, True),
    ("wis", "WIS", None, False),
    ("coverage_50", "50% coverage", 0.5, False),
    ("coverage_90", "90% coverage", 0.9, False),
)
"""Chart metrics: column, label, reference line and log axis."""

POOLS = ("national", "provinces", "zones")

DEFAULT_BASELINE = "randomwalk"
"""The baseline the chart opens with and the table and summary show."""


BASELINES = ("randomwalk", "randomwalknowcast")
"""Baselines the page offers. Each one adds a set per pool, split and
forecast set to the chart data."""


def _label(time: str, sha: str, pages: str) -> str:
    when = f"{time[5:10]} {time[11:16]}"
    return f"{when} {sha[:7]}" if pages else f"{when} {sha[:7]} (this build)"


def chart_data(
    hist: pd.DataFrame, colours: Mapping[str, tuple[str, str, str]]
) -> dict[str, dict[str, object]]:
    """Return each target's `checkschart.js` data, by pool, over deploys.

    One set per baseline and pool, keyed `<baseline>|<pool>|deploy`, for
    the chart's baseline picker. Deploys are the x axis, named by time and
    source commit. A deploy where any model's scored forecasts against
    that baseline changed (`break`) is starred and listed in `marks`,
    which the chart draws as a vertical line.
    """
    hist = _with_metrics(hist)
    models = list(dict.fromkeys(hist["model"]))
    default = DEFAULT_BASELINE if DEFAULT_BASELINE in models else models[0]
    grey = ("#888888", "#888888", "")
    out: dict[str, dict[str, object]] = {}
    for target, cell in hist.groupby("target", sort=False):
        sets = {}
        pools = [p for p in POOLS if p in set(cell["pool"])]
        splits, first = _splits(cell)
        series = ["baseline", "pool", "split", "common"]
        for (base, pool, split, common), g in cell.groupby(series, sort=False):
            key = (
                g["deploy_time"] + "|" + g["source_sha"] + "|" + g["pages_sha"]
            )
            brk = g.groupby(key, sort=False)["break"].any()
            x: list[str] = []
            marks: list[int] = []
            for i, (k, b) in enumerate(brk.items()):
                time, sha, pages = str(k).split("|")
                x.append(_label(time, sha, pages) + (" *" if b else ""))
                if b:
                    marks.append(i)
            index = {k: i for i, k in enumerate(brk.index)}
            rows = [
                [
                    models.index(r["model"]),
                    index[k],
                    int(r["n"]),
                    *(_round(r[c]) for c, *_ in CHART_METRICS),
                ]
                for k, r in zip(key, g.to_dict("records"), strict=True)
            ]
            name = f"{base}|{pool}|{split}" + ("|common" if common else "")
            sets[name] = {
                "x": x,
                "rows": rows,
                "marks": marks,
            }
        out[str(target)] = {
            "kind": "line",
            "x_label": "Deploy",
            "baseline": models.index(default),
            "baselines": list(dict.fromkeys(hist["baseline"])),
            "common": bool(cell["common"].any()),
            "models": [
                {
                    "name": m,
                    "light": colours.get(m, grey)[0],
                    "dark": colours.get(m, grey)[1],
                    "dash": colours.get(m, grey)[2],
                    "on": True,
                }
                for m in models
            ],
            "metrics": [
                {"key": c, "label": label, "ref": ref, "log": log}
                for c, label, ref, log in CHART_METRICS
            ],
            "scales": pools,
            "splits": splits,
            "split": first,
            "sets": sets,
        }
    return out


def _splits(cell: pd.DataFrame) -> tuple[list[str], str]:
    """Order the splits and pick the one the chart opens with.

    Target weeks come first, newest first, then all forecasts. The chart
    opens on the newest week scored at the most deploys.
    """
    names = set(cell["split"])
    weeks = sorted((n for n in names if n.startswith("week ")), reverse=True)
    full = cell.loc[~cell["common"]]
    deploys = full.groupby("split")["deploy_time"].nunique()
    most = max((deploys[w] for w in weeks), default=0)
    first = next((w for w in weeks if deploys[w] == most), "all")
    return [*weeks, "all"], first


def _default(hist: pd.DataFrame) -> pd.DataFrame:
    """Keep the rows relative to the default baseline, over all forecasts.

    The table shows these only. The chart has the rest.
    """
    keep = (
        (hist["baseline"] == DEFAULT_BASELINE)
        & (hist["split"] == "all")
        & ~hist["common"]
    )
    return hist.loc[keep]


def _with_metrics(hist: pd.DataFrame) -> pd.DataFrame:
    """Add any chart metric a summary lacked, as missing."""
    lacking = {c: float("nan") for c, *_ in CHART_METRICS if c not in hist}
    return hist.assign(**lacking)


def _round(v: Any) -> float | None:
    return None if pd.isna(v) else float(f"{float(v):.4g}")


def _table(hist: pd.DataFrame, repo: str) -> str:
    head = (
        "| Deployed | Commit | Target | Pool | Model | n | WIS "
        f"| Relative WIS ({DEFAULT_BASELINE}) "
        "| 50% coverage | 90% coverage | Break |\n"
        "|---|---|---|---|---|---:|---:|---:|---:|---:|---|\n"
    )

    def num(v: Any) -> str:
        return "" if pd.isna(v) else f"{v:.3g}"

    lines = [
        f"| {str(r['deploy_time'])[:16].replace('T', ' ')} "
        f"| [{str(r['source_sha'])[:7]}](https://github.com/{repo}/commit/"
        f"{r['source_sha']}) | {r['target']} | {r['pool']} | {r['model']} "
        f"| {r['n']} | {num(r['wis'])} | {num(r['rel_wis'])} "
        f"| {num(r['coverage_50'])} | {num(r['coverage_90'])} "
        f"| {'yes' if r['break'] else ''} |"
        for r in _with_metrics(hist).iloc[::-1].to_dict("records")
    ]
    return head + "\n".join(lines)


def page_markdown(hist: pd.DataFrame | None, repo: str) -> str:
    """Return the Score history page's chart and table, or a notice."""
    if hist is None or hist.empty:
        return (
            '!!! warning "Score history unavailable"\n\n'
            "    The deploy history (the gh-pages branch) could not be "
            "read for this build.\n"
        )
    srcs = {
        t: f"../../figures/scorehistory-{t}.json"
        for t in dict.fromkeys(hist["target"])
    }
    table = "\n".join(
        f"    {line}" if line else ""
        for line in (
            '<div class="tablefilter" data-filters="Target,Pool,Model" '
            'markdown="1">\n\n' + _table(_default(hist), repo) + "\n\n</div>"
        ).splitlines()
    )
    return (
        f"<div class=\"checkschart\" data-targets='{json.dumps(srcs)}' "
        'role="group" aria-label="Backtest scores by deploy">\n'
        "<noscript>The score history chart needs JavaScript.</noscript>\n"
        "</div>\n\n"
        '??? note "Table"\n\n'
        f"{table}\n"
    )


def _overview(hist: pd.DataFrame, targets: Sequence[str]) -> str:
    """Relative WIS by deploy and model, for each target and pool."""
    parts = []
    for (target, pool), cell in hist.groupby(["target", "pool"]):
        if target not in targets or pool == "zones":
            continue
        cell = cell.assign(
            deploy=[
                _label(t, s, p)
                for t, s, p in zip(
                    cell["deploy_time"],
                    cell["source_sha"],
                    cell["pages_sha"],
                    strict=True,
                )
            ]
            + cell.groupby("pages_sha")["break"]
            .transform("any")
            .map({True: " *", False: ""})
        )
        wide = cell.pivot_table(
            index="deploy", columns="model", values="rel_wis", sort=False
        )
        parts.append(f"Relative WIS, {target}, {pool}\n{wide.round(3)}")
    return "\n\n".join(parts)


def main(argv: Sequence[str] | None = None) -> None:
    """Write the score history and, if asked, the Score history page."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--ref", default="origin/gh-pages")
    parser.add_argument("--out", default="results/score-history.csv")
    parser.add_argument("--docs", help="Write score-history.md here")
    parser.add_argument("--figures", help="Write the chart JSON here")
    parser.add_argument("--current", help="This build's figures directory")
    parser.add_argument("--current-sha", default="HEAD")
    parser.add_argument("--repo", default="seabbs/BDBVHubRenewal")
    args = parser.parse_args(argv)
    try:
        deploys = walk(args.ref)
    except subprocess.CalledProcessError as err:
        print(f"Deploy history unavailable: {err.stderr}", file=sys.stderr)
        deploys = []
    if args.current:
        now = pd.Timestamp.now(tz="UTC").isoformat(timespec="seconds")
        sha = args.current_sha
        if sha == "HEAD":
            with contextlib.suppress(subprocess.CalledProcessError):
                sha = _git("rev-parse", "HEAD").strip()
        deploys += current_deploys(Path(args.current), sha, now)
    hist = history(deploys, BASELINES) if deploys else None
    if args.docs:
        docs = Path(args.docs)
        docs.mkdir(parents=True, exist_ok=True)
        page = page_markdown(hist, args.repo)
        (docs / "score-history.md").write_text(page)
    if hist is None:
        return
    if args.figures:
        colours = {
            m["name"]: (m["light"], m["dark"], m.get("dash", ""))
            for d in deploys
            for m in d[4]["models"]
            if "light" in m
        }
        figures = Path(args.figures)
        figures.mkdir(parents=True, exist_ok=True)
        for target, data in chart_data(hist, colours).items():
            path = figures / f"scorehistory-{target}.json"
            path.write_text(json.dumps(data, separators=(",", ":")))
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        hist.to_csv(out, index=False, float_format="%.6g")
        print(f"Wrote {len(hist)} rows to {out}", file=sys.stderr)
    if not args.docs:
        print(_overview(_default(hist), ("latest", "complete")))
        print("\n* what a model is scored on changed at this deploy")


if __name__ == "__main__":
    main()
