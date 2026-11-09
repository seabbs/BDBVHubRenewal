# Copyright (c) 2026 Sam Abbott. MIT licence, see LICENSE.
"""Backtest scores over time, one point per deploy of the site from main.

The gh-pages branch keeps every deploy. A deploy from main is the commit
`Deploying to gh-pages from @ <repo>@<sha>`, so each one maps to the
main commit it was built from. Pull request previews are left out. Each
deploy's `figures/scoresummary-backtest-<target>.json` holds the
per-forecast scores the Backtest page summarises. This script summarises
them as the page's overall view does (as `relative_skill`, over the
forecasts the baseline also made), by pool and model, and writes one row per
deploy, target, pool and model.

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


def summary_scores(summary: Mapping[str, Any]) -> pd.DataFrame:
    """Summarise one score summary by pool and model.

    Parameters
    ----------
    summary : mapping
        A `summary_rows` JSON: `models`, `baseline`, `metrics`,
        `origins`, `releases`, `places` (with `pool`), `weeks` and `rows`.

    Returns
    -------
    pandas.DataFrame
        `pool`, `model`, `n` (forecasts the baseline also made), the mean
        of each metric and, for relative metrics, the baseline's mean on
        the same forecasts `base_<metric>` and `rel_<metric>`.
    """
    models = [m["name"] for m in summary["models"]]
    keys = [m["key"] for m in summary["metrics"]]
    relative = [m["key"] for m in summary["metrics"] if m["relative"]]
    pools = [p["pool"] for p in summary["places"]]
    width = len(MATCH) + 1 + len(keys)
    rows = pd.DataFrame(
        [r[:width] for r in summary["rows"]],
        columns=["model", *MATCH, *keys],
    )
    rows["pool"] = [pools[i] for i in rows["place"]]
    rows["model"] = [models[i] for i in rows["model"]]
    rows = rows.loc[~rows["model"].isin(SKIP_MODELS)]
    # As `relative_skill`, without importing the package (and JAX): each
    # model over the forecasts the baseline also made.
    base_rows = rows.loc[rows["model"] == models[summary["baseline"]]]
    merged = rows.merge(
        base_rows[["pool", *MATCH, *relative]],
        on=["pool", *MATCH],
        suffixes=("", "_b"),
    )
    grp = merged.groupby(["pool", "model"], sort=False)
    out = grp[keys].mean()
    out.insert(0, "n", grp.size())
    for k in relative:
        out[f"base_{k}"] = grp[f"{k}_b"].mean()
        out[f"rel_{k}"] = out[k] / out[f"base_{k}"].where(out[f"base_{k}"] > 0)
    out = out.reset_index()
    rel = [f"rel_{k}" for k in relative]
    base = [f"base_{k}" for k in relative]
    return out[["pool", "model", "n", *keys, *base, *rel]]


def history(deploys: Iterable[Deploy]) -> pd.DataFrame:
    """Summarise each deploy's scores, oldest first, and flag breaks.

    Parameters
    ----------
    deploys : iterable of (str, str, str, str, mapping)
        Source commit, deploy time, gh-pages commit, target and summary,
        oldest first.

    Returns
    -------
    pandas.DataFrame
        One row per deploy, target, pool and model: `deploy_time`,
        `source_sha`, `pages_sha`, `target`, `pool`, `model`, the
        `summary_scores` columns, `origins` and `releases` (`;`-joined)
        and `break`, true where the origins or releases, or the number of
        forecasts scored, differ from the model's previous deploy of that
        target and pool, or where the model first appears.
    """
    frames = []
    for sha, time, pages, target, summary in deploys:
        frame = summary_scores(summary)
        frame.insert(0, "target", target)
        frame.insert(0, "pages_sha", pages)
        frame.insert(0, "source_sha", sha)
        frame.insert(0, "deploy_time", time)
        frame["origins"] = ";".join(summary["origins"])
        frame["releases"] = ";".join(summary["releases"])
        frames.append(frame)
    out = pd.concat(frames, ignore_index=True)
    scored = ["origins", "releases", "n"]
    prev = out.groupby(["target", "pool", "model"])[scored].shift()
    out["break"] = (prev != out[scored]).any(axis=1)
    return out


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


def _label(time: str, sha: str, pages: str) -> str:
    when = f"{time[5:10]} {time[11:16]}"
    return f"{when} {sha[:7]}" if pages else f"{when} {sha[:7]} (this build)"


def chart_data(
    hist: pd.DataFrame, colours: Mapping[str, tuple[str, str, str]]
) -> dict[str, dict[str, object]]:
    """Return each target's `checkschart.js` data, by pool, over deploys.

    Deploys are the x axis, named by time and source commit. A deploy
    where any model's scored forecasts changed (`break`) is starred and
    listed in `marks`, which the chart draws as a vertical line.
    """
    hist = _with_metrics(hist)
    models = list(dict.fromkeys(hist["model"]))
    grey = ("#888888", "#888888", "")
    out: dict[str, dict[str, object]] = {}
    for target, cell in hist.groupby("target", sort=False):
        sets = {}
        pools = [p for p in POOLS if p in set(cell["pool"])]
        for pool in pools:
            g = cell.loc[cell["pool"] == pool]
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
            sets[f"{pool}|deploy"] = {"x": x, "rows": rows, "marks": marks}
        out[str(target)] = {
            "kind": "line",
            "x_label": "Deploy",
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
            "splits": ["deploy"],
            "sets": sets,
        }
    return out


def _with_metrics(hist: pd.DataFrame) -> pd.DataFrame:
    """Add any chart metric a summary lacked, as missing."""
    lacking = {c: float("nan") for c, *_ in CHART_METRICS if c not in hist}
    return hist.assign(**lacking)


def _round(v: Any) -> float | None:
    return None if pd.isna(v) else round(float(v), 6)


def _table(hist: pd.DataFrame, repo: str) -> str:
    head = (
        "| Deployed | Commit | Target | Pool | Model | n | WIS "
        "| Relative WIS | 50% coverage | 90% coverage | Break |\n"
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
            'markdown="1">\n\n' + _table(hist, repo) + "\n\n</div>"
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
    hist = history(deploys) if deploys else None
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
        print(_overview(hist, ("latest", "complete")))
        print("\n* what a model is scored on changed at this deploy")


if __name__ == "__main__":
    main()
