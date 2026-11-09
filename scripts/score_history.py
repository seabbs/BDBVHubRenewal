# Copyright (c) 2026 Sam Abbott. MIT licence, see LICENSE.
"""Backtest scores over time, one point per deploy of the site from main.

The gh-pages branch keeps every deploy. A deploy from main is the commit
`Deploying to gh-pages from @ <repo>@<sha>`, so each one maps to the
main commit it was built from. Pull request previews are left out. Each
deploy's `figures/scoresummary-backtest-<target>.json` holds the
per-forecast scores the Backtest page summarises. This script summarises
them as the page's overall view does (`relative_skill` over the forecasts
the baseline also made), by pool and model, and writes one row per
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
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from collections.abc import Iterable, Iterator, Mapping, Sequence
from pathlib import Path
from typing import Any

import pandas as pd

from bdbvhubrenewal.evaluation.scoring import relative_skill

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
    out = relative_skill(
        rows,
        baseline=models[summary["baseline"]],
        by=("pool",),
        metrics=relative,
        match_on=MATCH,
        means=keys,
    )
    out = out.rename(columns={f"relative_{k}": f"rel_{k}" for k in relative})
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


def walk(ref: str) -> Iterator[Deploy]:
    """Yield the score summaries of each deploy from main, oldest first."""
    log = _git("log", "--reverse", "--format=%H%x09%cI%x09%s", ref)
    for line in log.splitlines():
        pages, time, message = line.split("\t", 2)
        sha = source_sha(message)
        if sha is None:
            continue
        listed = set(_git("ls-tree", "--name-only", pages, "figures/").split())
        seen = set()
        for path, target in FILES.items():
            if path not in listed or target in seen:
                continue
            try:
                summary = json.loads(_git("show", f"{pages}:{path}"))
            except json.JSONDecodeError:
                continue
            seen.add(target)
            yield sha, time, pages, target, summary


def _overview(hist: pd.DataFrame, targets: Sequence[str]) -> str:
    """Relative WIS by deploy and model, for each target and pool."""
    parts = []
    for (target, pool), cell in hist.groupby(["target", "pool"]):
        if target not in targets or pool == "zones":
            continue
        cell = cell.assign(
            deploy=cell["deploy_time"].str[:16]
            + " "
            + cell["source_sha"].str[:8]
            + cell.groupby("pages_sha")["break"]
            .transform("any")
            .map({True: " *", False: ""})
        )
        wide = cell.pivot_table(
            index="deploy", columns="model", values="rel_wis", sort=False
        )
        parts.append(
            f"Relative WIS, {target}, {pool}\n{wide.round(3).to_string()}"
        )
    return "\n\n".join(parts)


def main(argv: Sequence[str] | None = None) -> None:
    """Write the score history and print relative WIS by deploy."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--ref", default="origin/gh-pages")
    parser.add_argument("--out", default="results/score-history.csv")
    args = parser.parse_args(argv)
    hist = history(walk(args.ref))
    hist.to_csv(Path(args.out), index=False, float_format="%.6g")
    print(_overview(hist, ("latest", "complete")))
    print("\n* what a model is scored on changed at this deploy")
    print(f"Wrote {len(hist)} rows to {args.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
