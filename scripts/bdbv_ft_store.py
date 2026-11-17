"""Store a checkpoint's BDBV forecasts for CI on a branch of their own.

The checkpoint scorer's cache holds 1,000 draws per model, origin, place
and week (`checkpoint_scores.py bdbv`), which CI cannot reach. This
writes each model's quantiles at the hub's levels
(`reporting.hubtables.HUB_QUANTILE_LEVELS`), with the scored truth (the
first release D days after the week, `rolling_truth`) where it exists,
to `<release tag>/forecasts.csv.gz` with a `manifest.json` on branch
`STORE_BRANCH`, as `germany_ft_store.py` does for Germany. The reference
(zero-shot Chronos-2 and randomwalknowcast) is stored as `reference`.

Usage: python scripts/bdbv_ft_store.py CACHE LABEL [--repo DIR]

LABEL is `reference`, a hash (release `chronos-ft-<hash>`) or
`slots-<hash>` (release `chronos-slots-<hash>`). Rerunning an unchanged
label commits nothing.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from collections.abc import Sequence
from pathlib import Path

import pandas as pd

STORE_BRANCH = "bdbv-ft-forecasts"
KEYS = ["model", "origin", "location", "target_end_date"]


def bdbv_quantiles(
    draws: pd.DataFrame, truth: pd.DataFrame, levels: Sequence[float]
) -> pd.DataFrame:
    """Return each model, origin, place and week's quantiles with truth.

    Weeks with no truth yet keep an empty `truth`.
    """
    q = draws.groupby(KEYS)["value"].quantile(list(levels))
    out = q.rename_axis([*KEYS, "quantile"]).rename("value").reset_index()
    truth = truth.astype({"target_end_date": str})
    out = out.astype({"target_end_date": str}).merge(
        truth[["location", "target_end_date", "truth", "truth_vintage"]],
        on=["location", "target_end_date"],
        how="left",
    )
    return out.sort_values([*KEYS, "quantile"]).reset_index(drop=True)


def tag_of(label: str) -> str:
    """Return a label's folder on the store branch (its release tag)."""
    if label == "reference":
        return "reference"
    from bdbvhubrenewal.forecasters.foundation import release_tag

    return release_tag(label)


def build(cache: Path, label: str) -> pd.DataFrame:
    """Return the label's quantiles over every cached origin."""
    from bdbvhubrenewal.evaluation import backtest as bt
    from bdbvhubrenewal.evaluation.nowcast import (
        resolve_truth_delay,
        rolling_truth,
    )
    from bdbvhubrenewal.reporting.hubtables import HUB_QUANTILE_LEVELS

    config = bt.load_config()
    target = pd.read_csv("data/target.csv", dtype={"kind": str})
    real = target.loc[target["kind"] == "real"]
    truth = rolling_truth(real, resolve_truth_delay(config, target))
    read = {"dtype": {"origin": str, "target_end_date": str}}
    parts = []
    for f in sorted((cache / "bdbv" / label).glob("*.csv.gz")):
        draws = pd.read_csv(f, **read)
        if not draws.empty:
            parts.append(bdbv_quantiles(draws, truth, HUB_QUANTILE_LEVELS))
    if not parts:
        raise SystemExit(f"No cached BDBV draws for {label}.")
    return pd.concat(parts, ignore_index=True)


def _git(repo: Path, *args: str, env: dict[str, str] | None = None) -> str:
    run = subprocess.run(
        ["git", "-C", str(repo), *args],
        check=True,
        capture_output=True,
        text=True,
        env=env,
    )
    return run.stdout.strip()


def commit_and_push(repo: Path, tree: Path, tag: str, label: str) -> bool:
    """Commit TAG's files onto the store branch and push; False if same."""
    ref = f"refs/remotes/origin/{STORE_BRANCH}"
    for attempt in range(3):
        fetched = subprocess.run(
            ["git", "-C", str(repo), "fetch", "-q", "origin",
             f"+refs/heads/{STORE_BRANCH}:{ref}"],
            capture_output=True,
        )  # fmt: skip
        parent = ""
        if fetched.returncode == 0:
            parent = _git(repo, "rev-parse", "--verify", "-q", ref)
        with tempfile.TemporaryDirectory() as tmp:
            env = {**os.environ, "GIT_INDEX_FILE": str(Path(tmp) / "index")}
            if parent:
                _git(repo, "read-tree", parent, env=env)
            else:
                _git(repo, "read-tree", "--empty", env=env)
            for path in sorted(p for p in tree.rglob("*") if p.is_file()):
                name = path.relative_to(tree).as_posix()
                blob = _git(repo, "hash-object", "-w", str(path))
                _git(repo, "update-index", "--add", "--cacheinfo",
                     f"100644,{blob},{name}", env=env)  # fmt: skip
            new_tree = _git(repo, "write-tree", env=env)
        if parent and new_tree == _git(repo, "rev-parse", f"{parent}^{{tree}}"):
            print(f"{tag}: unchanged on {STORE_BRANCH}.")
            return False
        commit = _git(
            repo, "commit-tree", new_tree,
            *(["-p", parent] if parent else []),
            "-m", f"Store BDBV forecasts of {tag} (label {label})",
        )  # fmt: skip
        pushed = subprocess.run(
            ["git", "-C", str(repo), "push", "-q", "origin",
             f"{commit}:refs/heads/{STORE_BRANCH}"],
            capture_output=True,
            text=True,
        )  # fmt: skip
        if pushed.returncode == 0:
            print(f"{tag}: pushed {commit[:12]} to {STORE_BRANCH}.")
            return True
        print(f"Push refused (try {attempt + 1}): {pushed.stderr.strip()}",
              file=sys.stderr)  # fmt: skip
    raise SystemExit(f"Could not push to {STORE_BRANCH}.")


def main(argv: list[str] | None = None) -> int:
    """Store one label's BDBV quantiles."""
    parser = argparse.ArgumentParser()
    parser.add_argument("cache", type=Path)
    parser.add_argument("label")
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    args = parser.parse_args(argv)
    out = build(args.cache.expanduser(), args.label)
    tag = tag_of(args.label)
    with tempfile.TemporaryDirectory() as tmp:
        folder = Path(tmp) / tag
        folder.mkdir()
        out.to_csv(folder / "forecasts.csv.gz", index=False)
        manifest = {
            "label": args.label,
            "tag": tag,
            "models": sorted(out["model"].unique()),
            "origins": sorted(out["origin"].unique()),
            "rows": len(out),
            "levels": "reporting.hubtables.HUB_QUANTILE_LEVELS",
            "truth": "first release D days after each week (rolling_truth)",
        }
        (folder / "manifest.json").write_text(json.dumps(manifest, indent=1))
        commit_and_push(args.repo, Path(tmp), tag, args.label)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
