"""Store a checkpoint's German forecasts for the site's Germany page.

    uv run python scripts/germany_ft_store.py CACHE LABEL

CACHE is the checkpoint scorer's cache (`scripts/checkpoint_scores.py
germany CACHE LABEL` on the checkpoint-scoring branch), with one file
per development date in CACHE/germany/LABEL/. LABEL is a settings hash
(release `chronos-ft-<hash>`) or `slots-<hash>` (`chronos-slots-<hash>`).

The script keeps `fm_chronos_daily_ragged_ft` on the development dates
(refusing any date on or after the split date in config/germany.toml),
writes them to `<release tag>/forecasts.csv.gz` with a `manifest.json`,
commits that to the `germany-ft-forecasts` branch of `origin` and pushes
it (no force). An unchanged checkpoint commits nothing. The Pages build
reads the branch (`bdbvhubrenewal germany-ft-load`).

Options:
  --repo DIR     the git checkout to commit from (default: this one).
  --no-push      write the files only, to --out, and commit nothing.
  --out DIR      where --no-push writes (default: a temporary directory).
  --partial      store a label that lacks some development dates.
  --dispatch     after pushing, run the Pages workflow on main, so the
                 site shows the new checkpoint.
  --config FILE  config/germany.toml.

Exits non-zero, storing nothing, if the label lacks development dates
(without --partial) or holds a held-out date.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import pandas as pd

from bdbvhubrenewal.evaluation.germany import (
    load_germany_config,
    nowcast_dates,
)
from bdbvhubrenewal.evaluation.germany_store import (
    STORE_BRANCH,
    release_tag,
    write_stored,
)

ROOT = Path(__file__).resolve().parents[1]
README = """# Stored German forecasts of the fine-tuned Chronos-2 model

Written by `scripts/germany_ft_store.py` on main from the checkpoint
scorer's cache, and read by the Pages build (`bdbvhubrenewal
germany-ft-load`). One directory per checkpoint release tag, with
`forecasts.csv.gz` (`germany-backtest`'s forecast columns,
`fm_chronos_daily_ragged_ft` only) and `manifest.json`. Development
nowcast dates only: nothing on or after the split date in
config/germany.toml.
"""


def read_cache(cache: Path, label: str, split_date: str) -> pd.DataFrame:
    """Stack a label's per-date files, development dates only."""
    files = sorted((cache / "germany" / label).glob("*.csv.gz"))
    late = [f.name for f in files if f.name[:10] >= split_date]
    if late:
        raise SystemExit(f"Held-out files in the cache, refusing: {late[:3]}")
    if not files:
        raise SystemExit(f"No cached forecasts in {cache / 'germany' / label}.")
    dtype = {"origin": str, "truth_vintage": str, "target_end_date": str}
    return pd.concat(
        [pd.read_csv(f, dtype=dtype) for f in files], ignore_index=True
    )


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
            [
                "git",
                "-C",
                str(repo),
                "fetch",
                "-q",
                "origin",
                f"+refs/heads/{STORE_BRANCH}:{ref}",
            ],
            capture_output=True,
        )
        parent = ""
        if fetched.returncode == 0:
            parent = _git(repo, "rev-parse", "--verify", "-q", ref)
        with tempfile.TemporaryDirectory() as tmp:
            env = {**os.environ, "GIT_INDEX_FILE": str(Path(tmp) / "index")}
            if parent:
                _git(repo, "read-tree", parent, env=env)
            else:
                _git(repo, "read-tree", "--empty", env=env)
            files = sorted(p for p in tree.rglob("*") if p.is_file())
            for path in files:
                name = path.relative_to(tree).as_posix()
                if name != "README.md" and not name.startswith(f"{tag}/"):
                    continue
                blob = _git(repo, "hash-object", "-w", str(path))
                _git(
                    repo,
                    "update-index",
                    "--add",
                    "--cacheinfo",
                    f"100644,{blob},{name}",
                    env=env,
                )
            new_tree = _git(repo, "write-tree", env=env)
        if parent and new_tree == _git(repo, "rev-parse", f"{parent}^{{tree}}"):
            print(f"{tag}: unchanged on {STORE_BRANCH}.")
            return False
        commit = _git(
            repo,
            "commit-tree",
            new_tree,
            *(["-p", parent] if parent else []),
            "-m",
            f"Store German forecasts of {tag} (label {label})",
        )
        pushed = subprocess.run(
            [
                "git",
                "-C",
                str(repo),
                "push",
                "-q",
                "origin",
                f"{commit}:refs/heads/{STORE_BRANCH}",
            ],
            capture_output=True,
            text=True,
        )
        if pushed.returncode == 0:
            print(f"{tag}: pushed {commit[:12]} to {STORE_BRANCH}.")
            return True
        print(
            f"Push refused (try {attempt + 1}): {pushed.stderr.strip()}",
            file=sys.stderr,
        )
    raise SystemExit(f"Could not push to {STORE_BRANCH}.")


def main(argv: list[str] | None = None) -> int:
    """Store LABEL's forecasts from CACHE."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("cache", type=Path)
    parser.add_argument("label")
    parser.add_argument("--repo", type=Path, default=ROOT)
    parser.add_argument("--no-push", action="store_true")
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--partial", action="store_true")
    parser.add_argument("--dispatch", action="store_true")
    parser.add_argument(
        "--config", type=Path, default=ROOT / "config" / "germany.toml"
    )
    args = parser.parse_args(argv)
    config = load_germany_config(args.config)
    tag = release_tag(args.label)
    frame = read_cache(args.cache, args.label, config.split_date)
    want = set(nowcast_dates(config))
    missing = want - set(frame["origin"].astype(str))
    if missing and not args.partial:
        raise SystemExit(
            f"{args.label}: {len(missing)} development dates missing "
            f"(first {min(missing)}); not stored. --partial stores it anyway."
        )
    with tempfile.TemporaryDirectory() as tmp:
        tree = args.out or Path(tmp)
        source = str(args.cache / "germany" / args.label)
        out = write_stored(
            tree, tag, frame, config.split_date, args.label, source=source
        )
        (tree / "README.md").write_text(README)
        print(f"{tag}: wrote {out}.")
        if args.no_push:
            return 0
        pushed = commit_and_push(args.repo, tree, tag, args.label)
    if pushed and args.dispatch:
        subprocess.run(
            ["gh", "workflow", "run", "pages.yml", "--ref", "main"],
            check=True,
            cwd=args.repo,
        )
        print("Dispatched the Pages workflow on main.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
