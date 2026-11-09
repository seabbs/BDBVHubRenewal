# Copyright (c) 2026 Sam Abbott. MIT licence, see LICENSE.
"""Hash what a backtest run reads, to key its cached outputs.

The Pages workflow runs the backtests that fit no model as separate
jobs and caches each one's outputs under `backtest_digest` of what it
reads. A change to report-only code or to docstrings and comments then
reruns none of them.

This lives apart from `evaluation.store`, whose own code is part of
every stored fit's key, so a change here refits nothing.
"""

from __future__ import annotations

import hashlib
import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Final

from bdbvhubrenewal.evaluation.store import _strip

__all__ = ["BACKTEST_INPUTS", "REPORT_ONLY", "backtest_digest"]

REPORT_ONLY: Final = tuple(
    f"src/bdbvhubrenewal/reporting/{name}.py"
    for name in (
        "hubtables",
        "nowcast_plots",
        "report",
        "scorechart",
        "store_note",
        "tables",
        "zone_nowcast",
    )
)
"""Modules only the `report` command runs, so no backtest reads them."""

BACKTEST_INPUTS: Final = ("src/bdbvhubrenewal", "config", "data")
"""What every backtest run reads, relative to the repository root."""

_SKIP: Final = ("data/raw", "data/sources.json", *REPORT_ONLY)
"""Not read by a backtest. `data/sources.json` records when the raw
downloads were made."""


def backtest_digest(
    root: Path | str = ".", extra: Sequence[str] = (), text: str = ""
) -> str:
    """Hash what a backtest run reads.

    The files in `BACKTEST_INPUTS` and `extra`, less `REPORT_ONLY`,
    `data/raw` and `data/sources.json`, are hashed by path and content,
    Python files as syntax trees without docstrings (as
    `evaluation.store.code_digest`). Then `text` (such as the commands
    run), the Python minor version and the locked runtime dependencies.

    Parameters
    ----------
    root : pathlib.Path or str
        The repository root.
    extra : sequence of str
        Further files or directories the run reads, relative to `root`.
    text : str
        Added to the hash as it is.

    Returns
    -------
    str
        SHA-256 hex digest.
    """
    base = Path(root)
    skipped = [base / s for s in _SKIP]
    files: list[Path] = []
    for name in (*BACKTEST_INPUTS, *extra):
        path = base / name
        files += sorted(path.rglob("*")) if path.is_dir() else [path]
    digest = hashlib.sha256()
    for path in files:
        if not path.is_file() or "__pycache__" in path.parts:
            continue
        if any(path == s or s in path.parents for s in skipped):
            continue
        digest.update(path.relative_to(base).as_posix().encode())
        if path.suffix == ".py":
            digest.update(_strip(path.read_text()).encode())
        else:
            digest.update(path.read_bytes())
    digest.update(text.encode())
    digest.update(repr(sys.version_info[:2]).encode())
    deps = subprocess.run(
        [
            "uv",
            "export",
            "--frozen",
            "--no-dev",
            "--all-extras",
            "--no-hashes",
            "--no-header",
            "--no-emit-project",
        ],
        cwd=base,
        capture_output=True,
        text=True,
        check=True,
    )
    digest.update(deps.stdout.encode())
    return digest.hexdigest()
