# Copyright (c) 2026 Sam Abbott. MIT licence, see LICENSE.
"""Say where the fitted models' results on a page come from.

Every page with fitted-model results includes `fit-store.md`, written
from the record `bdbvhubrenewal fit-store load` leaves in the results
(`bdbvhubrenewal.evaluation.store.STATUS_FILE`): the commits the fits
were made at, and the fits for this build's code not stored yet.
"""

from __future__ import annotations

import json
import os
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Final

from bdbvhubrenewal.evaluation.store import STATUS_FILE

__all__ = ["store_note", "write_store_note"]

REPO: Final = "https://github.com/seabbs/BDBVHubRenewal"
"""Where commits and workflow runs link to."""


def _commit(sha: str) -> str:
    return f"[`{sha[:7]}`]({REPO}/commit/{sha})"


def _pairs(rows: Sequence[Mapping[str, Any]]) -> str:
    """List fits by model: `renewal` at 2026-10-24 and 2026-11-01."""
    origins: dict[str, list[str]] = {}
    for r in rows:
        origins.setdefault(str(r["model"]), []).append(str(r["origin"]))
    return "; ".join(
        f"`{m}` at {' and '.join(sorted(o))}" for m, o in origins.items()
    )


def store_note(
    status: Sequence[Mapping[str, Any]],
    build: str | None = None,
    running: str | None = None,
) -> str:
    """Write the note for a page.

    Parameters
    ----------
    status : sequence of mapping
        The `fit-store load` record, one entry per (model, origin).
    build : str, optional
        The commit this site is built from.
    running : str, optional
        The URL of a Fit store run in progress, if there is one.

    Returns
    -------
    str
        Markdown: the fit commits, then an admonition listing the fits
        not made for this build's code, if there are any.
    """
    used = sorted({str(s["commit"]) for s in status if s["commit"]})
    if not used:
        lines = ["No fitted model has a stored fit yet."]
    else:
        lines = [
            "Fitted models' results come from the "
            f"[fit store]({REPO}/tree/fit-store), fitted at "
            + ", ".join(_commit(c) for c in used)
            + "."
        ]
    stale = [s for s in status if s["used"] and not s["current"]]
    absent = [s for s in status if not s["used"]]
    failed = [s for s in status if s["failed"]]
    if not (stale or absent):
        return "\n".join(lines) + "\n"
    code = f"this build's code ({_commit(build)})" if build else "this code"
    body = [f"Not yet fitted for {code}: {_pairs(stale + absent)}."]
    if failed:
        body.append(f"The fit failed for {_pairs(failed)}.")
    if running:
        body.append(f"A [Fit store run]({running}) is fitting them.")
    if stale:
        body.append(f"Fits of older code are shown for {_pairs(stale)}.")
    if absent:
        body.append(f"Not shown: {_pairs(absent)}.")
    lines += ["", '!!! note "Fits not yet made for this code"', ""]
    lines += [f"    {b}" for b in body]
    return "\n".join(lines) + "\n"


def write_store_note(results: Path | str, generated: Path | str) -> None:
    """Write `fit-store.md` from the results' store record.

    The build commit is `$GITHUB_SHA` and a Fit store run in progress is
    `$FIT_STORE_RUN`, which the Pages workflow sets.

    Parameters
    ----------
    results : pathlib.Path or str
        The results directory.
    generated : pathlib.Path or str
        Where the docs snippets go.
    """
    path = Path(results) / STATUS_FILE
    status = json.loads(path.read_text()) if path.exists() else []
    out = Path(generated) / "fit-store.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        store_note(
            status,
            os.environ.get("GITHUB_SHA"),
            os.environ.get("FIT_STORE_RUN") or None,
        )
    )
