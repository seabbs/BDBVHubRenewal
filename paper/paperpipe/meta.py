"""Each study's run notes: where its data came from.

The `data` step writes `<study>-meta.json` beside the study's data, with
`source` either `real` or `dummy` (`paperpipe.studies.missing`), the
baseline its scores are relative to where that is not the usual one,
and a note on what is provisional. Later steps read it, so figures and
the paper mark dummy and provisional results as such.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Final

__all__ = ["DUMMY_LABEL", "baseline_of", "read_meta", "write_meta"]

DUMMY_LABEL: Final = "Illustrative, dummy data"
"""Title of every figure, and part of every caption, from dummy data."""


def write_meta(
    path: Path,
    study: str,
    source: str,
    baseline: str | None = None,
    note: str = "",
) -> None:
    """Write a study's run notes.

    `baseline` is the model its scores are relative to, where it is not
    the study's usual one, and `note` says what is provisional.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    meta = {"study": study, "source": source, "note": note}
    if baseline:
        meta["baseline"] = baseline
    path.write_text(json.dumps(meta, indent=2) + "\n")


def read_meta(path: Path) -> dict[str, Any]:
    """Read a study's run notes."""
    return json.loads(Path(path).read_text())


def baseline_of(meta: dict[str, Any]) -> str:
    """Return the model a study's scores are relative to."""
    from paperfigs.scores import STUDY_BASELINES

    return meta.get("baseline") or STUDY_BASELINES[meta["study"]]
