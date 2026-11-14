"""Each study's run notes: where its data came from.

The `data` step writes `<study>-meta.json` beside the study's data, with
`source` either `real` or `dummy` (`paperpipe.studies.source_of`). Later
steps read it, so figures and the paper mark dummy results as such.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Final

__all__ = ["DUMMY_LABEL", "read_meta", "write_meta"]

DUMMY_LABEL: Final = "Illustrative, dummy data"
"""Label on every figure and table built from dummy data."""


def write_meta(path: Path, study: str, source: str) -> None:
    """Write a study's run notes."""
    path.parent.mkdir(parents=True, exist_ok=True)
    meta = {"study": study, "source": source}
    path.write_text(json.dumps(meta, indent=2) + "\n")


def read_meta(path: Path) -> dict[str, Any]:
    """Read a study's run notes."""
    return json.loads(Path(path).read_text())
