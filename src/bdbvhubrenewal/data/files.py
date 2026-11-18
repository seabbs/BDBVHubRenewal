# Copyright (c) 2026 Sam Abbott. MIT licence, see LICENSE.
"""Tables written so the same table always gives the same bytes."""

from __future__ import annotations

from pathlib import Path
from typing import Final

import pandas as pd

__all__ = ["write_csv"]

_GZIP: Final = {"method": "gzip", "mtime": 0}
"""Gzip settings with a fixed time, so a table always writes the same bytes."""


def write_csv(frame: pd.DataFrame, path: Path | str) -> None:
    """Write a table as CSV, gzipped when `path` ends in `.gz`.

    The bytes depend on the table only, so an unchanged table leaves its
    file, its git diff and any hash of it unchanged.

    Parameters
    ----------
    frame : pandas.DataFrame
        The table, written without its index.
    path : pathlib.Path or str
        Where to write it.
    """
    gz = Path(path).suffix == ".gz"
    frame.to_csv(path, index=False, compression=_GZIP if gz else None)
