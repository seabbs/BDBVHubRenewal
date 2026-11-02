# Copyright (c) 2026 Sam Abbott. MIT licence, see LICENSE.
"""Write the weekly cross-check points and figure.

Usage::

    uv run python scripts/weekly_crosscheck.py

See `bdbvhubrenewal.evaluation.crosscheck`.
"""

from bdbvhubrenewal.evaluation.crosscheck import make_crosscheck

if __name__ == "__main__":
    make_crosscheck()
