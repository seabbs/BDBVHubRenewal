# Copyright (c) 2026 Sam Abbott. MIT licence, see LICENSE.
"""Check regenerated fixtures against the committed ones.

R's numbers move between R builds and platforms: the last floating point
digit everywhere, and the negative binomial sizes from `optimize` (whose
default tolerance is about 1e-4) further. Numbers therefore match to a
relative 1e-3, the tolerance the tests compare those sizes to, and
everything else exactly. A file added or removed by the regeneration
fails too.

Usage::

    python3 scripts/check_fixtures.py tests/baselinenowcast/fixtures
"""

from __future__ import annotations

import math
import re
import subprocess
import sys
from pathlib import Path

TOKEN = re.compile(r"[,\s]+")


def git(*args: str) -> str:
    """Run git and return its output."""
    return subprocess.run(
        ["git", *args], check=True, capture_output=True, text=True
    ).stdout


def same(a: str, b: str) -> bool:
    """Whether two tokens match: numbers to 1e-3, others exactly."""
    try:
        return math.isclose(float(a), float(b), rel_tol=1e-3, abs_tol=1e-9)
    except ValueError:
        return a == b


def differs(committed: str, current: str) -> bool:
    """Whether two file contents differ beyond the numeric tolerance."""
    old, new = committed.splitlines(), current.splitlines()
    if len(old) != len(new):
        return True
    for a, b in zip(old, new, strict=True):
        ta, tb = TOKEN.split(a.strip()), TOKEN.split(b.strip())
        if len(ta) != len(tb) or not all(map(same, ta, tb)):
            return True
    return False


def main(root: str) -> int:
    """Print each fixture that differs; return 1 if any does."""
    changed = git("diff", "--name-only", "--", root).split()
    added = git("ls-files", "--others", "--exclude-standard", root).split()
    removed = git("ls-files", "--deleted", "--", root).split()
    bad = added + removed
    for path in sorted(set(changed) - set(removed)):
        current = Path(path).read_text(encoding="utf-8")
        if differs(git("show", f"HEAD:{path}"), current):
            bad.append(path)
    for path in bad:
        print(f"{path} differs from the committed fixture")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1]))
