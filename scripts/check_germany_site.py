"""Check the built site's German page names no held-out nowcast date.

    uv run python scripts/check_germany_site.py SITE

SITE is the built site (`mkdocs build`, `_site`). Reads the German page
and its summary and explorer JSON (`reporting.germany.held_out_leaks`)
and exits non-zero, listing them, if any names a date on or after the
split date in config/germany.toml (or after it, in the page's prose).
"""

from __future__ import annotations

import sys
from pathlib import Path

from bdbvhubrenewal.evaluation.germany import load_germany_config
from bdbvhubrenewal.reporting.germany import held_out_leaks

ROOT = Path(__file__).resolve().parents[1]


def main(argv: list[str]) -> int:
    """Check SITE's German page."""
    site = Path(argv[0]) if argv else Path("_site")
    split = load_germany_config(ROOT / "config" / "germany.toml").split_date
    files = [
        site / "germany" / "index.html",
        site / "figures" / "scoresummary-germany-latest.json",
        site / "figures" / "explorer-germany.json",
    ]
    missing = [str(f) for f in files if not f.exists()]
    if missing:
        print(f"Not built: {', '.join(missing)}", file=sys.stderr)
        return 1
    leaks = held_out_leaks(files, split)
    for leak in leaks:
        print(f"::error::held-out date on the German page: {leak}")
    if not leaks:
        print(f"The German page names no nowcast date from {split} on.")
    return 1 if leaks else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
