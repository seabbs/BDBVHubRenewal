"""The simulator's pins, read from scripts/finetune_simulate.R itself."""

from __future__ import annotations

import re
from pathlib import Path

SCRIPT = Path(__file__).parents[2] / "scripts" / "finetune_simulate.R"


def epinowcast_sha(script: Path | str = SCRIPT) -> str:
    """Return the epinowcast commit the simulator is pinned to.

    Raises `ValueError` if the script sets no `epinowcast_sha`.
    """
    text = Path(script).read_text()
    found = re.search(r'epinowcast_sha <- "([0-9a-f]+)"', text)
    if not found:
        raise ValueError(f"{script} sets no epinowcast_sha")
    return found[1]
