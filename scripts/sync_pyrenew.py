# Copyright (c) 2026 Sam Abbott. MIT licence, see LICENSE.
"""Sync the PyRenew model code from the prototype into the package.

Usage: `uv run python scripts/sync_pyrenew.py <prototype dir>`.

Copies the model modules of the prototype (`src/province_forecast/`) into
`src/bdbv_hub_renewal/pyrenew/` and its province centres into
`src/bdbv_hub_renewal/province_centres.csv` (package data, read by
`bdbv_hub_renewal.data.load_province_centres`). References to the old package name
`bdbv_forecasts` become `bdbv_hub_renewal`. The adapter
(`src/bdbv_hub_renewal/models/renewal.py`) is owned by this repository
and is not synced. Run `task format` afterwards.
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

MODULES = ("processes", "latent", "delays", "observation", "model")
ROOT = Path(__file__).resolve().parent.parent
DEST = ROOT / "src" / "bdbv_hub_renewal" / "pyrenew"
HEADER = (
    "# Copyright (c) 2026 Sam Abbott. MIT licence, see LICENSE.\n"
    "# Synced from the prototype by scripts/sync_pyrenew.py. Edit there.\n"
)


def main(prototype: str) -> None:
    """Copy the prototype's model modules and data into the package."""
    src = Path(prototype) / "src" / "province_forecast"
    DEST.mkdir(parents=True, exist_ok=True)
    for name in MODULES:
        text = (src / f"{name}.py").read_text()
        text = text.replace("bdbv_forecasts", "bdbv_hub_renewal")
        (DEST / f"{name}.py").write_text(HEADER + text)
    shutil.copy(
        src / "data" / "province_centres.csv",
        ROOT / "src" / "bdbv_hub_renewal" / "province_centres.csv",
    )
    print(f"Synced {', '.join(MODULES)} from {src} into {DEST}.")


if __name__ == "__main__":
    main(sys.argv[1])
