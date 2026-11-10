"""Location names: hub locations, zones with cases, and the 11-07 flags."""

from pathlib import Path

import pandas as pd


def location_names(S: Path, data: Path) -> dict[str, str]:
    """Map location codes to names, falling back on the 11-07 flag files."""
    out: dict[str, str] = {}
    old = S / "flags2_1107.csv"
    if old.exists():
        f = pd.read_csv(old).dropna(subset=["name"])
        out |= dict(zip(f.location, f.name, strict=True))
    for path in [data / "locations.csv", S / "hub" / "ts.csv"]:
        t = pd.read_csv(path).drop_duplicates("location")
        out |= dict(zip(t.location, t.location_name, strict=True))
    return out
