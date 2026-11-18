# Copyright (c) 2026 Sam Abbott. MIT licence, see LICENSE.
"""Build `src/bdbvhubrenewal/data/zone_centres.csv`.

Reads the INRB-UMIE build of the DRC health zone map, the source of
`province_centres.csv`, and the hub's zone code mapping, and writes one
row per hub zone: its centroid and WorldPop 2025 count. The centroid is
the area-weighted centroid of the zone's largest ring, as
BVDOutbreakSize computes it.

Usage: uv run python scripts/build_zone_centres.py GEOJSON MAPPING
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

OUT = Path("src/bdbvhubrenewal/data/zone_centres.csv")
HEADER = (
    "# Area-weighted centroids of each health zone's largest ring and "
    "WorldPop 2025 counts. Source: INRB-UMIE/BDBV2026-Data@f7d3907 "
    "build/drc_health_zones.geojson (MoH DRC_Health_zones via HDX, ODbL "
    "1.0; WorldPop 2025, CC BY 4.0), joined to hub codes on DHIS2 code "
    "(auxiliary-data/health-zone-code-mapping.csv). Built by "
    "scripts/build_zone_centres.py.\n"
)


def ring_centroid(ring: np.ndarray) -> tuple[float, float, float]:
    """Return the area and centroid (lon, lat) of a closed ring."""
    x, y = ring[:, 0], ring[:, 1]
    cross = x[:-1] * y[1:] - x[1:] * y[:-1]
    area = cross.sum() / 2
    if area == 0:
        return 0.0, float(x.mean()), float(y.mean())
    cx = ((x[:-1] + x[1:]) * cross).sum() / (6 * area)
    cy = ((y[:-1] + y[1:]) * cross).sum() / (6 * area)
    return abs(float(area)), float(cx), float(cy)


def largest_ring(geometry: dict) -> np.ndarray:
    """Return the outer ring with the largest area."""
    polygons = (
        [geometry["coordinates"]]
        if geometry["type"] == "Polygon"
        else geometry["coordinates"]
    )
    rings = [np.asarray(p[0], dtype=float) for p in polygons]
    return max(rings, key=lambda r: ring_centroid(r)[0])


def main(geojson: str, mapping: str) -> None:
    """Write the zone centres."""
    features = json.loads(Path(geojson).read_text())["features"]
    # Two Kinshasa zones share one DHIS2 code and so one feature; both
    # take it. Zones with no code are left out.
    codes = pd.read_csv(mapping).dropna(subset=["Code_DHIS2"])
    by_code = codes.groupby("Code_DHIS2")["location"].apply(list)
    rows = []
    for f in features:
        props = f["properties"]
        if props["zscode"] not in by_code.index:
            continue
        _, lon, lat = ring_centroid(largest_ring(f["geometry"]))
        pop = round(props["worldpop"]["pop_count"]["pop_count"])
        for code in by_code[props["zscode"]]:
            rows.append((code, round(lat, 4), round(lon, 4), pop))
    table = pd.DataFrame(
        rows, columns=["location", "latitude", "longitude", "worldpop"]
    ).sort_values("location")
    with OUT.open("w") as fh:
        fh.write(HEADER)
        table.to_csv(fh, index=False)
    print(f"wrote {len(table)} zones to {OUT}")


if __name__ == "__main__":
    main(*sys.argv[1:3])
