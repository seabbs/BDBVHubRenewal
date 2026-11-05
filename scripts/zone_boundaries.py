# Copyright (c) 2026 Sam Abbott. MIT licence, see LICENSE.
r"""Build the health zone boundaries the Nowcast page's zone map draws.

The boundaries are the Ministry of Health (DSNIS) health zones on the
Humanitarian Data Exchange, "DRC Health Zone and Health Area boundaries"
(https://data.humdata.org/dataset/drc-health-data, resource
"RDC_Zones de santé.zip", ODbL). The hub's
`auxiliary-data/health-zone-code-mapping.csv` maps each hub zone code to
the P-code those boundaries carry, so zones are joined on the P-code and
only the hub's zones are kept. Shapes are simplified to about 1 km and
coordinates rounded to 0.001 degrees, and each feature keeps only its hub
code as `location`.

Usage::

    uv run --with pyshp --with shapely python scripts/zone_boundaries.py \
        rdc_zones-de-sante.zip health-zone-code-mapping.csv \
        docs/assets/zone-boundaries.geojson
"""

from __future__ import annotations

import csv
import io
import json
import re
import sys
import zipfile
from pathlib import Path

import shapefile
from shapely.geometry import mapping, shape

PCODE = re.compile(r"^CD\d{4}ZS\d{2}$")
"""A health zone P-code, as in the hub's mapping."""

TOLERANCE = 0.01
"""Simplification tolerance in degrees, about 1 km."""


def read_shapes(path: Path) -> shapefile.Reader:
    """Open the one shapefile inside the HDX zip."""
    z = zipfile.ZipFile(path)
    stem = next(n for n in z.namelist() if n.lower().endswith(".shp"))[:-4]
    part = {
        ext: io.BytesIO(z.read(stem + "." + ext))
        for ext in ("shp", "shx", "dbf")
    }
    return shapefile.Reader(**part)


def pcode_field(reader: shapefile.Reader) -> str:
    """Return the attribute that holds the zone P-codes."""
    names = [f[0] for f in reader.fields[1:]]
    records = reader.records()
    hits = {
        n: sum(bool(PCODE.match(str(r[n]))) for r in records) for n in names
    }
    return max(hits, key=hits.__getitem__)


def rounded(geom: object) -> object:
    """Round every coordinate to 0.001 degrees."""
    if isinstance(geom, (list, tuple)):
        if geom and isinstance(geom[0], float):
            return [round(v, 3) for v in geom]
        return [rounded(g) for g in geom]
    return geom


def main(zip_path: str, mapping_csv: str, out: str) -> None:
    """Write the hub's zones from the HDX zip as GeoJSON to `out`."""
    with Path(mapping_csv).open(newline="") as fh:
        codes = {r["Pcode"]: r["location"] for r in csv.DictReader(fh)}
    reader = read_shapes(Path(zip_path))
    field = pcode_field(reader)
    features = []
    for sr in reader.iterShapeRecords():
        loc = codes.get(str(sr.record[field]))
        if loc is None:
            continue
        geom = mapping(shape(sr.shape.__geo_interface__).simplify(TOLERANCE))
        features.append(
            {
                "type": "Feature",
                "properties": {"location": loc},
                "geometry": {
                    "type": geom["type"],
                    "coordinates": rounded(geom["coordinates"]),
                },
            }
        )
    missing = sorted(
        set(codes.values()) - {f["properties"]["location"] for f in features}
    )
    if missing:
        print(f"No boundary for {len(missing)} hub zones: {', '.join(missing)}")
    features.sort(key=lambda f: f["properties"]["location"])
    Path(out).write_text(
        json.dumps(
            {"type": "FeatureCollection", "features": features},
            separators=(",", ":"),
        )
    )
    print(f"Wrote {len(features)} zones to {out}")


if __name__ == "__main__":
    main(*sys.argv[1:])
