# Copyright (c) 2026 Sam Abbott. MIT licence, see LICENSE.
"""Health zone boundaries for the zone map on the Nowcast page.

The boundaries are BVDOutbreakSize's `src/assets/health_zones.geojson`
(MIT licence), built from the INRB-UMIE map of DRC health zones, whose
source is the Ministry of Health boundaries on the Humanitarian Data
Exchange. They are read at the BVDOutbreakSize release tag the
`renewalbvd` model uses for the release (`bvd_for_vintage`).

Each feature is joined to its hub zone code on its DHIS2 code (`zscode`)
through the hub's `auxiliary-data/health-zone-code-mapping.csv`, at the
hub commit in `data/sources.json`. Only the geometry and `zscode` are
read. A feature whose code the mapping lacks is left out.
"""

from __future__ import annotations

import io
import json
from typing import Any, Final

import pandas as pd

from bdbvhubrenewal.data.bvd import BVD_REPO, Downloader, _download
from bdbvhubrenewal.data.tidy import HUB_REPO

__all__ = ["BOUNDARIES_PATH", "MAPPING_PATH", "zone_boundaries"]

BOUNDARIES_PATH: Final = "src/assets/health_zones.geojson"
"""The boundaries in the BVDOutbreakSize repository."""

MAPPING_PATH: Final = "auxiliary-data/health-zone-code-mapping.csv"
"""The hub's zone codes with their DHIS2 codes."""


def _raw(repo: str, ref: str, path: str) -> str:
    return f"https://raw.githubusercontent.com/{repo}/{ref}/{path}"


def zone_boundaries(
    bvd_tag: str, hub_commit: str, download: Downloader = _download
) -> dict[str, Any]:
    """Return the hub's health zones as GeoJSON keyed by hub code.

    Parameters
    ----------
    bvd_tag : str
        The BVDOutbreakSize release tag to read the boundaries at.
    hub_commit : str
        The hub commit to read the code mapping at.
    download : callable
        Returns the bytes at a URL.

    Returns
    -------
    dict
        A GeoJSON FeatureCollection whose features carry only the
        geometry and `location`, the hub zone code.
    """
    geo = json.loads(download(_raw(BVD_REPO, bvd_tag, BOUNDARIES_PATH)))
    mapping = pd.read_csv(
        io.BytesIO(download(_raw(HUB_REPO, hub_commit, MAPPING_PATH)))
    )
    codes = dict(zip(mapping["Code_DHIS2"], mapping["location"], strict=True))
    features = [
        {
            "type": "Feature",
            "properties": {"location": codes[f["properties"]["zscode"]]},
            "geometry": f["geometry"],
        }
        for f in geo["features"]
        if f["properties"]["zscode"] in codes
    ]
    return {"type": "FeatureCollection", "features": features}
