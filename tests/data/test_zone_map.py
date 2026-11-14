from __future__ import annotations

import json

from bdbvhubrenewal.data.zone_map import zone_boundaries, zone_names

SQUARE = {"type": "Polygon", "coordinates": [[[0, 0], [1, 0], [1, 1], [0, 0]]]}


def _download(url: str) -> bytes:
    if url.endswith("health_zones.geojson"):
        assert "/results-3180/" in url
        features = [
            {
                "type": "Feature",
                "properties": {"zscode": code, "zone": "x", "label": "y"},
                "geometry": SQUARE,
            }
            for code in ("abc", "unknown")
        ]
        return json.dumps(
            {"type": "FeatureCollection", "features": features}
        ).encode()
    assert "/deadbeef/" in url
    return (
        b"location,location_name,location_type,parent,Pcode,Code_DHIS2\n"
        b"CD-IT-01,Adi,health zone,Ituri,CD5401ZS01,abc\n"
    )


def test_zone_boundaries_key_features_by_hub_code() -> None:
    geo = zone_boundaries("results-3180", "deadbeef", _download)
    # The unmapped feature is left out, and only the hub code is kept.
    assert geo["features"] == [
        {
            "type": "Feature",
            "properties": {"location": "CD-IT-01"},
            "geometry": SQUARE,
        }
    ]


def test_zone_names_come_from_the_hub_mapping() -> None:
    assert zone_names("deadbeef", _download) == {"CD-IT-01": "Adi"}
