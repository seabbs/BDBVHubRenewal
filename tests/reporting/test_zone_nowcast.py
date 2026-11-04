from __future__ import annotations

import json

import numpy as np
import pandas as pd

from bdbvhubrenewal.reporting.zone_nowcast import (
    ZONE_MODEL,
    place_chart_data,
    zone_counts,
    zone_table,
)

V = "2026-11-01"
NAMES = {"CD-IT": "Ituri", "CD-NK": "Nord Kivu"}


def _daily() -> pd.DataFrame:
    rows = [
        ("CD-IT-01", "2026-10-13", 2),
        ("CD-IT-01", "2026-10-28", 1),
        ("CD-NK-02", "2026-10-21", 5),
        ("CD-NK-03", "2026-08-01", 4),
    ]
    return pd.DataFrame(
        [
            {
                "vintage": V,
                "location": z,
                "onset_date": d,
                "observed": n,
                "imputed": 0,
            }
            for z, d, n in rows
        ]
    )


def _samples() -> pd.DataFrame:
    rng = np.random.default_rng(1)
    weeks = ["2026-10-24", "2026-10-31", "2026-11-07"]
    return pd.DataFrame(
        [
            {
                "location": z,
                "target_end_date": w,
                "sample": s,
                "value": float(rng.poisson(3)),
                "model": ZONE_MODEL,
                "vintage": V,
                "reference_date": "2026-11-07",
                "horizon": h,
            }
            for z in ("CD-IT-01", "CD-NK-02")
            for h, w in zip((-2, -1, 0), weeks, strict=True)
            for s in range(20)
        ]
    )


def _scores() -> pd.DataFrame:
    rows = []
    for model, wis in (
        ("randomwalk", 3.0),
        ("randomwalk (summed to provinces)", 2.0),
        (ZONE_MODEL + " (summed to provinces)", 1.0),
    ):
        rows.append(
            {
                "model": model,
                "origin": "2026-10-24",
                "target": "reported",
                "truth_vintage": V,
                "location": "CD-IT-01",
                "target_end_date": "2026-10-24",
                "wis": wis,
                "log_wis": wis,
                "bias": 0.0,
                "coverage_50": 1.0,
                "coverage_90": 1.0,
            }
        )
    return pd.DataFrame(rows)


def test_zone_counts_fill_quiet_weeks_with_zero() -> None:
    counts = zone_counts(_daily(), V, "2026-10-10", NAMES)
    # CD-NK-03 has no cases since the start, so it is left out.
    assert set(counts["location"]) == {"CD-IT-01", "CD-NK-02"}
    it = counts.loc[counts["location"] == "CD-IT-01"]
    assert it["target_end_date"].tolist() == [
        "2026-10-10",
        "2026-10-17",
        "2026-10-24",
        "2026-10-31",
    ]
    assert it["total"].tolist() == [0, 2, 0, 1]
    assert set(it["location_name"]) == {"Ituri 01"}


def test_zone_table_has_the_newest_weeks_and_scores() -> None:
    counts = zone_counts(_daily(), V, "2026-10-10", NAMES)
    table = zone_table(_samples(), counts, _scores())
    assert set(table["week ending"]) == {"2026-10-24", "2026-10-31"}
    it = table.loc[table["zone"] == "CD-IT-01"]
    assert it["relative WIS"].tolist() == [0.5, 0.5]
    assert table.loc[table["zone"] == "CD-NK-02", "n"].isna().all()


COLOURS = {
    "randomwalk": ("#111111", "#eeeeee"),
    ZONE_MODEL: ("#222222", "#dddddd"),
    "renewalbvd": ("#333333", "#cccccc"),
}


def _all_scales() -> tuple[pd.DataFrame, pd.DataFrame]:
    zones = zone_counts(_daily(), V, "2026-10-10", NAMES)
    national = pd.DataFrame(
        {
            "location": ["CD", "CD-IT", "CD-NK"],
            "target_end_date": "2026-10-17",
            "total": [7, 2, 5],
        }
    )
    counts = pd.concat(
        [national, zones[["location", "target_end_date", "total"]]],
        ignore_index=True,
    )
    big = _samples().loc[lambda x: x["location"] == "CD-NK-02"]
    samples = pd.concat(
        [_samples(), big.assign(location="CD"), big.assign(location="CD-NK")],
        ignore_index=True,
    )
    return samples, counts


def test_place_chart_data_orders_scales_then_cases() -> None:
    samples, counts = _all_scales()
    data = place_chart_data(samples, counts, NAMES, COLOURS)
    json.dumps(data)
    places = [(p["id"], p["scale"]) for p in data["places"]]
    assert places == [
        ("CD", "national"),
        ("CD-NK", "provinces"),
        ("CD-IT", "provinces"),
        ("CD-NK-02", "zones"),
        ("CD-IT-01", "zones"),
    ]
    assert data["places"][1]["name"] == "Nord Kivu"
    assert data["places"][3]["province"] == "CD-NK"
    assert data["data"]["CD-IT-01"][1] == ["2026-10-17", 2]
    assert len(data["q"]["CD"]) == 3
    # All 20 draws are kept (under 50), each over the three weeks.
    traj = data["traj"]["CD-NK-02"]
    assert len(traj) == 20
    assert traj[0][:3] == [1, 0, "2026-10-24"]


def test_place_chart_data_marks_models_without_samples() -> None:
    samples, counts = _all_scales()
    data = place_chart_data(samples, counts, NAMES, COLOURS)
    fans = {m["name"]: m["fan"] for m in data["models"]}
    assert fans == {"randomwalk": False, ZONE_MODEL: True, "renewalbvd": False}
    assert data["models"][2]["dark"] == "#cccccc"
