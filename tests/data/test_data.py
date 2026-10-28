from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from bdbvhubrenewal.data import (
    PROVINCE_CENTRES,
    PROVINCE_POPULATIONS,
    imputed_share,
    load_health_zones,
    load_locations,
    load_province_centres,
    newly_missing,
    revision_table,
    tidy_daily,
    tidy_delays,
    tidy_locations,
    tidy_target,
    vintages,
)

DATA = Path(__file__).resolve().parents[2] / "data"


def _write(path: Path, rows: list[dict[str, object]]) -> Path:
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


@pytest.fixture
def raw(tmp_path: Path) -> dict[str, Path]:
    loc = _write(
        tmp_path / "locations.csv",
        [
            {
                "location": "CD",
                "location_name": "DRC",
                "location_type": "country",
                "parent": "",
            },
            {
                "location": "CD-IT",
                "location_name": "Ituri",
                "location_type": "province",
                "parent": "DRC",
            },
            {
                "location": "CD-NK",
                "location_name": "Nord Kivu",
                "location_type": "province",
                "parent": "DRC",
            },
            {
                "location": "CD-IT-01",
                "location_name": "Zone",
                "location_type": "health zone",
                "parent": "Ituri",
            },
        ],
    )
    daily = _write(
        tmp_path / "daily.csv",
        [
            # 4 and 7 Oct are both in the week ending 10 Oct.
            {
                "as_of": "2026-10-17",
                "target_end_date": "2026-10-04",
                "location": "CD-IT",
                "location_type": "province",
                "observation": 2,
                "imputed": False,
            },
            {
                "as_of": "2026-10-17",
                "target_end_date": "2026-10-07",
                "location": "CD-IT",
                "location_type": "province",
                "observation": 1,
                "imputed": True,
            },
            {
                "as_of": "2026-10-17",
                "target_end_date": "2026-10-07",
                "location": "CD-IT-01",
                "location_type": "health zone",
                "observation": 1,
                "imputed": True,
            },
        ],
    )
    ts = _write(
        tmp_path / "ts.csv",
        [
            {
                "as_of": "2026-10-17",
                "target_end_date": "2026-10-10",
                "target": "wk inc confirmed cases",
                "location": "CD-IT",
                "location_name": "Ituri",
                "location_type": "province",
                "observation": 3,
            },
            {
                "as_of": "2026-10-17",
                "target_end_date": "2026-10-10",
                "target": "wk inc confirmed cases",
                "location": "CD-IT-01",
                "location_name": "Zone",
                "location_type": "health zone",
                "observation": 1,
            },
        ],
    )
    old = _write(
        tmp_path / "2026-10-09.csv",
        [
            {
                "location": "IT",
                "location_name": "Ituri",
                "week_ending": "2026-10-03",
                "days_sym_to_report": 4,
            }
        ],
    )
    new = _write(
        tmp_path / "2026-10-17.csv",
        [
            {
                "week_ending": "2026-10-10",
                "location_name": "Nord Kivu",
                "days_sym_to_report": 2,
                "as_of": "2026-10-17",
            }
        ],
    )
    return {"loc": loc, "daily": daily, "ts": ts, "old": old, "new": new}


def test_tidy_target_splits_observed_and_imputed(raw: dict[str, Path]) -> None:
    daily = tidy_daily(raw["daily"])
    assert set(daily["location"]) == {"CD-IT"}
    target = tidy_target(raw["ts"], daily)
    row = target.iloc[0]
    assert (row["observed"], row["imputed"], row["total"]) == (2, 1, 3)
    assert row["kind"] == "real"


def test_tidy_target_rejects_inconsistent_split(
    raw: dict[str, Path],
) -> None:
    daily = tidy_daily(raw["daily"])
    daily.loc[0, "observed"] = 5
    with pytest.raises(ValueError, match="observed plus imputed"):
        tidy_target(raw["ts"], daily)


def test_tidy_delays_maps_old_and_new_formats(raw: dict[str, Path]) -> None:
    locations = tidy_locations(raw["loc"])
    delays = tidy_delays(
        {"2026-10-09": raw["old"], "2026-10-17": raw["new"]}, locations
    )
    assert list(delays["location"]) == ["CD-IT", "CD-NK"]
    assert list(delays["vintage"]) == ["2026-10-09", "2026-10-17"]
    assert list(delays["delay"]) == [4, 2]


def test_tidy_locations_adds_populations(raw: dict[str, Path]) -> None:
    loc = tidy_locations(raw["loc"]).set_index("location")
    assert loc.loc["CD-IT", "population"] == PROVINCE_POPULATIONS["CD-IT"]
    assert loc.loc["CD", "population"] == sum(PROVINCE_POPULATIONS.values())
    assert "CD-IT-01" not in loc.index


def test_committed_target_is_consistent(target: pd.DataFrame) -> None:
    assert vintages(target) == [
        "2026-08-18",
        "2026-10-09",
        "2026-10-17",
        "2026-10-24",
    ]
    assert (target["observed"] + target["imputed"] == target["total"]).all()
    assert set(target["kind"]) == {"real"}
    # The national row equals the sum of the provinces.
    prov = target.loc[target["location"] != "CD"]
    sums = prov.groupby(["vintage", "target_end_date"])["total"].sum()
    nat = target.loc[target["location"] == "CD"].set_index(
        ["vintage", "target_end_date"]
    )["total"]
    pd.testing.assert_series_equal(
        sums.sort_index(), nat.sort_index(), check_names=False
    )


def test_committed_delays_match_known_onsets(
    target: pd.DataFrame, delays: pd.DataFrame
) -> None:
    rows = delays.groupby("vintage").size()
    nat = (
        target.loc[target["location"] == "CD"]
        .groupby("vintage")["observed"]
        .sum()
    )
    for vintage, n in rows.items():
        assert n == nat[vintage]


def test_other_committed_tables_load() -> None:
    assert len(load_locations(DATA)) == 27
    assert (load_health_zones(DATA)["total"] > 0).all()


def test_imputed_share_and_newly_missing(target: pd.DataFrame) -> None:
    share = imputed_share(target)
    valid = share["imputed_share"].dropna()
    assert valid.between(0, 1).all()
    new = newly_missing(target)
    nat = new.loc[new["location"] == "CD"]
    assert list(nat["vintage"]) == ["2026-10-09", "2026-10-17", "2026-10-24"]
    assert (nat["new_observed"] + nat["new_imputed"] == nat["new_total"]).all()


def test_revision_table_latest_is_one(target: pd.DataFrame) -> None:
    rev = revision_table(target.loc[target["location"] == "CD"])
    latest = rev.loc[rev["vintage"] == rev["vintage"].max()]
    assert latest["fraction"].dropna().eq(1).all()
    assert (latest["weeks_before"] >= 0).all()


def test_province_centres_cover_every_province() -> None:
    centres = load_province_centres()
    assert set(centres["location"]) == set(PROVINCE_POPULATIONS)
    assert set(PROVINCE_CENTRES) == set(PROVINCE_POPULATIONS)
    # The eight provinces with cases match BVDOutbreakSize's centres.
    old = {
        "CD-IT": (2.0929, 30.3385),
        "CD-NK": (-0.5754, 29.1574),
        "CD-SK": (-2.869, 28.6843),
        "CD-HU": (2.915, 28.5755),
        "CD-TO": (0.6072, 24.8468),
        "CD-BU": (3.3937, 25.1094),
        "CD-SU": (3.0653, 19.4528),
        "CD-NU": (3.8889, 20.9641),
    }
    for loc, (lat, lon) in old.items():
        assert PROVINCE_CENTRES[loc] == pytest.approx((lat, lon), abs=1e-3)


def test_tidy_locations_has_a_centre_for_every_province() -> None:
    table = load_locations(DATA)
    provinces = table.loc[table["location_type"] == "province"]
    assert len(provinces) == 26
    assert provinces[["latitude", "longitude"]].notna().all().all()


def test_tidy_daily_keeps_zones_apart(raw: dict[str, Path]) -> None:
    zones = tidy_daily(raw["daily"], level="zone")
    assert set(zones["location"]) == {"CD-IT-01"}
    assert list(zones.columns) == [
        "vintage",
        "location",
        "onset_date",
        "observed",
        "imputed",
    ]
    assert "CD-IT-01" not in set(tidy_daily(raw["daily"])["location"])
    with pytest.raises(ValueError, match="level"):
        tidy_daily(raw["daily"], level="ward")


def test_load_daily_zones_reads_the_tidy_file(tmp_path: Path) -> None:
    from bdbvhubrenewal.data import load_daily_zones

    pd.DataFrame(
        {
            "vintage": ["2026-10-24"],
            "location": ["CD-IT-01"],
            "onset_date": ["2026-10-01"],
            "observed": [1],
            "imputed": [0],
        }
    ).to_csv(tmp_path / "daily_zones.csv.gz", index=False)
    assert list(load_daily_zones(tmp_path)["location"]) == ["CD-IT-01"]
