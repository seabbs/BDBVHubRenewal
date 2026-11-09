from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from bdbvhubrenewal.data import (
    PROVINCE_CENTRES,
    PROVINCE_POPULATIONS,
    cutoff_window,
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
    write_csv,
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
    # New releases add vintages, so only the first and the order are fixed.
    got = vintages(target)
    assert got[0] == "2026-08-18"
    assert got == sorted(set(target["vintage"]))
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
    table = load_locations(DATA)
    assert len(table) == 27
    provinces = table.loc[table["location_type"] == "province"]
    assert len(provinces) == 26
    assert provinces[["latitude", "longitude"]].notna().all().all()
    assert (load_health_zones(DATA)["total"] > 0).all()


def test_imputed_share_and_newly_missing(target: pd.DataFrame) -> None:
    share = imputed_share(target)
    valid = share["imputed_share"].dropna()
    assert valid.between(0, 1).all()
    new = newly_missing(target)
    nat = new.loc[new["location"] == "CD"]
    # Every vintage after the first has a change from the one before.
    assert list(nat["vintage"]) == vintages(target)[1:]
    assert (nat["new_observed"] + nat["new_imputed"] == nat["new_total"]).all()
    # Days between the releases' cut-offs, 10-24 to 10-31.
    assert nat.set_index("vintage").loc["2026-11-01", "days"] == 7


def test_revision_table_latest_is_one(target: pd.DataFrame) -> None:
    rev = revision_table(target.loc[target["location"] == "CD"])
    latest = rev.loc[rev["vintage"] == rev["vintage"].max()]
    assert latest["fraction"].dropna().eq(1).all()
    assert (latest["weeks_before"] >= 0).all()
    # Weeks count from the cut-off's week: 11-01 was cut on 10-31.
    newest = rev.loc[rev["vintage"] == "2026-11-01"]
    assert (
        newest.set_index("target_end_date").loc["2026-10-31"]["weeks_before"]
        == 0
    )


def test_cutoff_window_fills_days_and_flags_partial_week() -> None:
    # Cut-off on Wednesday 2026-11-04; 10-27 and 11-01 have no onsets.
    days = pd.date_range("2026-10-25", "2026-11-04").strftime("%Y-%m-%d")
    daily = pd.DataFrame(
        {
            "vintage": "2026-11-08",
            "location": "CD",
            "onset_date": days,
            "observed": range(1, 12),
            "imputed": 1,
            "cutoff": "2026-11-04",
        }
    )
    daily = daily.loc[~daily["onset_date"].isin(["2026-10-27", "2026-11-01"])]
    other = daily.assign(location="CD-IT")
    win = cutoff_window(pd.concat([daily, other]), "2026-11-08", "CD", 2)
    assert list(win["onset_date"]) == list(days)
    assert win.set_index("onset_date").loc["2026-10-27", "total"] == 0
    weeks = win.groupby("target_end_date").agg(
        total=("total", "sum"),
        days=("week_days", "first"),
        partial=("partial", "first"),
    )
    assert list(weeks.index) == ["2026-10-31", "2026-11-07"]
    assert weeks["days"].tolist() == [7, 4]
    assert weeks["partial"].tolist() == [False, True]
    # Observed 1..11 less the dropped days 3 and 8, plus one imputed each.
    assert weeks["total"].tolist() == [
        1 + 2 + 4 + 5 + 6 + 7 + 6,
        9 + 10 + 11 + 3,
    ]
    assert (
        win["week_total"] == win["target_end_date"].map(weeks["total"])
    ).all()


def test_cutoff_window_matches_target_on_real_data(
    target: pd.DataFrame,
) -> None:
    daily = pd.read_csv(DATA / "daily.csv")
    win = cutoff_window(daily, "2026-11-08", "CD", 4)
    weeks = win.groupby("target_end_date")["total"].sum()
    real = target.loc[
        (target["vintage"] == "2026-11-08") & (target["location"] == "CD")
    ].set_index("target_end_date")["total"]
    assert weeks.loc["2026-11-07"] == 68
    assert (weeks == real.loc[weeks.index]).all()


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


def test_release_updates_count_a_vanished_row_as_a_fall() -> None:
    from bdbvhubrenewal.data import release_updates

    counts = pd.DataFrame(
        {
            "vintage": ["2026-10-17"] * 3 + ["2026-10-24"] * 2,
            "location": ["A", "B", "A", "A", "A"],
            "target_end_date": [
                "2026-10-03",
                "2026-10-03",
                "2026-10-10",
                "2026-10-03",
                "2026-10-17",
            ],
            "total": [5, 2, 3, 4, 6],
            "cutoff": ["2026-10-16"] * 3 + ["2026-10-24"] * 2,
        }
    )
    out = release_updates(counts).set_index(["location", "target_end_date"])
    # B's row is gone, so it fell to zero; A's 10-10 row also fell to zero.
    assert out.loc[("B", "2026-10-03"), "change"] == -2
    assert out.loc[("A", "2026-10-10"), "change"] == -3
    assert out.loc[("A", "2026-10-03"), "change"] == -1
    # A week first published in the later release is not an update.
    assert ("A", "2026-10-17") not in out.index
    assert out.loc[("A", "2026-10-10"), "weeks_before"] == 1
    assert set(out["previous"]) == {"2026-10-17"}


def test_downward_summary_shares() -> None:
    from bdbvhubrenewal.data import downward_summary

    updates = pd.DataFrame(
        {"location": ["A", "A", "A", "B"], "change": [-3, 1, 0, 6]}
    )
    out = downward_summary(updates, by=("location",)).set_index("location")
    assert out.loc["A", "weeks"] == 3
    assert out.loc["A", "fell"] == 1
    assert out.loc["A", "cases_down"] == 3
    assert out.loc["A", "downward_share"] == pytest.approx(0.75)
    assert out.loc["B", "downward_share"] == 0
    assert out.loc["A", "largest_fall"] == 3


def test_imputed_by_release(target: pd.DataFrame) -> None:
    from bdbvhubrenewal.data import imputed_by_release

    out = imputed_by_release(target)
    cd = out.loc[out["location"] == "CD"]
    assert len(cd) == target["vintage"].nunique()
    assert cd["imputed_share"].between(0, 1).all()
    assert (out["total"] > 0).all()


def test_write_csv_writes_the_same_bytes_for_the_same_table(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    table = pd.DataFrame({"location": ["CD", "CD-IT"], "value": [1, 2]})
    path = tmp_path / "table.csv.gz"
    write_csv(table, path)
    first = path.read_bytes()
    # A later clock must not change the file.
    monkeypatch.setattr("time.time", lambda: 2e9)
    write_csv(table, path)
    assert path.read_bytes() == first
    pd.testing.assert_frame_equal(pd.read_csv(path), table)
