from __future__ import annotations

import datetime as dt
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from bdbv_hub_renewal import bvd

FIXTURES = Path(__file__).resolve().parent / "data" / "bvd"

AS_OF = {
    "results-3169": "2026-10-20",
    "results-3151": "2026-10-20",
    "results-3160": "2026-10-22",
    "results-3086": "2026-10-15",
    "results-2007": "2026-09-08",
    "results-2005": "2026-09-09",
}


@pytest.fixture
def releases() -> pd.DataFrame:
    return bvd.parse_releases(
        json.loads((FIXTURES / "releases.json").read_text())
    )


def test_parse_releases_keeps_published_main_builds(
    releases: pd.DataFrame,
) -> None:
    # No drafts, no `v*` and no `results-v*` tags.
    assert set(releases["tag"]) == set(AS_OF)
    assert releases["published_at"].dt.tz is not None
    # Newest publication first, not highest run number.
    assert list(releases["tag"])[-2:] == ["results-2005", "results-2007"]
    row = releases.set_index("tag").loc["results-3086"]
    assert set(row["assets"]) == {"posterior_draws.csv", "observations.toml"}


def test_cutoff_is_the_end_of_the_sunday_opening_the_window() -> None:
    cut = bvd.bvd_cutoff("2026-10-24")
    assert cut == dt.datetime(2026, 10, 25, 23, 59, 59, tzinfo=bvd.HUB_TIMEZONE)
    # A vintage published after the window opened is its own cut-off.
    late = bvd.bvd_cutoff("2026-08-18")
    assert late.date() == dt.date(2026, 8, 18)


def test_republished_old_data_does_not_feed_an_earlier_vintage(
    releases: pd.DataFrame,
) -> None:
    # results-3169 has as_of 20 October but was published on 27 October,
    # after the 24 October vintage's cut-off. results-3160 was published
    # in time, but an asset was replaced after the cut-off.
    tag = bvd.select_release(
        releases, "2026-10-24", bvd.bvd_cutoff("2026-10-24"), AS_OF.get
    )
    assert tag == "results-3151"


def test_select_release_needs_as_of_on_or_before_the_forecast_date(
    releases: pd.DataFrame,
) -> None:
    cutoff = bvd.bvd_cutoff("2026-10-24") + dt.timedelta(days=30)
    # Every release is published by then, so the newest publication whose
    # data end by the forecast date wins.
    assert (
        bvd.select_release(releases, "2026-10-22", cutoff, AS_OF.get)
        == "results-3169"
    )
    assert (
        bvd.select_release(releases, "2026-10-21", cutoff, AS_OF.get)
        == "results-3169"
    )
    assert (
        bvd.select_release(releases, "2026-10-17", cutoff, AS_OF.get)
        == "results-3086"
    )


def test_select_release_orders_by_publication(releases: pd.DataFrame) -> None:
    cutoff = dt.datetime(2026, 9, 12, tzinfo=dt.UTC)
    tag = bvd.select_release(releases, "2026-09-12", cutoff, AS_OF.get)
    assert tag == "results-2005"
    assert bvd.select_release(releases, "2026-09-01", cutoff, AS_OF.get) is None


def test_load_release_with_patches() -> None:
    rel = bvd.load_release(FIXTURES / "results-3151")
    assert rel.tag == "results-3151"
    assert rel.as_of == dt.date(2026, 10, 20)
    draws = pd.read_csv(FIXTURES / "results-3151" / "posterior_draws.csv")
    np.testing.assert_allclose(rel.national_log_rt, np.log(draws["R_T"]))
    assert len(rel.patches) == 8
    ituri = rel.patches["Ituri"]
    assert ituri.members == ("CD-IT",)
    assert ituri.log_median == pytest.approx(np.log(1.0429598420571795))
    width = np.log(1.5193759833646396 / 0.7128676497094169)
    assert ituri.log_sd == pytest.approx(width / (2 * 1.6448536), rel=1e-6)


def test_load_release_maps_the_pooled_patch() -> None:
    rel = bvd.load_release(FIXTURES / "results-2817")
    assert set(rel.patches) == {
        "Ituri",
        "Nord-Kivu",
        "Haut-Uele",
        "Other provinces",
    }
    other = rel.patches["Other provinces"]
    assert other.members == ("CD-SK", "CD-TO", "CD-BU", "CD-SU")
    assert other.pooled


def test_load_release_without_patch_assets_is_national_only() -> None:
    rel = bvd.load_release(FIXTURES / "results-3086")
    assert rel.as_of == dt.date(2026, 10, 15)
    assert rel.patches == {}
    assert rel.national_log_rt.size == 40


def test_unknown_patch_key_raises(tmp_path: Path) -> None:
    shutil.copytree(FIXTURES / "results-3151", tmp_path / "results-1")
    path = tmp_path / "results-1" / "province_estimates.csv"
    est = pd.read_csv(path)
    est.loc[0, "province"] = "kinshasa"
    est.to_csv(path, index=False)
    with pytest.raises(ValueError, match="kinshasa"):
        bvd.load_release(tmp_path / "results-1")


def test_fetch_release_caches_by_tag(
    releases: pd.DataFrame, tmp_path: Path
) -> None:
    got: list[str] = []

    def download(url: str) -> bytes:
        got.append(url)
        name = url.rsplit("/", 1)[1]
        return (FIXTURES / "results-3151" / name).read_bytes()

    row = releases.set_index("tag").loc["results-3151"]
    out = bvd.fetch_release("results-3151", row["assets"], tmp_path, download)
    assert out == tmp_path / "results-3151"
    assert {p.name for p in out.iterdir()} == set(bvd.ASSETS)
    n = len(got)
    bvd.fetch_release("results-3151", row["assets"], tmp_path, download)
    assert len(got) == n


def test_bvd_for_vintage_end_to_end(
    releases: pd.DataFrame, tmp_path: Path
) -> None:
    def download(url: str) -> bytes:
        tag, name = url.rsplit("/", 2)[1:]
        return (FIXTURES / tag / name).read_bytes()

    # Only the fixture releases with files can be read.
    have = releases.loc[releases["tag"].isin(["results-3151", "results-3086"])]
    rel = bvd.bvd_for_vintage("2026-10-24", tmp_path, have, download)
    assert rel is not None
    assert rel.tag == "results-3151"
    rel = bvd.bvd_for_vintage("2026-10-17", tmp_path, have, download)
    assert rel is not None
    assert rel.tag == "results-3086"
    assert bvd.bvd_for_vintage("2026-10-01", tmp_path, have, download) is None


def test_model_rows_use_one_table_for_patch_deviations() -> None:
    rel = bvd.load_release(FIXTURES / "results-3151")
    rows = bvd.model_rows(rel)
    assert list(rows.columns) == [
        "group",
        "members",
        "as_of",
        "log_mean",
        "log_sd",
        "ref_log_mean",
        "ref_log_sd",
    ]
    nat = rows.iloc[0]
    assert nat["group"] == "national"
    assert nat["log_mean"] == pytest.approx(rel.national_log_rt.mean())
    ituri = rows.set_index("group").loc["Ituri"]
    assert ituri["members"] == "CD-IT"
    # The reference is the time series' own national row at as_of.
    assert ituri["ref_log_mean"] == pytest.approx(np.log(1.109))
    width = np.log(1.459 / 0.78) / (2 * 1.6448536)
    assert ituri["ref_log_sd"] == pytest.approx(width, rel=1e-6)


def test_model_rows_national_only() -> None:
    rows = bvd.model_rows(bvd.load_release(FIXTURES / "results-3086"))
    assert list(rows["group"]) == ["national"]
    assert rows["as_of"].iloc[0] == "2026-10-15"


def test_releases_round_trip_through_the_cache(
    releases: pd.DataFrame, tmp_path: Path
) -> None:
    bvd.save_releases(releases, tmp_path)
    back = bvd.load_releases(tmp_path)
    assert back is not None
    pd.testing.assert_frame_equal(
        back.drop(columns="assets"), releases.drop(columns="assets")
    )
    assert back["assets"].tolist() == releases["assets"].tolist()
    assert bvd.load_releases(tmp_path / "empty") is None


def test_bvd_for_vintage_reads_the_cached_release_list(
    releases: pd.DataFrame, tmp_path: Path
) -> None:
    have = releases.loc[releases["tag"].isin(["results-3151", "results-3086"])]
    bvd.save_releases(have, tmp_path)
    for tag in ("results-3151", "results-3086"):
        shutil.copytree(FIXTURES / tag, tmp_path / tag)

    def offline(url: str) -> bytes:
        raise AssertionError(f"downloaded {url}")

    rel = bvd.bvd_for_vintage("2026-10-24", tmp_path, download=offline)
    assert rel is not None
    assert rel.tag == "results-3151"


def test_refresh_cache_fetches_the_release_of_each_vintage(
    releases: pd.DataFrame, tmp_path: Path
) -> None:
    def download(url: str) -> bytes:
        tag, name = url.rsplit("/", 2)[1:]
        return (FIXTURES / tag / name).read_bytes()

    have = releases.loc[releases["tag"].isin(["results-3151", "results-3086"])]
    chosen = bvd.refresh_cache(
        tmp_path, ["2026-10-24", "2026-10-17", "2026-10-01"], have, download
    )
    assert chosen == {
        "2026-10-24": "results-3151",
        "2026-10-17": "results-3086",
        "2026-10-01": None,
    }
    assert bvd.load_releases(tmp_path) is not None
    assert (tmp_path / "results-3151" / "posterior_draws.csv").exists()
