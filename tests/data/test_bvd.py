from __future__ import annotations

import datetime as dt
import io
import json
import shutil
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from bdbv_hub_renewal.data import bvd

FIXTURES = Path(__file__).resolve().parent / "bvd"

RHAT = {
    "results-3169": 1.03,
    "results-3151": 1.03,
    "results-3160": 1.03,
    "results-3086": 1.04,
    "results-2007": 1.02,
    "results-2005": 1.02,
}

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


def _site_zip(
    pages: dict[str, str] | None = None, rhat: str | None = None
) -> bytes:
    """A site.zip holding the given pages, the new diagnostics page by
    default, with the joint R-hat swapped for `rhat`."""
    if pages is None:
        pages = {
            "BVDOutbreakSize/dev/summary_assets/diagnostics.html": (
                "diagnostics.html"
            )
        }
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        for name, fixture in pages.items():
            html = (FIXTURES / "site" / fixture).read_text()
            if rhat is not None:
                html = html.replace(">1.041<", f">{rhat}<")
            z.writestr(name, html)
    return buf.getvalue()


def _tag(releases: pd.DataFrame, day: str, cutoff: dt.datetime) -> str | None:
    choice = bvd.select_release(releases, day, cutoff, AS_OF.get, RHAT.get)
    return None if choice is None else choice.tag


def test_parse_releases_keeps_published_main_builds(
    releases: pd.DataFrame,
) -> None:
    # No drafts, no `v*` and no `results-v*` tags.
    assert set(releases["tag"]) == set(AS_OF)
    assert releases["published_at"].dt.tz is not None
    # Newest publication first, not highest run number.
    assert list(releases["tag"])[-2:] == ["results-2005", "results-2007"]
    row = releases.set_index("tag").loc["results-3086"]
    assert set(row["assets"]) == {
        "posterior_draws.csv",
        "observations.toml",
        "site.zip",
    }


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
    choice = bvd.select_release(
        releases,
        "2026-10-24",
        bvd.bvd_cutoff("2026-10-24"),
        AS_OF.get,
        RHAT.get,
    )
    assert choice is not None
    assert choice.tag == "results-3151"


def test_select_release_needs_as_of_on_or_before_the_forecast_date(
    releases: pd.DataFrame,
) -> None:
    cutoff = bvd.bvd_cutoff("2026-10-24") + dt.timedelta(days=30)
    # Every release is published by then, so the newest publication whose
    # data end by the forecast date wins.
    assert _tag(releases, "2026-10-22", cutoff) == "results-3169"
    assert _tag(releases, "2026-10-21", cutoff) == "results-3169"
    assert _tag(releases, "2026-10-17", cutoff) == "results-3086"


def test_select_release_orders_by_publication(releases: pd.DataFrame) -> None:
    cutoff = dt.datetime(2026, 9, 12, tzinfo=dt.UTC)
    assert _tag(releases, "2026-09-12", cutoff) == "results-2005"
    assert _tag(releases, "2026-09-01", cutoff) is None


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
    # Only the data cut-off is kept from the (large) observations file.
    kept = (out / "observations.toml").read_text()
    assert kept == 'as_of_date = "2026-10-20"\n'


def test_bvd_for_vintage_end_to_end(
    releases: pd.DataFrame, tmp_path: Path
) -> None:
    def download(url: str) -> bytes:
        tag, name = url.rsplit("/", 2)[1:]
        if name == bvd.SITE_ASSET:
            return _site_zip()
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
        if name == bvd.SITE_ASSET:
            return _site_zip()
        return (FIXTURES / tag / name).read_bytes()

    have = releases.loc[releases["tag"].isin(["results-3151", "results-3086"])]
    chosen = bvd.refresh_cache(
        tmp_path, ["2026-10-24", "2026-10-17", "2026-10-01"], have, download
    )
    assert {k: None if v is None else v.tag for k, v in chosen.items()} == {
        "2026-10-24": "results-3151",
        "2026-10-17": "results-3086",
        "2026-10-01": None,
    }
    assert bvd.load_releases(tmp_path) is not None
    assert (tmp_path / "results-3151" / "posterior_draws.csv").exists()


def test_early_releases_give_rt_from_the_growth_rate() -> None:
    # Early builds publish the growth rate r but not R_T.
    rel = bvd.load_release(FIXTURES / "results-657")
    draws = pd.read_csv(FIXTURES / "results-657" / "posterior_draws.csv")
    k = (bvd.BVD_GI_MEAN / bvd.BVD_GI_SD) ** 2
    theta = bvd.BVD_GI_SD**2 / bvd.BVD_GI_MEAN
    expected = k * np.log1p(draws["r"].to_numpy() * theta)
    np.testing.assert_allclose(rel.national_log_rt, expected)


def test_joint_rhat_from_the_diagnostics_page() -> None:
    html = (FIXTURES / "site" / "diagnostics.html").read_text()
    assert bvd.joint_rhat_from_html(html) == pytest.approx(1.041)


def test_joint_rhat_from_the_older_analysis_page() -> None:
    # Documenter builds before 17 September print the table in the
    # analysis page, with a row-number column and a row of column types.
    html = (FIXTURES / "site" / "analysis.html").read_text()
    assert bvd.joint_rhat_from_html(html) == pytest.approx(1.026)


def test_joint_rhat_needs_a_joint_row() -> None:
    html = (FIXTURES / "site" / "diagnostics.html").read_text()
    assert bvd.joint_rhat_from_html(html.replace(">joint<", ">other<")) is None
    assert bvd.joint_rhat_from_html("<p>No table.</p>") is None
    bad = html.replace(">1.041<", ">n/a<")
    assert bvd.joint_rhat_from_html(bad) is None


def test_joint_rhat_from_site_prefers_the_diagnostics_page() -> None:
    site = _site_zip(
        {
            "1/analysis.html": "analysis.html",
            "1/summary_assets/diagnostics.html": "diagnostics.html",
        }
    )
    rhat, source = bvd.joint_rhat_from_site(site)
    assert rhat == pytest.approx(1.041)
    assert source == "1/summary_assets/diagnostics.html"
    old = _site_zip({"1/analysis.html": "analysis.html"})
    assert bvd.joint_rhat_from_site(old) == (
        pytest.approx(1.026),
        "1/analysis.html",
    )
    # Hashed script chunks under assets/ are not pages.
    chunk = _site_zip({"dev/assets/analysis.html": "analysis.html"})
    rhat, reason = bvd.joint_rhat_from_site(chunk)
    assert rhat is None
    assert "no fit diagnostics table" in reason
    rhat, reason = bvd.joint_rhat_from_site(b"not a zip")
    assert rhat is None
    assert "not a zip" in reason


def test_fetch_diagnostics_caches_only_the_value(
    releases: pd.DataFrame, tmp_path: Path
) -> None:
    got: list[str] = []

    def download(url: str) -> bytes:
        got.append(url)
        return _site_zip(rhat="1.073")

    assets = releases.set_index("tag").loc["results-3151", "assets"]
    record = bvd.fetch_diagnostics("results-3151", assets, tmp_path, download)
    assert record["joint_max_rhat"] == pytest.approx(1.073)
    assert record["source"].endswith("summary_assets/diagnostics.html")
    assert record["reason"] is None
    assert [u.rsplit("/", 1)[1] for u in got] == [bvd.SITE_ASSET]
    # The zip is not kept, only the value.
    assert {p.name for p in (tmp_path / "results-3151").iterdir()} == {
        bvd.DIAGNOSTICS_FILE
    }
    again = bvd.fetch_diagnostics("results-3151", assets, tmp_path, download)
    assert again == record
    assert len(got) == 1


def test_fetch_diagnostics_without_a_site(tmp_path: Path) -> None:
    def offline(url: str) -> bytes:
        raise AssertionError(f"downloaded {url}")

    record = bvd.fetch_diagnostics("results-1", {}, tmp_path, offline)
    assert record["joint_max_rhat"] is None
    assert "site.zip" in record["reason"]


def test_select_release_skips_unconverged_and_unreadable_releases(
    releases: pd.DataFrame,
) -> None:
    cutoff = bvd.bvd_cutoff("2026-10-24") + dt.timedelta(days=30)
    asked: list[str] = []
    rhat = {**RHAT, "results-3169": 1.23, "results-3160": None}

    def lookup(tag: str) -> float | None:
        asked.append(tag)
        return rhat[tag]

    choice = bvd.select_release(
        releases, "2026-10-24", cutoff, AS_OF.get, lookup, max_rhat=1.1
    )
    assert choice is not None
    assert choice.tag == "results-3151"
    assert choice.joint_max_rhat == pytest.approx(1.03)
    assert choice.max_rhat == 1.1
    assert dict(choice.skipped) == {
        "results-3169": "joint R-hat 1.230 is above 1.1",
        "results-3160": "no readable joint R-hat",
    }
    # Diagnostics are only read for releases that are otherwise eligible.
    assert asked == ["results-3169", "results-3160", "results-3151"]


def test_select_release_records_releases_too_new_for_the_forecast(
    releases: pd.DataFrame,
) -> None:
    cutoff = bvd.bvd_cutoff("2026-10-24") + dt.timedelta(days=30)
    asked: list[str] = []

    def lookup(tag: str) -> float | None:
        asked.append(tag)
        return RHAT[tag]

    choice = bvd.select_release(
        releases, "2026-10-17", cutoff, AS_OF.get, lookup
    )
    assert choice is not None
    assert choice.tag == "results-3086"
    reasons = dict(choice.skipped)
    assert reasons["results-3169"] == "data end 2026-10-20, after 2026-10-17"
    assert "results-3086" not in reasons
    assert asked == ["results-3086"]


def test_select_release_skips_assets_updated_after_the_cut_off(
    releases: pd.DataFrame,
) -> None:
    # results-3160 was published in time but replaced after the cut-off.
    # results-3169 was published after it, so it is not considered at all.
    choice = bvd.select_release(
        releases,
        "2026-10-24",
        bvd.bvd_cutoff("2026-10-24"),
        AS_OF.get,
        RHAT.get,
    )
    assert choice is not None
    assert dict(choice.skipped) == {
        "results-3160": "an asset was updated after the cut-off"
    }


def test_release_choice_record_is_json() -> None:
    choice = bvd.ReleaseChoice(
        tag="results-3151",
        joint_max_rhat=1.032,
        max_rhat=1.1,
        skipped=(("results-3169", "joint R-hat 1.230 is above 1.1"),),
    )
    record = json.loads(json.dumps(choice.record()))
    assert record == {
        "tag": "results-3151",
        "joint_max_rhat": 1.032,
        "max_rhat": 1.1,
        "skipped": [
            {"tag": "results-3169", "reason": "joint R-hat 1.230 is above 1.1"}
        ],
    }


def test_bvd_for_vintage_gates_on_the_joint_rhat(
    releases: pd.DataFrame, tmp_path: Path
) -> None:
    def download(url: str) -> bytes:
        tag, name = url.rsplit("/", 2)[1:]
        if name == bvd.SITE_ASSET:
            return _site_zip(rhat="1.25" if tag == "results-3151" else None)
        return (FIXTURES / tag / name).read_bytes()

    have = releases.loc[releases["tag"].isin(["results-3151", "results-3086"])]
    rel = bvd.bvd_for_vintage("2026-10-24", tmp_path, have, download)
    assert rel is not None
    assert rel.tag == "results-3086"
    assert rel.choice is not None
    assert rel.choice.skipped == (
        ("results-3151", "joint R-hat 1.250 is above 1.1"),
    )
    # A looser limit takes the newer release, from the cached value.
    loose = bvd.bvd_for_vintage(
        "2026-10-24", tmp_path, have, download, max_rhat=1.3
    )
    assert loose is not None
    assert loose.tag == "results-3151"
    assert loose.choice is not None
    assert loose.choice.joint_max_rhat == pytest.approx(1.25)


def test_refresh_cache_records_the_limit_for_later_fits(
    releases: pd.DataFrame, tmp_path: Path
) -> None:
    def download(url: str) -> bytes:
        tag, name = url.rsplit("/", 2)[1:]
        if name == bvd.SITE_ASSET:
            return _site_zip(rhat="1.25" if tag == "results-3151" else None)
        return (FIXTURES / tag / name).read_bytes()

    have = releases.loc[releases["tag"].isin(["results-3151", "results-3086"])]
    chosen = bvd.refresh_cache(
        tmp_path, ["2026-10-24"], have, download, max_rhat=1.3
    )
    choice = chosen["2026-10-24"]
    assert choice is not None
    assert choice.tag == "results-3151"
    assert bvd.read_max_rhat(tmp_path) == 1.3
    assert bvd.read_max_rhat(tmp_path / "empty") == bvd.MAX_RHAT

    def offline(url: str) -> bytes:
        raise AssertionError(f"downloaded {url}")

    # A fit reads the cache with the recorded limit, offline.
    rel = bvd.bvd_for_vintage("2026-10-24", tmp_path, download=offline)
    assert rel is not None
    assert rel.tag == "results-3151"
