# Copyright (c) 2026 Sam Abbott. MIT licence, see LICENSE.
"""Reproduction number estimates published by BVDOutbreakSize.

The `renewalbvd` variant uses the latest posterior estimates of the
main [BVDOutbreakSize](https://github.com/epiforecasts/BVDOutbreakSize)
model. They come from its public `results-<run>` GitHub releases, one per
docs build.

A forecast may use a release only if it existed at the forecast's
cut-off. So `select_release` takes the newest `results-<run>` release
that was published, with every asset last updated, by the cut-off, and
whose data end (`as_of`) on or before the forecast date. Builds are
republished on old data with newer code, so `as_of` alone is not enough:
`results-3169`, published on 27 October with `as_of` 20 October, must not
feed the 24 October vintage.

The release must also come from a fit that converged, under a generous
check: the joint fit's worst R-hat at most `MAX_RHAT` (1.1, the config
value `[bvd] max_rhat`), with effective sample sizes ignored. The
releases carry no diagnostics asset. The rendered site (`site.zip`) does,
as the fit diagnostics table: `summary_assets/diagnostics.html` from
17 September, and the analysis page before that. A release whose table
cannot be read does not qualify, and the next older one is tried.
`site.zip` is large, so it is downloaded once, only for a release that
is otherwise eligible, and only the R-hat is kept.

Each release is cached by tag. `load_release` returns plain arrays: the
national `R_T` draws on the log scale and, where the release has
province estimates, each BVD patch with the hub provinces it covers.
Releases before 23 October have no `province_estimates.csv` asset, and
are national only here.

BVDOutbreakSize results are MIT licensed. Cite them as
<https://doi.org/10.5281/zenodo.20312758>.
"""

from __future__ import annotations

import datetime as dt
import io
import json
import re
import tomllib
import urllib.request
import zipfile
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from html.parser import HTMLParser
from pathlib import Path
from typing import Any, Final

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from bdbv_hub_renewal.data.dates import (
    HUB_TIMEZONE,
    reference_date_for_vintage,
    submission_window,
)

__all__ = [
    "ASSETS",
    "BVD_CITATION",
    "BVD_GI_MEAN",
    "BVD_GI_SD",
    "BVD_PROVINCES",
    "BVD_REPO",
    "DIAGNOSTICS_FILE",
    "HUB_TIMEZONE",
    "MAX_RHAT",
    "MODEL_ROW_COLUMNS",
    "SITE_ASSET",
    "BvdPatch",
    "BvdRelease",
    "ReleaseChoice",
    "bvd_cutoff",
    "bvd_for_vintage",
    "fetch_diagnostics",
    "fetch_release",
    "joint_rhat_from_html",
    "joint_rhat_from_site",
    "list_releases",
    "load_release",
    "load_releases",
    "model_rows",
    "parse_releases",
    "read_max_rhat",
    "refresh_cache",
    "release_as_of",
    "save_releases",
    "select_release",
]

BVD_REPO: Final = "epiforecasts/BVDOutbreakSize"
"""The GitHub repository of BVDOutbreakSize."""

BVD_CITATION: Final = (
    "BVDOutbreakSize results, MIT licence, "
    "https://doi.org/10.5281/zenodo.20312758"
)
"""How to cite the BVDOutbreakSize results."""

BVD_PROVINCES: Final[dict[str, str]] = {
    "ituri": "CD-IT",
    "nord_kivu": "CD-NK",
    "haut_uele": "CD-HU",
    "sud_kivu": "CD-SK",
    "tshopo": "CD-TO",
    "bas_uele": "CD-BU",
    "sud_ubangi": "CD-SU",
    "nord_ubangi": "CD-NU",
}
"""BVDOutbreakSize province keys and their hub codes."""

ASSETS: Final = (
    "posterior_draws.csv",
    "province_estimates.csv",
    "province_timeseries.csv",
    "observations.toml",
)
"""Release assets used: national draws, patch summaries, the national Rt
row the patch summaries are relative to, and the data cut-off
(`as_of_date`) for releases with no patch summaries."""

SITE_ASSET: Final = "site.zip"
"""The rendered site, read only for the joint fit's R-hat."""

TIMED_ASSETS: Final = (*ASSETS, SITE_ASSET)
"""Every asset a forecast reads, so each must predate its cut-off."""

DIAGNOSTICS_PAGES: Final = ("summary_assets/diagnostics.html", "analysis.html")
"""Pages of `site.zip` with the fit diagnostics table, in order of
preference: the diagnostics page from 17 September, the analysis page
before."""

DIAGNOSTICS_FILE: Final = "diagnostics.json"
"""The joint R-hat of a release, cached in place of `site.zip`."""

MAX_RHAT: Final = 1.1
"""Default limit on the joint fit's worst R-hat, BVDOutbreakSize's own
fail level. Its pass level (1.05, with ESS and divergence limits) is
stricter."""

GATE_FILE: Final = "gate.json"
"""The R-hat limit the cache was refreshed with, for later fits."""

BVD_GI_MEAN: Final = 15.3
"""Prior mean of BVDOutbreakSize's Gamma generation interval, in days."""

BVD_GI_SD: Final = 9.3
"""Prior SD of BVDOutbreakSize's Gamma generation interval, in days."""

Z90: Final = 1.6448536269514722
"""The standard normal 95th centile, for SDs from 90% intervals."""

RESULTS_TAG: Final = re.compile(r"^results-\d+$")
"""Main-build release tags. `v*` and `results-v*` are left out."""

Downloader = Callable[[str], bytes]


def _download(url: str) -> bytes:  # pragma: no cover - network
    req = urllib.request.Request(
        url, headers={"User-Agent": "bdbv-hub-renewal"}
    )
    with urllib.request.urlopen(req, timeout=120) as resp:
        data: bytes = resp.read()
    return data


def parse_releases(payload: Sequence[Mapping[str, Any]]) -> pd.DataFrame:
    """Tidy the GitHub release list of BVDOutbreakSize.

    Parameters
    ----------
    payload : list of mapping
        Releases as the GitHub REST API returns them.

    Returns
    -------
    pandas.DataFrame
        One row per published `results-<run>` release, newest publication
        first, with `tag`, `published_at` (UTC) and `assets`, a mapping
        from asset name to its download URL and `updated_at`.
    """
    rows = []
    for rel in payload:
        tag = str(rel["tag_name"])
        if rel.get("draft") or not RESULTS_TAG.match(tag):
            continue
        if not rel.get("published_at"):
            continue
        assets = {
            str(a["name"]): {
                "url": str(a["browser_download_url"]),
                "updated_at": pd.Timestamp(a["updated_at"]),
            }
            for a in rel.get("assets", [])
        }
        rows.append((tag, pd.Timestamp(rel["published_at"]), assets))
    out = pd.DataFrame(rows, columns=["tag", "published_at", "assets"])
    out["published_at"] = pd.to_datetime(out["published_at"], utc=True)
    return out.sort_values("published_at", ascending=False).reset_index(
        drop=True
    )


def list_releases(
    token: str | None = None,
) -> pd.DataFrame:  # pragma: no cover - network
    """List the `results-<run>` releases of BVDOutbreakSize.

    Parameters
    ----------
    token : str, optional
        A GitHub token, for the API rate limit.

    Returns
    -------
    pandas.DataFrame
        See `parse_releases`.
    """
    payload: list[Mapping[str, Any]] = []
    page = 1
    while True:
        url = (
            f"https://api.github.com/repos/{BVD_REPO}/releases"
            f"?per_page=100&page={page}"
        )
        headers = {"User-Agent": "bdbv-hub-renewal"}
        if token:
            headers["Authorization"] = f"Bearer {token}"
        req = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(req, timeout=60) as resp:
            batch = json.loads(resp.read())
        if not batch:
            break
        payload.extend(batch)
        page += 1
    return parse_releases(payload)


RELEASES_FILE: Final = "releases.json"
"""The release list in the cache, so a backtest needs no network."""


def save_releases(releases: pd.DataFrame, cache_dir: Path | str) -> Path:
    """Write a release list to the cache.

    Parameters
    ----------
    releases : pandas.DataFrame
        From `parse_releases`.
    cache_dir : pathlib.Path or str
        The cache.

    Returns
    -------
    pathlib.Path
        The file written.
    """
    rows = [
        {
            "tag": str(tag),
            "published_at": pd.Timestamp(pub).isoformat(),
            "assets": {
                name: {
                    "url": str(a["url"]),
                    "updated_at": pd.Timestamp(a["updated_at"]).isoformat(),
                }
                for name, a in assets.items()
                if name in TIMED_ASSETS
            },
        }
        for tag, pub, assets in zip(
            releases["tag"],
            releases["published_at"],
            releases["assets"],
            strict=True,
        )
    ]
    path = Path(cache_dir) / RELEASES_FILE
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(rows, indent=1) + "\n")
    return path


def load_releases(cache_dir: Path | str) -> pd.DataFrame | None:
    """Read the cached release list.

    Parameters
    ----------
    cache_dir : pathlib.Path or str
        The cache.

    Returns
    -------
    pandas.DataFrame or None
        As `parse_releases`, keeping only the assets in `TIMED_ASSETS`, or
        None if there is no cached list.
    """
    path = Path(cache_dir) / RELEASES_FILE
    if not path.exists():
        return None
    rows = json.loads(path.read_text())
    payload = [
        {
            "tag_name": r["tag"],
            "published_at": r["published_at"],
            "assets": [
                {
                    "name": name,
                    "browser_download_url": a["url"],
                    "updated_at": a["updated_at"],
                }
                for name, a in r["assets"].items()
            ],
        }
        for r in rows
    ]
    return parse_releases(payload)


def bvd_cutoff(vintage: dt.date | str) -> dt.datetime:
    """Return the time by which a release must exist to feed a vintage.

    The end of the Sunday that opens the vintage's submission window, in
    Kinshasa time, or the end of the vintage's own day if that is later.

    Parameters
    ----------
    vintage : datetime.date or str
        The data vintage.

    Returns
    -------
    datetime.datetime
        An aware datetime in `HUB_TIMEZONE`.
    """
    opens, _ = submission_window(reference_date_for_vintage(vintage))
    sunday = dt.datetime.combine(
        opens.date(), dt.time(23, 59, 59), HUB_TIMEZONE
    )
    own = dt.datetime.combine(
        pd.Timestamp(vintage).date(), dt.time(23, 59, 59), HUB_TIMEZONE
    )
    return max(sunday, own)


@dataclass(frozen=True)
class ReleaseChoice:
    """The release a forecast uses, and why newer ones were not.

    Attributes
    ----------
    tag : str
        The release tag.
    joint_max_rhat : float
        The joint fit's worst R-hat.
    max_rhat : float
        The limit it was held to.
    skipped : tuple of (str, str)
        Each newer release published by the cut-off, with the reason it
        was not used, newest first.
    """

    tag: str
    joint_max_rhat: float
    max_rhat: float
    skipped: tuple[tuple[str, str], ...] = ()

    def record(self) -> dict[str, Any]:
        """Return the choice as plain JSON values.

        Returns
        -------
        dict
            `tag`, `joint_max_rhat`, `max_rhat` and `skipped`, a list of
            `tag` and `reason` mappings.
        """
        return {
            "tag": self.tag,
            "joint_max_rhat": self.joint_max_rhat,
            "max_rhat": self.max_rhat,
            "skipped": [{"tag": t, "reason": r} for t, r in self.skipped],
        }


def select_release(
    releases: pd.DataFrame,
    forecast_date: dt.date | str,
    cutoff: dt.datetime,
    as_of: Callable[[str], dt.date | str | None],
    rhat: Callable[[str], float | None],
    max_rhat: float = MAX_RHAT,
) -> ReleaseChoice | None:
    """Choose the release a forecast may use.

    The newest release, by publication, that was published and whose
    used assets (`TIMED_ASSETS`) were last updated by `cutoff`, whose
    data end on or before `forecast_date`, and whose joint fit's worst
    R-hat is at most `max_rhat`. Releases published after `cutoff` did
    not exist for the forecast and are not recorded as skipped.

    Parameters
    ----------
    releases : pandas.DataFrame
        From `parse_releases`.
    forecast_date : datetime.date or str
        The forecast's data vintage.
    cutoff : datetime.datetime
        An aware datetime, usually `bvd_cutoff(forecast_date)`.
    as_of : callable
        Returns a release's data cut-off from its tag, or None if it is
        unknown. Only called for releases published in time.
    rhat : callable
        Returns a release's joint R-hat from its tag, or None if it
        cannot be read. Only called for releases that are otherwise
        eligible.
    max_rhat : float
        The R-hat limit.

    Returns
    -------
    ReleaseChoice or None
        The choice, or None if no release qualifies.
    """
    limit = pd.Timestamp(cutoff).tz_convert("UTC")
    day = pd.Timestamp(forecast_date).date()
    ordered = releases.sort_values("published_at", ascending=False)
    skipped: list[tuple[str, str]] = []
    for tag, published, assets in zip(
        ordered["tag"], ordered["published_at"], ordered["assets"], strict=True
    ):
        tag = str(tag)
        if pd.Timestamp(published) > limit:
            continue
        updated = [
            pd.Timestamp(a["updated_at"])
            for name, a in assets.items()
            if name in TIMED_ASSETS
        ]
        if any(t > limit for t in updated):
            skipped.append((tag, "an asset was updated after the cut-off"))
            continue
        data_end = as_of(tag)
        if data_end is None:
            skipped.append((tag, "no data cut-off"))
            continue
        end = pd.Timestamp(data_end).date()
        if end > day:
            skipped.append((tag, f"data end {end}, after {day}"))
            continue
        value = rhat(tag)
        if value is None:
            skipped.append((tag, "no readable joint R-hat"))
            continue
        if value > max_rhat:
            skipped.append(
                (tag, f"joint R-hat {value:.3f} is above {max_rhat:g}")
            )
            continue
        return ReleaseChoice(tag, float(value), float(max_rhat), tuple(skipped))
    return None


class _Tables(HTMLParser):
    """Collect the cell text of every table row in an HTML page."""

    def __init__(self) -> None:
        super().__init__()
        self.tables: list[list[list[str]]] = []
        self._row: list[str] | None = None
        self._cell: list[str] | None = None

    def handle_starttag(
        self, tag: str, attrs: list[tuple[str, str | None]]
    ) -> None:
        del attrs
        if tag == "table":
            self.tables.append([])
        elif tag == "tr" and self.tables:
            self._row = []
        elif tag in ("td", "th") and self._row is not None:
            self._cell = []

    def handle_endtag(self, tag: str) -> None:
        if tag in ("td", "th") and self._cell is not None:
            assert self._row is not None
            self._row.append(" ".join("".join(self._cell).split()))
            self._cell = None
        elif tag == "tr" and self._row is not None:
            self.tables[-1].append(self._row)
            self._row = None

    def handle_data(self, data: str) -> None:
        if self._cell is not None:
            self._cell.append(data)


def joint_rhat_from_html(html: str) -> float | None:
    """Read the joint fit's worst R-hat from a rendered diagnostics table.

    The first table with `fit` and `max_rhat` header cells and a `joint`
    row. Columns are found by name, so the row-number and column-type
    rows of older pages are ignored.

    Parameters
    ----------
    html : str
        A rendered page.

    Returns
    -------
    float or None
        The R-hat, or None if no such table or value is found.
    """
    parser = _Tables()
    parser.feed(html)
    for table in parser.tables:
        for i, header in enumerate(table):
            if "fit" not in header or "max_rhat" not in header:
                continue
            fit, col = header.index("fit"), header.index("max_rhat")
            for row in table[i + 1 :]:
                if len(row) > max(fit, col) and row[fit] == "joint":
                    try:
                        return float(row[col])
                    except ValueError:
                        return None
    return None


def joint_rhat_from_site(site: bytes) -> tuple[float | None, str]:
    """Read the joint fit's worst R-hat from a release's `site.zip`.

    Pages are tried in the order of `DIAGNOSTICS_PAGES`. Bundled script
    chunks under `assets/` are not pages and are skipped.

    Parameters
    ----------
    site : bytes
        The zip archive.

    Returns
    -------
    tuple
        The R-hat and the page it came from, or None and the reason it
        could not be read.
    """
    try:
        archive = zipfile.ZipFile(io.BytesIO(site))
    except zipfile.BadZipFile:
        return None, "site.zip is not a zip archive"
    with archive:
        names = [n for n in archive.namelist() if "/assets/" not in n]
        for page in DIAGNOSTICS_PAGES:
            for name in sorted(n for n in names if n.endswith(page)):
                html = archive.read(name).decode("utf-8", "replace")
                value = joint_rhat_from_html(html)
                if value is not None:
                    return value, name
    return None, "site.zip has no fit diagnostics table with a joint row"


def fetch_diagnostics(
    tag: str,
    assets: Mapping[str, Mapping[str, Any]],
    cache_dir: Path | str,
    download: Downloader = _download,
) -> dict[str, Any]:
    """Read a release's joint R-hat once, into `cache_dir/<tag>/`.

    `site.zip` is downloaded, read and dropped. Only the value is kept,
    in `DIAGNOSTICS_FILE`, so a release is never downloaded twice.

    Parameters
    ----------
    tag : str
        The release tag.
    assets : mapping
        Asset name to its `url`, as in `parse_releases`.
    cache_dir : pathlib.Path or str
        The cache.
    download : callable
        Returns the bytes at a URL.

    Returns
    -------
    dict
        `joint_max_rhat` (float or None), `source` (the page read, or
        None) and `reason` (why there is no value, or None).
    """
    path = Path(cache_dir) / tag / DIAGNOSTICS_FILE
    if path.exists():
        cached: dict[str, Any] = json.loads(path.read_text())
        return cached
    if SITE_ASSET in assets:
        value, where = joint_rhat_from_site(
            download(str(assets[SITE_ASSET]["url"]))
        )
    else:
        value, where = None, "the release has no site.zip"
    record = {
        "joint_max_rhat": value,
        "source": where if value is not None else None,
        "reason": None if value is not None else where,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record, indent=1) + "\n")
    return record


def read_max_rhat(cache_dir: Path | str) -> float:
    """Return the R-hat limit the cache was refreshed with.

    Parameters
    ----------
    cache_dir : pathlib.Path or str
        The cache.

    Returns
    -------
    float
        The limit in `GATE_FILE`, or `MAX_RHAT` if there is none.
    """
    path = Path(cache_dir) / GATE_FILE
    if not path.exists():
        return MAX_RHAT
    return float(json.loads(path.read_text())["max_rhat"])


def fetch_release(
    tag: str,
    assets: Mapping[str, Mapping[str, Any]],
    cache_dir: Path | str,
    download: Downloader = _download,
) -> Path:
    """Download a release's assets once, into `cache_dir/<tag>/`.

    Parameters
    ----------
    tag : str
        The release tag.
    assets : mapping
        Asset name to its `url`, as in `parse_releases`. Only `ASSETS`
        are fetched, and only those the release has.
    cache_dir : pathlib.Path or str
        The cache. A file already there is not downloaded again. Of
        `observations.toml` only `as_of_date` is kept.
    download : callable
        Returns the bytes at a URL.

    Returns
    -------
    pathlib.Path
        The release's directory.
    """
    out = Path(cache_dir) / tag
    out.mkdir(parents=True, exist_ok=True)
    for name in ASSETS:
        path = out / name
        if name in assets and not path.exists():
            content = download(str(assets[name]["url"]))
            if name == "observations.toml":
                # The input data are large; only the cut-off is used.
                found = _find_key(tomllib.loads(content.decode()), "as_of_date")
                content = f'as_of_date = "{found}"\n'.encode()
            path.write_bytes(content)
    return out


@dataclass(frozen=True)
class BvdPatch:
    """BVDOutbreakSize's reproduction number for one patch at `as_of`.

    Attributes
    ----------
    members : tuple of str
        The hub provinces the patch covers.
    log_median : float
        Log of the posterior median of `R_T`.
    log_sd : float
        Log-scale SD, from the 90% interval.
    pooled : bool
        Whether several provinces share the patch.
    """

    members: tuple[str, ...]
    log_median: float
    log_sd: float
    pooled: bool = False


@dataclass(frozen=True)
class BvdRelease:
    """The estimates of one BVDOutbreakSize release.

    Attributes
    ----------
    tag : str
        The release tag.
    as_of : datetime.date
        BVDOutbreakSize's data cut-off.
    national_log_rt : numpy.ndarray
        Posterior draws of the national `log R_T`.
    patches : dict
        Patch name to `BvdPatch`. Empty for a national-only release.
    reference : tuple of float or None
        Log median and log-scale SD of the national Rt at `as_of` in the
        same table as the patches (`province_timeseries.csv`), so patch
        deviations use one source. None if the release has no such row.
    choice : ReleaseChoice or None
        Why this release was chosen, when it was chosen for a vintage.
    """

    tag: str
    as_of: dt.date
    national_log_rt: NDArray[np.float64]
    patches: dict[str, BvdPatch] = field(default_factory=dict)
    reference: tuple[float, float] | None = None
    choice: ReleaseChoice | None = None


def release_as_of(path: Path | str) -> dt.date | None:
    """Read a cached release's data cut-off.

    From `province_estimates.csv` where present, otherwise from the
    `as_of_date` in `observations.toml`.

    Parameters
    ----------
    path : pathlib.Path or str
        The release's directory.

    Returns
    -------
    datetime.date or None
        The cut-off, or None if neither file gives it.
    """
    path = Path(path)
    est = path / "province_estimates.csv"
    if est.exists():
        return pd.Timestamp(pd.read_csv(est)["as_of"].iloc[0]).date()
    obs = path / "observations.toml"
    if obs.exists():
        found = _find_key(tomllib.loads(obs.read_text()), "as_of_date")
        if found is not None:
            return pd.Timestamp(found).date()
    return None


def _find_key(table: Mapping[str, Any], key: str) -> Any:
    if key in table:
        return table[key]
    for value in table.values():
        if isinstance(value, Mapping):
            found = _find_key(value, key)
            if found is not None:
                return found
    return None


def _patches(estimates: pd.DataFrame) -> dict[str, BvdPatch]:
    unknown = sorted(set(estimates["province"]) - set(BVD_PROVINCES))
    if unknown:
        raise ValueError(f"Unknown BVDOutbreakSize province keys {unknown}.")
    out = {}
    for patch, rows in estimates.groupby("patch", sort=False):
        row = rows.iloc[0]
        width = np.log(float(row["R_T_upper"]) / float(row["R_T_lower"]))
        out[str(patch)] = BvdPatch(
            members=tuple(BVD_PROVINCES[k] for k in rows["province"]),
            log_median=float(np.log(float(row["R_T_median"]))),
            log_sd=float(width / (2 * Z90)),
            pooled=bool(rows["pooled"].astype(bool).any()),
        )
    return out


def load_release(path: Path | str) -> BvdRelease:
    """Read a cached release into plain arrays.

    Parameters
    ----------
    path : pathlib.Path or str
        The release's directory, named by its tag.

    Returns
    -------
    BvdRelease
        The national draws and any patch estimates.

    Raises
    ------
    ValueError
        If the cut-off cannot be read, or a patch is an unknown province.
    """
    path = Path(path)
    as_of = release_as_of(path)
    if as_of is None:
        raise ValueError(f"No data cut-off in BVDOutbreakSize release {path}.")
    draws = pd.read_csv(path / "posterior_draws.csv")
    if "R_T" in draws:
        log_rt = np.log(draws["R_T"].to_numpy(dtype=float))
    else:
        # Early builds publish the growth rate only. With a Gamma
        # generation interval, R = (1 + r theta)^k (Wallinga and Lipsitch).
        k = (BVD_GI_MEAN / BVD_GI_SD) ** 2
        theta = BVD_GI_SD**2 / BVD_GI_MEAN
        log_rt = k * np.log1p(draws["r"].to_numpy(dtype=float) * theta)
    est = path / "province_estimates.csv"
    patches = _patches(pd.read_csv(est)) if est.exists() else {}
    return BvdRelease(
        tag=path.name,
        as_of=as_of,
        national_log_rt=log_rt,
        patches=patches,
        reference=_reference(path / "province_timeseries.csv", as_of),
    )


def _reference(path: Path, as_of: dt.date) -> tuple[float, float] | None:
    if not path.exists():
        return None
    ts = pd.read_csv(path)
    row = ts.loc[
        (ts["area"] == "National")
        & (ts["series"] == "rt")
        & (pd.to_datetime(ts["date"]).dt.date == as_of)
    ]
    if row.empty:
        return None
    r = row.iloc[0]
    width = np.log(float(r["upper_90"]) / float(r["lower_90"]))
    return float(np.log(float(r["median"]))), float(width / (2 * Z90))


MODEL_ROW_COLUMNS: Final = (
    "group",
    "members",
    "as_of",
    "log_mean",
    "log_sd",
    "ref_log_mean",
    "ref_log_sd",
)
"""Columns of `model_rows`, as `bdbv_hub_renewal.model.inputs.bvd_rows`."""


def model_rows(release: BvdRelease) -> pd.DataFrame:
    """Turn a release into the rows the model's BVD term reads.

    The national row is the mean and SD of the national log `R_T` draws.
    Each patch row is its log median and log-scale SD, with the national
    row of the same table as its reference, so the deviation uses one
    source. Without that table, the national draws are the reference.

    Parameters
    ----------
    release : BvdRelease
        From `load_release`.

    Returns
    -------
    pandas.DataFrame
        Columns `MODEL_ROW_COLUMNS`, the national row first.
    """
    as_of = release.as_of.isoformat()
    log_rt = release.national_log_rt
    nat = (float(log_rt.mean()), float(log_rt.std(ddof=1)))
    ref = release.reference or nat
    rows = [("national", "", as_of, *nat, *nat)]
    for name, patch in release.patches.items():
        rows.append(
            (
                name,
                ";".join(patch.members),
                as_of,
                patch.log_median,
                patch.log_sd,
                *ref,
            )
        )
    return pd.DataFrame(rows, columns=list(MODEL_ROW_COLUMNS))


def bvd_for_vintage(
    vintage: dt.date | str,
    cache_dir: Path | str,
    releases: pd.DataFrame | None = None,
    download: Downloader = _download,
    max_rhat: float | None = None,
) -> BvdRelease | None:
    """Return the BVDOutbreakSize estimates a vintage's forecast may use.

    See `select_release`. Release files and R-hat values already in the
    cache are not downloaded again.

    Parameters
    ----------
    vintage : datetime.date or str
        The data vintage, which is also the forecast date.
    cache_dir : pathlib.Path or str
        The release cache.
    releases : pandas.DataFrame, optional
        From `parse_releases`. Read from the cache (`load_releases`) if
        not given, or listed from GitHub if the cache has none.
    download : callable
        Returns the bytes at a URL.
    max_rhat : float, optional
        The R-hat limit. Defaults to the one the cache was refreshed with
        (`read_max_rhat`).

    Returns
    -------
    BvdRelease or None
        The estimates, with the choice, or None if no release qualifies.
    """
    if releases is None:
        releases = load_releases(cache_dir)
    if releases is None:  # pragma: no cover - network
        releases = list_releases()
    assets = dict(zip(releases["tag"], releases["assets"], strict=True))

    def as_of(tag: str) -> dt.date | None:
        return release_as_of(
            fetch_release(tag, assets[tag], cache_dir, download)
        )

    def rhat(tag: str) -> float | None:
        value = fetch_diagnostics(tag, assets[tag], cache_dir, download)
        found = value["joint_max_rhat"]
        return None if found is None else float(found)

    limit = read_max_rhat(cache_dir) if max_rhat is None else max_rhat
    choice = select_release(
        releases, vintage, bvd_cutoff(vintage), as_of, rhat, limit
    )
    if choice is None:
        return None
    release = load_release(Path(cache_dir) / choice.tag)
    return replace(release, choice=choice)


def refresh_cache(
    cache_dir: Path | str,
    vintages: Sequence[dt.date | str],
    releases: pd.DataFrame | None = None,
    download: Downloader = _download,
    max_rhat: float = MAX_RHAT,
) -> dict[str, ReleaseChoice | None]:
    """Cache the release list and the release each vintage may use.

    The R-hat limit is recorded in the cache (`GATE_FILE`), so a later fit
    that reads the cache applies the same one.

    Parameters
    ----------
    cache_dir : pathlib.Path or str
        The cache, usually `data/bvd/`.
    vintages : sequence of datetime.date or str
        The data vintages.
    releases : pandas.DataFrame, optional
        From `parse_releases`. Listed from GitHub if not given.
    download : callable
        Returns the bytes at a URL.
    max_rhat : float
        The R-hat limit.

    Returns
    -------
    dict
        The choice for each vintage, or None.
    """
    if releases is None:  # pragma: no cover - network
        releases = list_releases()
    save_releases(releases, cache_dir)
    gate = Path(cache_dir) / GATE_FILE
    gate.write_text(json.dumps({"max_rhat": max_rhat}) + "\n")
    chosen: dict[str, ReleaseChoice | None] = {}
    for vintage in vintages:
        rel = bvd_for_vintage(vintage, cache_dir, releases, download, max_rhat)
        chosen[pd.Timestamp(vintage).date().isoformat()] = (
            None if rel is None else rel.choice
        )
    return chosen
