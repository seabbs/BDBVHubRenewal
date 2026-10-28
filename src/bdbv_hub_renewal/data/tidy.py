# Copyright (c) 2026 Sam Abbott. MIT licence, see LICENSE.
"""Download, tidy and load the BDBV Modeling Hub data.

The hub publishes these streams (see `DATA_STREAMS`):

- `target-data/time-series.csv`: weekly confirmed cases by onset week,
  for the country, provinces and health zones, one block per `as_of`
  vintage. This is the target, observed plus imputed onsets.
- `auxiliary-data/daily-confirmed-case.csv`: daily confirmed cases by
  onset day, split into observed and imputed onsets, per vintage.
- `auxiliary-data/weekly-reporting-delays-by-province.csv`: one row per
  case with a known onset, giving its province, onset week and
  onset-to-report delay. Only the latest vintage is in the file.
  Earlier vintages are in `archived-delay-data/` or only in the git
  history, so `fetch_hub_data` walks the history to find them all.
- `auxiliary-data/locations.csv` and `health-zone-code-mapping.csv`:
  location codes, names and parents.
- `hub-config/`: the round configuration, used for validation.

The hub publishes no deaths, suspected cases, report-date series or
populations.
Province populations come from a cited public source instead
(`PROVINCE_POPULATIONS`).

The tidy tables written by `build_tidy_data` are small and committed
under `data/`.
Raw downloads go to `data/raw/` and are not committed.
"""

from __future__ import annotations

import datetime as dt
import io
import json
import os
import re
import urllib.request
from collections.abc import Mapping
from importlib.resources import files
from pathlib import Path
from typing import Any, Final

import numpy as np
import pandas as pd

from bdbv_hub_renewal.data.dates import epiweek_end

__all__ = [
    "COHERENCE_HANDLING",
    "DATA_STREAMS",
    "HUB_REPO",
    "POPULATION_SOURCE",
    "PROVINCE_CENTRES",
    "PROVINCE_POPULATIONS",
    "TARGET",
    "build_tidy_data",
    "coherence_summary",
    "delays_for_vintage",
    "fetch_hub_data",
    "imputed_share",
    "load_daily",
    "load_daily_zones",
    "load_delays",
    "load_health_zones",
    "load_locations",
    "load_province_centres",
    "load_target",
    "national_coherence",
    "newly_missing",
    "revision_table",
    "tidy_daily",
    "tidy_delays",
    "tidy_health_zones",
    "tidy_locations",
    "tidy_target",
    "vintages",
]

HUB_REPO: Final = "InsightNet-US/BDBV-Modeling-Hub"
"""The hub's GitHub repository."""

TARGET: Final = "wk inc confirmed cases"
"""The hub target this repository forecasts."""

DATA_STREAMS: Final[dict[str, str]] = {
    "target-data/time-series.csv": (
        "Weekly confirmed cases by onset week and vintage (observed plus "
        "imputed onsets), for the country, provinces and health zones."
    ),
    "auxiliary-data/daily-confirmed-case.csv": (
        "Daily confirmed cases by onset day and vintage, split into "
        "observed and imputed onsets."
    ),
    "auxiliary-data/weekly-reporting-delays-by-province.csv": (
        "One row per known-onset case: province, onset week and "
        "onset-to-report delay in days. Latest vintage only; earlier "
        "vintages are in the archive folder or the git history."
    ),
    "auxiliary-data/locations.csv": (
        "Location codes, names, types and parents."
    ),
    "auxiliary-data/health-zone-code-mapping.csv": (
        "Health zone codes mapped to P-codes and DHIS2 codes."
    ),
    "hub-config/tasks.json": "Round, task and output type configuration.",
    "hub-config/admin.json": "Hub settings, including the time zone.",
    "hub-config/model-metadata-schema.json": "Model metadata schema.",
}
"""Files fetched from the hub, with what each holds."""

_LATEST_FILES: Final = tuple(
    path for path in DATA_STREAMS if "weekly-reporting-delays" not in path
)

PROVINCE_POPULATIONS: Final[dict[str, int]] = {
    "CD-KN": 13_916_000,
    "CD-BC": 6_365_000,
    "CD-KG": 2_416_000,
    "CD-KL": 6_169_000,
    "CD-MN": 2_082_000,
    "CD-KS": 3_165_000,
    "CD-KC": 3_743_000,
    "CD-KE": 3_601_000,
    "CD-LO": 2_801_000,
    "CD-SA": 2_417_000,
    "CD-MA": 2_654_000,
    "CD-SK": 6_565_000,
    "CD-NK": 7_574_000,
    "CD-IT": 4_008_000,
    "CD-HU": 2_046_000,
    "CD-TO": 2_582_000,
    "CD-BU": 1_250_000,
    "CD-NU": 1_425_000,
    "CD-MO": 1_950_000,
    "CD-SU": 2_755_000,
    "CD-EQ": 1_712_000,
    "CD-TU": 1_789_000,
    "CD-TA": 3_570_000,
    "CD-HL": 3_444_000,
    "CD-LU": 2_993_000,
    "CD-HK": 5_378_000,
}
"""Resident population of each province in 2019.

From the Institut National de la Statistique, *Annuaire statistique RDC
2020* (March 2021), as tabulated on Wikipedia's "Provinces of the
Democratic Republic of the Congo" page (accessed 27 October 2026).
These are the figures epiforecasts/BVDOutbreakSize uses.
"""

POPULATION_SOURCE: Final = (
    "Institut National de la Statistique, Annuaire statistique RDC 2020 "
    "(March 2021), 2019 figures, via "
    "https://en.wikipedia.org/wiki/Provinces_of_the_Democratic_Republic_of_the_Congo"
    " (accessed 2026-10-27)."
)
"""Citation for `PROVINCE_POPULATIONS`."""


def load_province_centres() -> pd.DataFrame:
    """Load the population-weighted centres of all 26 provinces.

    The mean of health zone centroids weighted by WorldPop 2025 counts,
    with zone boundaries from the Ministry of Health shapefile on the
    Humanitarian Data Exchange, via INRB-UMIE/BDBV2026-Data
    (`build/drc_health_zones.geojson` at `f7d3907`, CC BY 4.0).
    This is the source and method of epiforecasts/BVDOutbreakSize. The
    file ships with the package, so it does not depend on the data
    directory.

    Returns
    -------
    pandas.DataFrame
        Columns `location`, `latitude`, `longitude`, `worldpop` and
        `zones`.
    """
    path = files("bdbv_hub_renewal.data").joinpath("province_centres.csv")
    with path.open() as fh:
        return pd.read_csv(fh, comment="#")


_CENTRES = load_province_centres()
PROVINCE_CENTRES: Final[dict[str, tuple[float, float]]] = {
    str(loc): (float(lat), float(lon))
    for loc, lat, lon in zip(
        _CENTRES["location"],
        _CENTRES["latitude"].to_numpy(dtype=float),
        _CENTRES["longitude"].to_numpy(dtype=float),
        strict=True,
    )
}
"""Population-weighted centres (latitude, longitude) of all provinces.

See `load_province_centres`.
"""


# --------------------------------------------------------------------------
# Download
# --------------------------------------------------------------------------


def _request(url: str, token: str | None) -> bytes:
    headers = {"User-Agent": "bdbv-hub-renewal"}
    if token and url.startswith("https://api.github.com"):
        headers["Authorization"] = f"Bearer {token}"
    req = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(req, timeout=60) as resp:
        data: bytes = resp.read()
    return data


def _api(path: str, token: str | None) -> Any:
    return json.loads(_request(f"https://api.github.com/{path}", token))


def _raw(sha: str, path: str, token: str | None) -> bytes:
    url = f"https://raw.githubusercontent.com/{HUB_REPO}/{sha}/{path}"
    return _request(url, token)


def _delay_as_of(content: bytes, path: str) -> str:
    """Find the vintage of a delay file from its `as_of` column or name."""
    df = pd.read_csv(io.BytesIO(content), nrows=5)
    if "as_of" in df.columns:
        return str(df["as_of"].iloc[0])
    match = re.search(r"(\d{4}-\d{2}-\d{2})", path)
    if match is None:
        raise ValueError(f"Cannot find the vintage of delay file {path}.")
    return match.group(1)


def fetch_hub_data(
    raw_dir: Path | str,
    ref: str = "main",
    token: str | None = None,
) -> dict[str, Any]:
    """Download the hub's data, including every vintage of the delay data.

    The latest versions of the files in `DATA_STREAMS` are taken from
    `ref`.
    For the delay data, every commit that touched `auxiliary-data/` is
    searched for delay files, and one file is kept per `as_of` vintage
    (from the most recent commit that has it).

    Parameters
    ----------
    raw_dir : pathlib.Path or str
        Where to save the files. Delay files go to `raw_dir/delays/`.
    ref : str
        Git ref of the hub to take the latest files from.
    token : str, optional
        GitHub token for the API. Defaults to `GITHUB_TOKEN` if set.
        Unauthenticated requests are limited to 60 an hour, which is
        enough for one run.

    Returns
    -------
    dict
        A manifest with the commit SHA used and the source of each delay
        vintage. It is also written to `raw_dir/manifest.json`.
    """
    token = token or os.environ.get("GITHUB_TOKEN")
    raw = Path(raw_dir)
    (raw / "delays").mkdir(parents=True, exist_ok=True)
    head = _api(f"repos/{HUB_REPO}/commits/{ref}", token)["sha"]
    for path in _LATEST_FILES:
        dest = raw / path
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(_raw(head, path, token))

    commits = _api(
        f"repos/{HUB_REPO}/commits?path=auxiliary-data&sha={ref}&per_page=100",
        token,
    )
    delay_sources: dict[str, dict[str, str]] = {}
    for commit in commits:  # newest first
        sha = commit["sha"]
        tree = _api(f"repos/{HUB_REPO}/git/trees/{sha}?recursive=1", token)
        paths = [
            item["path"]
            for item in tree["tree"]
            if item["path"].startswith("auxiliary-data/")
            and "reporting-delays" in item["path"]
            and item["path"].endswith(".csv")
        ]
        for path in paths:
            content = _raw(sha, path, token)
            as_of = _delay_as_of(content, path)
            if as_of in delay_sources:
                continue
            (raw / "delays" / f"{as_of}.csv").write_bytes(content)
            delay_sources[as_of] = {"commit": sha, "path": path}
    manifest = {
        "repo": HUB_REPO,
        "ref": ref,
        "commit": head,
        "downloaded": dt.datetime.now(dt.UTC).isoformat(timespec="seconds"),
        "delay_vintages": dict(sorted(delay_sources.items())),
    }
    (raw / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


# --------------------------------------------------------------------------
# Tidy
# --------------------------------------------------------------------------


def tidy_locations(locations_csv: Path | str) -> pd.DataFrame:
    """Tidy the country and province rows of `locations.csv`.

    Adds the 2019 population (`PROVINCE_POPULATIONS`, summed for the
    country) and the province centre where known.

    Parameters
    ----------
    locations_csv : pathlib.Path or str
        Path to the hub's `auxiliary-data/locations.csv`.

    Returns
    -------
    pandas.DataFrame
        Columns `location`, `location_name`, `location_type`,
        `population`, `latitude` and `longitude`.
    """
    loc = pd.read_csv(locations_csv)
    loc = loc.loc[loc["location_type"].isin(["country", "province"])].copy()
    pop = pd.Series(PROVINCE_POPULATIONS, dtype="int64")
    loc["population"] = loc["location"].map(pop)
    loc.loc[loc["location"] == "CD", "population"] = int(pop.sum())
    loc["population"] = loc["population"].astype("int64")
    loc["latitude"] = loc["location"].map(
        {k: v[0] for k, v in PROVINCE_CENTRES.items()}
    )
    loc["longitude"] = loc["location"].map(
        {k: v[1] for k, v in PROVINCE_CENTRES.items()}
    )
    cols = [
        "location",
        "location_name",
        "location_type",
        "population",
        "latitude",
        "longitude",
    ]
    return loc[cols].reset_index(drop=True)


def _weekly_split(daily: pd.DataFrame) -> pd.DataFrame:
    """Sum daily observed and imputed onsets into epi weeks."""
    d = daily.copy()
    d["target_end_date"] = [epiweek_end(x).isoformat() for x in d["onset_date"]]
    return d.groupby(
        ["vintage", "location", "target_end_date"], as_index=False
    )[["observed", "imputed"]].sum()


def tidy_daily(daily_csv: Path | str, level: str = "province") -> pd.DataFrame:
    """Tidy daily onsets by vintage, for one level of geography.

    Parameters
    ----------
    daily_csv : pathlib.Path or str
        Path to the hub's `auxiliary-data/daily-confirmed-case.csv`.
    level : {"province", "zone"}
        The country and provinces, or the health zones, which are kept
        apart for a later zone stage.

    Returns
    -------
    pandas.DataFrame
        Columns `vintage`, `location`, `onset_date`, `observed` and
        `imputed`. Only days with at least one case are present.
    """
    types = {"province": ["country", "province"], "zone": ["health zone"]}
    if level not in types:
        raise ValueError(f"Unknown level {level!r}: use {sorted(types)}.")
    raw = pd.read_csv(daily_csv)
    raw = raw.loc[raw["location_type"].isin(types[level])]
    raw = raw.assign(
        kind=np.where(raw["imputed"].astype(bool), "imputed", "observed")
    )
    wide = raw.pivot_table(
        index=["as_of", "location", "target_end_date"],
        columns="kind",
        values="observation",
        aggfunc="sum",
        fill_value=0,
    ).reset_index()
    for col in ("observed", "imputed"):
        if col not in wide.columns:
            wide[col] = 0
    wide = wide.rename(
        columns={"as_of": "vintage", "target_end_date": "onset_date"}
    )
    wide.columns.name = None
    out = wide[["vintage", "location", "onset_date", "observed", "imputed"]]
    out = out.astype({"observed": "int64", "imputed": "int64"})
    return out.sort_values(["vintage", "location", "onset_date"]).reset_index(
        drop=True
    )


def tidy_target(
    time_series_csv: Path | str, daily: pd.DataFrame
) -> pd.DataFrame:
    """Tidy the target data for the country and provinces.

    Totals come from `time-series.csv`.
    The split into observed and imputed onsets comes from the daily file,
    summed into epi weeks.

    Parameters
    ----------
    time_series_csv : pathlib.Path or str
        Path to the hub's `target-data/time-series.csv`.
    daily : pandas.DataFrame
        Output of `tidy_daily`.

    Returns
    -------
    pandas.DataFrame
        Columns `kind` (always `"real"`), `vintage`, `location`,
        `location_name`, `target_end_date`, `observed`, `imputed` and
        `total`.

    Raises
    ------
    ValueError
        If observed plus imputed does not equal the total in any row.
    """
    ts = pd.read_csv(time_series_csv)
    ts = ts.loc[
        (ts["target"] == TARGET)
        & ts["location_type"].isin(["country", "province"])
    ]
    ts = ts.rename(columns={"as_of": "vintage", "observation": "total"})
    split = _weekly_split(daily)
    out = ts.merge(
        split, on=["vintage", "location", "target_end_date"], how="left"
    )
    out[["observed", "imputed"]] = (
        out[["observed", "imputed"]].fillna(0).astype("int64")
    )
    out["total"] = out["total"].astype("int64")
    bad = out["observed"] + out["imputed"] != out["total"]
    if bad.any():
        raise ValueError(
            f"{int(bad.sum())} rows where observed plus imputed is not "
            "the total."
        )
    out.insert(0, "kind", "real")
    cols = [
        "kind",
        "vintage",
        "location",
        "location_name",
        "target_end_date",
        "observed",
        "imputed",
        "total",
    ]
    return (
        out[cols]
        .sort_values(["vintage", "location", "target_end_date"])
        .reset_index(drop=True)
    )


def tidy_health_zones(time_series_csv: Path | str) -> pd.DataFrame:
    """Tidy the health zone rows of the target data.

    Health zones are not modelled yet.
    They are kept so the data layer covers every stream the hub
    publishes.

    Parameters
    ----------
    time_series_csv : pathlib.Path or str
        Path to the hub's `target-data/time-series.csv`.

    Returns
    -------
    pandas.DataFrame
        Columns `vintage`, `location`, `location_name`, `province`,
        `target_end_date` and `total`. Zero rows are dropped.
    """
    ts = pd.read_csv(time_series_csv)
    hz = ts.loc[
        (ts["location_type"] == "health zone") & (ts["observation"] > 0)
    ].copy()
    hz["province"] = hz["location"].str.slice(0, 5)
    hz = hz.rename(columns={"as_of": "vintage", "observation": "total"})
    cols = [
        "vintage",
        "location",
        "location_name",
        "province",
        "target_end_date",
        "total",
    ]
    return (
        hz[cols]
        .sort_values(["vintage", "location", "target_end_date"])
        .reset_index(drop=True)
    )


def _province_code_map(locations: pd.DataFrame) -> dict[str, str]:
    """Map names and short codes of provinces to hub location codes."""
    prov = locations.loc[locations["location_type"] == "province"]
    mapping: dict[str, str] = {}
    for code, name in zip(prov["location"], prov["location_name"], strict=True):
        mapping[name] = code
        mapping[name.replace(" ", "-")] = code
        mapping[code.removeprefix("CD-")] = code
        mapping[code] = code
    return mapping


def tidy_delays(
    delay_files: Mapping[str, Path | str] | Path | str,
    locations: pd.DataFrame,
) -> pd.DataFrame:
    """Tidy every vintage of the delay line data into one long table.

    Older files use a `location` column of short codes and newer files a
    `location_name` column, so both are mapped to hub location codes.

    Parameters
    ----------
    delay_files : mapping of str to path, or a directory
        Vintage to file, or a directory of `<vintage>.csv` files as
        written by `fetch_hub_data`.
    locations : pandas.DataFrame
        Output of `tidy_locations`.

    Returns
    -------
    pandas.DataFrame
        One row per known-onset case, with columns `vintage`,
        `location`, `location_name`, `onset_week` (the Saturday ending
        the onset week) and `delay` (days from onset to report).

    Raises
    ------
    ValueError
        If a province name cannot be mapped to a location code.
    """
    if isinstance(delay_files, Path | str):
        files = {p.stem: p for p in sorted(Path(delay_files).glob("*.csv"))}
    else:
        files = {k: Path(v) for k, v in delay_files.items()}
    codes = _province_code_map(locations)
    names = dict(
        zip(locations["location"], locations["location_name"], strict=True)
    )
    frames = []
    for vintage, path in sorted(files.items()):
        df = pd.read_csv(path)
        key = "location_name" if "location_name" in df.columns else "location"
        loc = df[key].map(codes)
        if loc.isna().any():
            missing = sorted(set(df.loc[loc.isna(), key]))
            raise ValueError(f"Unknown provinces in {path}: {missing}")
        frames.append(
            pd.DataFrame(
                {
                    "vintage": vintage,
                    "location": loc,
                    "location_name": loc.map(names),
                    "onset_week": df["week_ending"].astype(str),
                    "delay": df["days_sym_to_report"].astype("int64"),
                }
            )
        )
    out = pd.concat(frames, ignore_index=True)
    return out.sort_values(
        ["vintage", "location", "onset_week", "delay"]
    ).reset_index(drop=True)


def build_tidy_data(
    raw_dir: Path | str, out_dir: Path | str
) -> dict[str, pd.DataFrame]:
    """Tidy the raw downloads and write the tables to `out_dir`.

    Writes `locations.csv`, `target.csv`, `daily.csv`,
    `daily_zones.csv.gz`, `delays.csv.gz`, `health_zones.csv.gz` and
    `national-coherence.csv` (see `national_coherence`), and copies the
    hub configuration to `hub-config/` and the download manifest to
    `sources.json`.

    Parameters
    ----------
    raw_dir : pathlib.Path or str
        Directory written by `fetch_hub_data`.
    out_dir : pathlib.Path or str
        Directory for the tidy tables, usually `data/`.

    Returns
    -------
    dict of str to pandas.DataFrame
        The tidy tables, keyed by name.
    """
    raw, out = Path(raw_dir), Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    locations = tidy_locations(raw / "auxiliary-data" / "locations.csv")
    daily_csv = raw / "auxiliary-data" / "daily-confirmed-case.csv"
    daily = tidy_daily(daily_csv)
    daily_zones = tidy_daily(daily_csv, level="zone")
    target = tidy_target(raw / "target-data" / "time-series.csv", daily)
    delays = tidy_delays(raw / "delays", locations)
    zones = tidy_health_zones(raw / "target-data" / "time-series.csv")
    tables = {
        "locations": locations,
        "target": target,
        "daily": daily,
        "daily_zones": daily_zones,
        "delays": delays,
        "health_zones": zones,
    }
    locations.to_csv(out / "locations.csv", index=False)
    target.to_csv(out / "target.csv", index=False)
    daily.to_csv(out / "daily.csv", index=False)
    daily_zones.to_csv(out / "daily_zones.csv.gz", index=False)
    delays.to_csv(out / "delays.csv.gz", index=False)
    zones.to_csv(out / "health_zones.csv.gz", index=False)
    national_coherence(target).to_csv(
        out / "national-coherence.csv", index=False
    )
    config = out / "hub-config"
    config.mkdir(exist_ok=True)
    for path in (raw / "hub-config").glob("*.json"):
        (config / path.name).write_text(path.read_text())
    manifest = raw / "manifest.json"
    if manifest.exists():
        (out / "sources.json").write_text(manifest.read_text())
    return tables


COHERENCE_STREAMS: Final = ("observed", "imputed", "total")
"""Streams compared between the national row and the provinces."""

COHERENCE_HANDLING: Final = (
    "The submitted national value is the sum of the province values plus an "
    "unassigned component, for each sample. The unassigned component holds "
    "provinces the model does not report and the cases the hub counts "
    "nationally but in no province: the data's gap in weeks with data, and "
    "its recent share of the province total in later weeks."
)
"""How the models handle a gap between the national data and the provinces."""


def national_coherence(
    target: pd.DataFrame, national: str = "CD"
) -> pd.DataFrame:
    """Compare the national counts with the sum of the provinces.

    Parameters
    ----------
    target : pandas.DataFrame
        Tidy target table (`tidy_target` or `load_target`).
    national : str
        Location code of the country.

    Returns
    -------
    pandas.DataFrame
        One row per kind, vintage, onset week and stream (`observed`,
        `imputed`, `total`), with the national count, the sum of the
        provinces and `gap` (national minus provinces). A week missing on
        one side counts as zero there.
    """
    keys = ["kind", "vintage", "target_end_date"]
    streams = list(COHERENCE_STREAMS)
    nat = (
        target.loc[target["location"] == national].groupby(keys)[streams].sum()
    )
    prov = (
        target.loc[target["location"] != national].groupby(keys)[streams].sum()
    )
    nat, prov = nat.align(prov, join="outer", fill_value=0)
    long = nat.reset_index().melt(
        id_vars=keys, var_name="stream", value_name="national"
    )
    long["provinces"] = (
        prov.reset_index()
        .melt(id_vars=keys, var_name="stream", value_name="provinces")[
            "provinces"
        ]
        .to_numpy()
    )
    long["gap"] = long["national"] - long["provinces"]
    return long.sort_values([*keys, "stream"]).reset_index(drop=True)


def delays_for_vintage(
    delays: pd.DataFrame, vintage: str
) -> tuple[pd.DataFrame, str]:
    """Delay rows for a vintage, using the latest earlier one if it has none.

    The hub does not publish delays with every vintage (2026-08-18 has
    none) and can publish them late. Rather than fail, the latest delay
    vintage on or before `vintage` stands in, relabelled to `vintage`.

    Parameters
    ----------
    delays : pandas.DataFrame
        Tidy delay rows with a `vintage` column.
    vintage : str
        The data vintage being forecast from.

    Returns
    -------
    tuple of pandas.DataFrame and str
        The delay table with rows for `vintage`, and a note of what was
        used instead (empty when the vintage had its own delays).
    """
    if delays.empty or (delays["vintage"] == vintage).any():
        return delays, ""
    earlier = sorted(str(v) for v in delays["vintage"].unique() if v <= vintage)
    if not earlier:
        return delays, (
            f"delays: none published by {vintage}, so the model's default "
            "completeness lag was used"
        )
    use = earlier[-1]
    extra = delays.loc[delays["vintage"] == use].assign(vintage=vintage)
    note = (
        f"delays: none published for {vintage}, so used the {use} delays, "
        "the latest available"
    )
    return pd.concat([delays, extra], ignore_index=True), note


def coherence_summary(
    target: pd.DataFrame, vintage: str, national: str = "CD"
) -> dict[str, Any]:
    """Summarise national coherence for one real vintage.

    This is the `coherence.json` record the submission CI reads (see
    `ci/INTEGRATION.md`).

    Parameters
    ----------
    target : pandas.DataFrame
        Tidy target table.
    vintage : str
        The real vintage to check.
    national : str
        Location code of the country.

    Returns
    -------
    dict
        `as_of`, `coherent`, per-stream summaries, the rows with a gap and
        how the models handle a gap.

    Raises
    ------
    ValueError
        If the vintage has no real data.
    """
    rows = target.loc[
        (target["kind"] == "real") & (target["vintage"] == vintage)
    ]
    if rows.empty:
        raise ValueError(f"No real data for vintage {vintage}.")
    table = national_coherence(rows, national)
    gaps = table.loc[table["gap"] != 0]
    streams = {
        s: {
            "weeks_with_gap": int((g["gap"] != 0).sum()),
            "max_abs_gap": int(g["gap"].abs().max()) if len(g) else 0,
            "total_gap": int(g["gap"].sum()),
        }
        for s, g in table.groupby("stream")
    }
    return {
        "schema_version": 1,
        "as_of": vintage,
        "national": national,
        "coherent": gaps.empty,
        "streams": streams,
        "rows": [
            {
                "target_end_date": str(r["target_end_date"]),
                "stream": str(r["stream"]),
                "national": int(r["national"]),
                "provinces": int(r["provinces"]),
                "gap": int(r["gap"]),
            }
            for r in gaps.to_dict("records")
        ],
        "handling": COHERENCE_HANDLING,
    }


# --------------------------------------------------------------------------
# Load
# --------------------------------------------------------------------------

_DEFAULT_DATA_DIR = Path("data")


def _data_dir(data_dir: Path | str | None) -> Path:
    return Path(data_dir) if data_dir is not None else _DEFAULT_DATA_DIR


def load_target(data_dir: Path | str | None = None) -> pd.DataFrame:
    """Load the tidy target table written by `build_tidy_data`.

    Parameters
    ----------
    data_dir : pathlib.Path or str, optional
        Directory holding the tidy tables. Defaults to `data/`.

    Returns
    -------
    pandas.DataFrame
        See `tidy_target`.
    """
    return pd.read_csv(_data_dir(data_dir) / "target.csv", dtype={"kind": str})


def load_daily(data_dir: Path | str | None = None) -> pd.DataFrame:
    """Load the tidy daily onsets table. See `tidy_daily`.

    Parameters
    ----------
    data_dir : pathlib.Path or str, optional
        Directory holding the tidy tables. Defaults to `data/`.

    Returns
    -------
    pandas.DataFrame
        See `tidy_daily`.
    """
    return pd.read_csv(_data_dir(data_dir) / "daily.csv")


def load_daily_zones(data_dir: Path | str | None = None) -> pd.DataFrame:
    """Load the tidy daily onsets of the health zones.

    See `tidy_daily` with `level="zone"`.

    Parameters
    ----------
    data_dir : pathlib.Path or str, optional
        Directory holding the tidy tables. Defaults to `data/`.

    Returns
    -------
    pandas.DataFrame
        See `tidy_daily`.
    """
    return pd.read_csv(_data_dir(data_dir) / "daily_zones.csv.gz")


def load_delays(data_dir: Path | str | None = None) -> pd.DataFrame:
    """Load the tidy delay line data. See `tidy_delays`.

    Parameters
    ----------
    data_dir : pathlib.Path or str, optional
        Directory holding the tidy tables. Defaults to `data/`.

    Returns
    -------
    pandas.DataFrame
        See `tidy_delays`.
    """
    return pd.read_csv(_data_dir(data_dir) / "delays.csv.gz")


def load_locations(data_dir: Path | str | None = None) -> pd.DataFrame:
    """Load the tidy locations table. See `tidy_locations`.

    Parameters
    ----------
    data_dir : pathlib.Path or str, optional
        Directory holding the tidy tables. Defaults to `data/`.

    Returns
    -------
    pandas.DataFrame
        See `tidy_locations`.
    """
    return pd.read_csv(_data_dir(data_dir) / "locations.csv")


def load_health_zones(data_dir: Path | str | None = None) -> pd.DataFrame:
    """Load the tidy health zone table. See `tidy_health_zones`.

    Parameters
    ----------
    data_dir : pathlib.Path or str, optional
        Directory holding the tidy tables. Defaults to `data/`.

    Returns
    -------
    pandas.DataFrame
        See `tidy_health_zones`.
    """
    return pd.read_csv(_data_dir(data_dir) / "health_zones.csv.gz")


# --------------------------------------------------------------------------
# Derived summaries
# --------------------------------------------------------------------------


def vintages(target: pd.DataFrame, kind: str | None = None) -> list[str]:
    """List the vintages in a target table, oldest first.

    Parameters
    ----------
    target : pandas.DataFrame
        A target table with `vintage` and `kind` columns.
    kind : str, optional
        Keep only vintages of this kind, such as `"real"`.

    Returns
    -------
    list of str
        ISO dates.
    """
    rows = target if kind is None else target.loc[target["kind"] == kind]
    return sorted({str(v) for v in rows["vintage"]})


def imputed_share(target: pd.DataFrame) -> pd.DataFrame:
    """Return the share of each week's cases with an imputed onset.

    This measures the missing-onset fraction directly from the data.

    Parameters
    ----------
    target : pandas.DataFrame
        A target table.

    Returns
    -------
    pandas.DataFrame
        The input with an added `imputed_share` column, `imputed / total`,
        missing where `total` is zero.
    """
    out = target.copy()
    total = out["total"].where(out["total"] > 0)
    out["imputed_share"] = out["imputed"] / total
    return out


def newly_missing(target: pd.DataFrame) -> pd.DataFrame:
    """Return cases added between consecutive vintages, by onset status.

    For each location and each vintage after the first, the change in
    total observed and total imputed onsets since the previous vintage of
    the same kind.
    The imputed change approximates the number of cases without an onset
    reported between the two vintages.
    It is net of cases whose onset was filled in later, so it can be
    negative.

    Parameters
    ----------
    target : pandas.DataFrame
        A target table.

    Returns
    -------
    pandas.DataFrame
        Columns `kind`, `location`, `vintage`, `previous_vintage`,
        `days`, `new_observed`, `new_imputed`, `new_total` and
        `new_imputed_share`.
    """
    sums = target.groupby(["kind", "location", "vintage"], as_index=False)[
        ["observed", "imputed", "total"]
    ].sum()
    sums = sums.sort_values(["kind", "location", "vintage"])
    grp = sums.groupby(["kind", "location"])
    sums["previous_vintage"] = grp["vintage"].shift(1)
    for col in ("observed", "imputed", "total"):
        sums[f"new_{col}"] = sums[col] - grp[col].shift(1)
    out = sums.dropna(subset=["previous_vintage"]).copy()
    out["days"] = (
        pd.to_datetime(out["vintage"]) - pd.to_datetime(out["previous_vintage"])
    ).dt.days
    new_total = out["new_total"].where(out["new_total"] > 0)
    out["new_imputed_share"] = out["new_imputed"] / new_total
    cols = [
        "kind",
        "location",
        "vintage",
        "previous_vintage",
        "days",
        "new_observed",
        "new_imputed",
        "new_total",
        "new_imputed_share",
    ]
    out = out[cols].astype(
        {"new_observed": "int64", "new_imputed": "int64", "new_total": "int64"}
    )
    return out.reset_index(drop=True)


def revision_table(
    target: pd.DataFrame, reference_vintage: str | None = None
) -> pd.DataFrame:
    """Compare each vintage's counts with a later reference vintage.

    This is the reporting triangle in weekly form: how recent weeks fill
    in as later vintages arrive.

    Parameters
    ----------
    target : pandas.DataFrame
        A target table of one kind.
    reference_vintage : str, optional
        The vintage to compare against. Defaults to the latest.

    Returns
    -------
    pandas.DataFrame
        Columns `vintage`, `location`, `target_end_date`, `weeks_before`
        (whole weeks from the target week to the vintage's epi week),
        `total`, `reference_total` and `fraction` (`total` over
        `reference_total`, missing where the latter is zero).
    """
    ref = reference_vintage or max(target["vintage"])
    latest = target.loc[
        target["vintage"] == ref, ["location", "target_end_date", "total"]
    ].rename(columns={"total": "reference_total"})
    out = target.loc[
        target["vintage"] <= ref,
        ["vintage", "location", "target_end_date", "total"],
    ].merge(latest, on=["location", "target_end_date"], how="left")
    out["reference_total"] = out["reference_total"].fillna(0)
    vintage_week = pd.to_datetime(
        pd.Series([epiweek_end(v).isoformat() for v in out["vintage"]])
    )
    gap = vintage_week - pd.to_datetime(out["target_end_date"])
    out["weeks_before"] = (gap.dt.days // 7).to_numpy()
    out["fraction"] = out["total"] / out["reference_total"].where(
        out["reference_total"] > 0
    )
    return out.reset_index(drop=True)
