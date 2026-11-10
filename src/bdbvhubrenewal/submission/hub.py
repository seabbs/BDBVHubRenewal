# Copyright (c) 2026 Sam Abbott. MIT licence, see LICENSE.
"""Write hub submissions and model metadata.

A submission is a CSV with columns `reference_date`, `forecast_date`,
`target`, `target_end_date`, `location`, `output_type`, `output_type_id`
and `value`.
Every row is one week of one sample trajectory.
`forecast_date` is the day the PR is opened.
A resubmission in the same week keeps the file on the hub's main branch
unchanged and appends rows with one new `forecast_date`.
The round `BDBV 2026 nowcasting and forecasting` takes exactly 100
samples for each `(reference_date, location, target)`, and each sample
index covers the same target weeks within a location.

`submission.validate` checks the written files as `hubValidations` does.
The authoritative check is still `hubValidations` (`task validate-hub`).
"""

from __future__ import annotations

import datetime as dt
import re
import tomllib
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Final

import pandas as pd
import yaml

from bdbvhubrenewal.data.dates import as_date
from bdbvhubrenewal.data.tidy import write_csv

__all__ = [
    "HUB_COLUMNS",
    "METADATA_DEFAULTS",
    "OUTPUT_TYPE",
    "ROUND_NAME",
    "SUBMISSION_CONFIG",
    "TEAM_ABBR",
    "model_abbr_for",
    "model_id",
    "national_method",
    "render_model_metadata",
    "run_date",
    "submission_path",
    "to_hub_table",
    "write_model_metadata",
    "write_submission",
]

HUB_COLUMNS: Final = (
    "reference_date",
    "forecast_date",
    "target",
    "target_end_date",
    "location",
    "output_type",
    "output_type_id",
    "value",
)
"""Columns of a submission file, in the order we write them."""

ROUND_NAME: Final = "BDBV 2026 nowcasting and forecasting"
"""The round this repository submits to."""

OUTPUT_TYPE: Final = "sample"
"""The only output type the round accepts."""

TEAM_ABBR: Final = "seabbs"
"""Team abbreviation, of our hub models."""

# The schema allows 16 characters and the README says "less than 15" or
# "less than 16", so keep to 14 to satisfy all three
# (docs/submission/hub-spec.md).
_ID_PART = re.compile(r"^[A-Za-z0-9_]{1,14}$")

NATIONAL: Final = "CD"
"""Location code of the whole country."""

SUBMISSION_CONFIG: Final = Path("config/hub.toml")
"""Where the submitted model abbreviation is set."""


def model_abbr_for(model: str, path: Path | str = SUBMISSION_CONFIG) -> str:
    """Return the hub abbreviation a model submits under.

    A model with its own entry under `[models]` in `config/hub.toml` is a
    hub model of its own. Every other model submits under the
    `[submission]` abbreviation, one value not derived from a model key,
    so every renewal variant and every fallback submits under one id.

    Parameters
    ----------
    model : str
        The model key, such as `"randomwalknowcast"`.
    path : pathlib.Path or str
        The configuration file.

    Returns
    -------
    str
        The abbreviation, such as `"baseline"`.
    """
    with Path(path).open("rb") as fh:
        config = tomllib.load(fh)
    own = config.get("models", {}).get(model, {})
    return str(own.get("model_abbr") or config["submission"]["model_abbr"])


def model_id(model_abbr: str, team_abbr: str = TEAM_ABBR) -> str:
    """Join a team and model abbreviation into a hub model id.

    Parameters
    ----------
    model_abbr : str
        Model abbreviation, such as `"randomwalk"`.
    team_abbr : str
        Team abbreviation.

    Returns
    -------
    str
        `"<team_abbr>-<model_abbr>"`.

    Raises
    ------
    ValueError
        If either part is not 1 to 14 letters, digits or underscores.
    """
    for part in (team_abbr, model_abbr):
        if not _ID_PART.match(part):
            raise ValueError(
                f"{part!r} must be 1 to 14 letters, digits or underscores."
            )
    return f"{team_abbr}-{model_abbr}"


def national_method(model: object) -> str:
    """Return how a model makes its national value, `"sum"` or `"direct"`.

    Parameters
    ----------
    model : ForecastModel, str or None
        A model, or the name of a registered one. A name that is not
        registered, or None, is `"sum"`.

    Returns
    -------
    str
        `Settings.national` of a random walk model, otherwise `"sum"`.
    """
    if model is None or isinstance(model, str):
        from bdbvhubrenewal.forecasters import get_model

        try:
            model = get_model(str(model))
        except KeyError:
            return "sum"
    settings = getattr(model, "settings", None)
    return str(getattr(settings, "national", "sum"))


def to_hub_table(
    samples: pd.DataFrame,
    reference_date: dt.date | str,
    target: str = "wk inc confirmed cases",
    *,
    unassigned: pd.DataFrame | None = None,
    zones: pd.DataFrame | None = None,
    n_samples: int = 100,
    forecast_date: dt.date | str | None = None,
    national: str = "sum",
) -> pd.DataFrame:
    """Turn forecast samples into the hub's submission table.

    The table meets every guarantee in `docs/submission/hub-spec.md`:
    sample ids `1..n_samples` reused in every location, one row per id for every
    week, the same weeks for every location, whole non-negative counts
    and the national value as the sum of the provinces for each sample.
    Any national rows in `samples` are replaced, unless `national` is
    `"direct"`.

    Parameters
    ----------
    samples : pandas.DataFrame
        Columns `location`, `target_end_date`, `sample` (an integer
        index, one joint draw across weeks and provinces) and `value`.
    reference_date : datetime.date or str
        The round's reference date.
    target : str
        The target name.
    unassigned : pandas.DataFrame, optional
        A modelled component of the national count not assigned to any
        province, with columns `target_end_date`, `sample` and `value`.
        It is added to the national sum. Missing rows count as zero.
    zones : pandas.DataFrame, optional
        Health-zone samples (columns as `samples`), such as from
        `forecasters.zones.ZoneAllocator`. They must sum to the rounded
        province samples, sample by sample, and are not added to the
        national value.
    n_samples : int
        Number of samples each location must have.
    forecast_date : datetime.date or str, optional
        The day the PR is opened. The default is today (UTC), see
        `run_date`. The hub requires it to be on or before the
        reference date and within a day of the validation date.
    national : str
        `"sum"` makes national the sum of the provinces (and
        `unassigned`). `"direct"` keeps the national rows of `samples`, a
        model's own national fit, which the hub allows to differ from
        the sum.

    Returns
    -------
    pandas.DataFrame
        Columns `HUB_COLUMNS`, sorted by location, sample and week.
        Values are rounded to whole cases before the national sum, so a
        summed national value is exactly the sum of the written province
        values.

    Raises
    ------
    ValueError
        If a province or zone does not have sample ids `1..n_samples`, a
        sample lacks a week, locations cover different weeks, the zones
        do not sum to their provinces, or a direct national has no
        national rows or is given an unassigned component.

    Examples
    --------
    >>> from bdbvhubrenewal.submission.hub import (
    ...     model_id, to_hub_table, write_submission
    ... )
    >>> samples = forecasts.query("model == 'randomwalknowcast'")
    >>> table = to_hub_table(samples, task.reference_date)
    >>> write_submission(table, "output/submission", model_id("rwnowcast"))
    """
    ref = as_date(reference_date).isoformat()
    fdate = as_date(forecast_date or run_date()).isoformat()
    if national not in ("sum", "direct"):
        raise ValueError(f"national must be sum or direct, not {national!r}.")
    prov = _tidy(samples.loc[samples["location"] != NATIONAL])
    if prov.empty:
        raise ValueError("No province samples to write.")
    zone = _zone_rows(prov, zones)
    direct = prov.iloc[:0]
    if national == "direct":
        direct = _tidy(samples.loc[samples["location"] == NATIONAL])
        if direct.empty:
            raise ValueError("A direct national needs national samples.")
        if unassigned is not None and not unassigned.empty:
            raise ValueError("A direct national has no unassigned part.")
    ids = set(range(1, n_samples + 1))
    weeks = sorted(set(prov["target_end_date"]))
    for loc, rows in pd.concat([prov, zone, direct]).groupby("location"):
        if set(rows["sample"]) != ids:
            raise ValueError(
                f"{loc}: sample ids must be 1 to {n_samples}, the same in "
                "every location."
            )
        if sorted(set(rows["target_end_date"])) != weeks:
            raise ValueError(f"{loc}: every location needs the same weeks.")
        if (
            len(rows) != n_samples * len(weeks)
            or rows.duplicated(["sample", "target_end_date"]).any()
        ):
            raise ValueError(
                f"{loc}: every sample needs exactly one row for each week."
            )
    if national == "direct":
        both = pd.concat([prov, direct, zone], ignore_index=True)
        return _hub_rows(both, ref, fdate, target)
    summed = (
        prov.groupby(["target_end_date", "sample"])["value"].sum().reset_index()
    )
    if unassigned is not None and not unassigned.empty:
        extra = (
            unassigned.assign(
                target_end_date=unassigned["target_end_date"].astype(str),
                sample=unassigned["sample"].astype("int64"),
                value=unassigned["value"].clip(lower=0).round(),
            )
            .groupby(["target_end_date", "sample"])["value"]
            .sum()
        )
        summed = summed.set_index(["target_end_date", "sample"])
        summed["value"] = summed["value"].add(extra, fill_value=0)
        summed = summed.reset_index()
    summed["location"] = NATIONAL
    both = pd.concat([prov, summed, zone], ignore_index=True)
    return _hub_rows(both, ref, fdate, target)


def _tidy(rows: pd.DataFrame) -> pd.DataFrame:
    """Type sample rows and round them to whole non-negative cases."""
    out = rows.copy()
    out["target_end_date"] = out["target_end_date"].astype(str)
    out["location"] = out["location"].astype(str)
    out["sample"] = out["sample"].astype("int64")
    out["value"] = out["value"].clip(lower=0).round()
    return out


def _hub_rows(
    both: pd.DataFrame, ref: str, fdate: str, target: str
) -> pd.DataFrame:
    """Return sample rows as the sorted hub table."""
    out = pd.DataFrame(
        {
            "reference_date": ref,
            "forecast_date": fdate,
            "target": target,
            "target_end_date": both["target_end_date"],
            "location": both["location"],
            "output_type": OUTPUT_TYPE,
            "output_type_id": both["sample"].astype("int64"),
            "value": both["value"].astype("int64"),
        }
    )
    return out.sort_values(
        ["location", "output_type_id", "target_end_date"]
    ).reset_index(drop=True)[list(HUB_COLUMNS)]


def _zone_rows(prov: pd.DataFrame, zones: pd.DataFrame | None) -> pd.DataFrame:
    """Tidy zone samples and check they sum to the province samples."""
    if zones is None or zones.empty:
        return prov.iloc[:0]
    from bdbvhubrenewal.forecasters.zones import check_zone_coherence

    zone = zones[list(prov.columns)].copy()
    zone["target_end_date"] = zone["target_end_date"].astype(str)
    zone["location"] = zone["location"].astype(str)
    zone["sample"] = zone["sample"].astype("int64")
    zone["value"] = zone["value"].clip(lower=0).round()
    gaps = check_zone_coherence(prov, zone)
    if gaps:
        raise ValueError(f"{len(gaps)} zone sums differ: {gaps[0]}.")
    return zone


def submission_path(
    root: Path | str, reference_date: dt.date | str, mid: str
) -> Path:
    """Return where a submission file goes in a hub checkout.

    Parameters
    ----------
    root : pathlib.Path or str
        Root of the hub checkout (or any directory laid out like one).
    reference_date : datetime.date or str
        The round's reference date.
    mid : str
        The model id, such as `"seabbs-randomwalk"`.

    Returns
    -------
    pathlib.Path
        `root/model-output/<mid>/<reference_date>-<mid>.csv`.
    """
    ref = as_date(reference_date).isoformat()
    return Path(root) / "model-output" / mid / f"{ref}-{mid}.csv"


def run_date(now: dt.datetime | None = None) -> dt.date:
    """Return the date a run opens its PR on, in UTC.

    Parameters
    ----------
    now : datetime.datetime, optional
        The time of the run. Defaults to now.

    Returns
    -------
    datetime.date
        The UTC date.
    """
    moment = now or dt.datetime.now(dt.UTC)
    return moment.astimezone(dt.UTC).date()


def write_submission(
    table: pd.DataFrame,
    root: Path | str,
    mid: str,
) -> Path:
    """Write a submission table to its place under `root`.

    Parameters
    ----------
    table : pandas.DataFrame
        Output of `to_hub_table`.
    root : pathlib.Path or str
        Root of the hub checkout or an output directory.
    mid : str
        The model id.

    Returns
    -------
    pathlib.Path
        The path written.
    """
    refs = table["reference_date"].unique()
    if len(refs) != 1:
        raise ValueError("A submission must have one reference date.")
    if len(table["forecast_date"].unique()) != 1:
        raise ValueError("A submission must have one forecast date.")
    path = submission_path(root, str(refs[0]), mid)
    path.parent.mkdir(parents=True, exist_ok=True)
    write_csv(table, path)
    return path


# --------------------------------------------------------------------------
# Model metadata
# --------------------------------------------------------------------------

FUNDING: Final = (
    "This study is funded by the National Institute for Health and Care "
    "Research (NIHR) Health Protection Research Unit in Health Analytics & "
    "Modelling, a partnership between the UK Health Security Agency, "
    "Imperial College London and the London School of Hygiene & Tropical "
    "Medicine (grant code NIHR207404). The views expressed are those of the "
    "author(s) and not necessarily those of the NIHR, UK Health Security "
    "Agency or the Department of Health and Social Care."
)
"""Sam Abbott's HPRU funding statement (`team_funding` in the metadata)."""

METADATA_DEFAULTS: Final[dict[str, Any]] = {
    "team_name": "seabbs",
    "team_abbr": TEAM_ABBR,
    "model_contributors": [
        {
            "name": "Sam Abbott",
            "affiliation": "London School of Hygiene & Tropical Medicine",
            "orcid": "0000-0001-8057-8037",
            "email": "sam.abbott@lshtm.ac.uk",
        }
    ],
    "license": "CC-BY-4.0",
    "repo_url": "https://github.com/seabbs/BDBVHubRenewal",
    "website_url": "https://samabbott.co.uk/BDBVHubRenewal/",
    "ensemble_of_hub_models": False,
    "team_funding": FUNDING,
}
"""Fields shared by every model's metadata."""


def render_model_metadata(
    model_abbr: str,
    model_name: str,
    methods: str,
    methods_long: str,
    data_inputs: str,
    model_version: str | None = None,
    overrides: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a model metadata record in the hub's field order.

    Parameters
    ----------
    model_abbr : str
        Model abbreviation (the part after `seabbs-`).
    model_name : str
        Full model name, under 50 characters.
    methods : str
        Short description, at most 200 characters.
    methods_long : str
        Full description.
    data_inputs : str
        The data the model uses.
    model_version : str, optional
        A version identifier, such as a git SHA.
    overrides : mapping, optional
        Any fields to replace in `METADATA_DEFAULTS`.

    Returns
    -------
    dict
        The metadata record.
    """
    base = {**METADATA_DEFAULTS, **(overrides or {})}
    record: dict[str, Any] = {
        "team_name": base["team_name"],
        "team_abbr": base["team_abbr"],
        "model_name": model_name,
        "model_abbr": model_abbr,
        "model_contributors": base["model_contributors"],
        "license": base["license"],
        "model_details": {
            "data_inputs": data_inputs,
            "methods": methods,
            "methods_long": methods_long,
        },
    }
    if model_version is not None:
        record["model_version"] = model_version
    for key in (
        "website_url",
        "repo_url",
        "team_funding",
        "ensemble_of_hub_models",
    ):
        record[key] = base[key]
    return record


def write_model_metadata(
    record: Mapping[str, Any], directory: Path | str
) -> Path:
    """Write a metadata record to `<team_abbr>-<model_abbr>.yml`.

    Parameters
    ----------
    record : mapping
        The metadata record.
    directory : pathlib.Path or str
        Where to write it, such as `hub/model-metadata/`.

    Returns
    -------
    pathlib.Path
        The path written.
    """
    path = Path(directory) / f"{record['team_abbr']}-{record['model_abbr']}.yml"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        yaml.safe_dump(
            dict(record), sort_keys=False, allow_unicode=True, width=1000
        )
    )
    return path
