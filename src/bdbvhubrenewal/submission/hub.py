# Copyright (c) 2026 Sam Abbott. MIT licence, see LICENSE.
"""Write and check hub submissions and model metadata.

A submission is a CSV with columns `reference_date`, `target`,
`target_end_date`, `location`, `output_type`, `output_type_id` and
`value`.
Every row is one week of one sample trajectory.
The round `BDBV 2026 nowcasting and forecasting` takes exactly 100
samples for each `(reference_date, location, target)`, and each sample
index covers the same target weeks within a location.

`validate_submission` mirrors the checks `hubValidations` runs on a pull
request, so problems show up before a PR is opened.
The authoritative check is still `hubValidations` (`task validate-hub`).
"""

from __future__ import annotations

import datetime as dt
import json
import math
import re
import tomllib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

import jsonschema
import pandas as pd
import yaml

from bdbvhubrenewal.data.dates import (
    HUB_TIMEZONE,
    as_date,
    is_saturday,
    submission_window,
)

__all__ = [
    "HUB_COLUMNS",
    "METADATA_DEFAULTS",
    "OUTPUT_TYPE",
    "ROUND_NAME",
    "SUBMISSION_CONFIG",
    "TEAM_ABBR",
    "HubRound",
    "load_hub_round",
    "model_abbr_for",
    "model_id",
    "render_model_metadata",
    "submission_model_abbr",
    "submission_path",
    "to_hub_table",
    "validate_model_id",
    "validate_model_metadata",
    "validate_submission",
    "validate_submission_file",
    "write_model_metadata",
    "write_submission",
]

HUB_COLUMNS: Final = (
    "reference_date",
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


@dataclass(frozen=True)
class HubRound:
    """The task configuration of one round, read from `tasks.json`.

    Attributes
    ----------
    round_name : str
        Name of the round.
    reference_dates : tuple of str
        Accepted reference dates.
    target_end_dates : tuple of str
        Accepted target end dates.
    locations : frozenset of str
        Accepted location codes.
    targets : tuple of str
        Accepted targets.
    n_samples : int
        Samples required per compound task (minimum equals maximum here).
    compound_taskid_set : tuple of str
        Task ids that a sample index is shared within.
    window_start : int
        First day of the submission window, relative to the reference date.
    window_end : int
        Last day of the submission window, relative to the reference date.
    """

    round_name: str
    reference_dates: tuple[str, ...]
    target_end_dates: tuple[str, ...]
    locations: frozenset[str]
    targets: tuple[str, ...]
    n_samples: int
    compound_taskid_set: tuple[str, ...]
    window_start: int
    window_end: int


def _options(task: Mapping[str, Any]) -> list[str]:
    values = (task.get("required") or []) + (task.get("optional") or [])
    return [str(v) for v in values]


def load_hub_round(
    tasks_json: Path | str, round_name: str = ROUND_NAME
) -> HubRound:
    """Read the configuration of a round from the hub's `tasks.json`.

    Parameters
    ----------
    tasks_json : pathlib.Path or str
        Path to `hub-config/tasks.json`.
    round_name : str
        The round to read.

    Returns
    -------
    HubRound
        The round's settings.

    Raises
    ------
    ValueError
        If the round is not found or has no sample output type.
    """
    config = json.loads(Path(tasks_json).read_text())
    for rnd in config["rounds"]:
        if rnd.get("round_name") != round_name:
            continue
        task = rnd["model_tasks"][0]
        ids = task["task_ids"]
        sample = task["output_type"].get(OUTPUT_TYPE)
        if sample is None:
            raise ValueError(f"Round {round_name!r} takes no samples.")
        params = sample["output_type_id_params"]
        if params["min_samples_per_task"] != params["max_samples_per_task"]:
            raise ValueError("Expected a fixed number of samples per task.")
        due = rnd["submissions_due"]
        return HubRound(
            round_name=round_name,
            reference_dates=tuple(_options(ids["reference_date"])),
            target_end_dates=tuple(_options(ids["target_end_date"])),
            locations=frozenset(_options(ids["location"])),
            targets=tuple(_options(ids["target"])),
            n_samples=int(params["min_samples_per_task"]),
            compound_taskid_set=tuple(params["compound_taskid_set"]),
            window_start=int(due["start"]),
            window_end=int(due["end"]),
        )
    raise ValueError(f"Round {round_name!r} not found in {tasks_json}.")


SUBMISSION_CONFIG: Final = Path("config/hub.toml")
"""Where the submitted model abbreviation is set."""


def submission_model_abbr(path: Path | str = SUBMISSION_CONFIG) -> str:
    """Return the model abbreviation every submission uses.

    It is one value in `config/hub.toml`, not derived from a model key,
    so every renewal variant and every fallback submits under one id.

    Parameters
    ----------
    path : pathlib.Path or str
        The configuration file.

    Returns
    -------
    str
        The abbreviation, such as `"renewal"`.
    """
    with Path(path).open("rb") as fh:
        config = tomllib.load(fh)
    return str(config["submission"]["model_abbr"])


def model_abbr_for(model: str, path: Path | str = SUBMISSION_CONFIG) -> str:
    """Return the hub abbreviation a model submits under.

    A model with its own entry under `[models]` in `config/hub.toml` is a
    hub model of its own. Every other model submits under the renewal
    abbreviation (`submission_model_abbr`).

    Parameters
    ----------
    model : str
        The model key, such as `"persistencenow"`.
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
        Model abbreviation, such as `"persistence"`.
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


def validate_model_id(value: str) -> list[str]:
    """Check a model id against the hub's naming rules.

    Parameters
    ----------
    value : str
        A model id such as `"seabbs-persistence"`.

    Returns
    -------
    list of str
        Problems found. Empty if the id is valid.
    """
    parts = value.split("-")
    if len(parts) != 2:
        return [f"Model id {value!r} must be `team-model` with one hyphen."]
    return [
        f"{part!r} in {value!r} must be 1 to 14 letters, digits or underscores."
        for part in parts
        if not _ID_PART.match(part)
    ]


def to_hub_table(
    samples: pd.DataFrame,
    reference_date: dt.date | str,
    target: str = "wk inc confirmed cases",
    *,
    unassigned: pd.DataFrame | None = None,
    national: pd.DataFrame | None = None,
    zones: pd.DataFrame | None = None,
    n_samples: int = 100,
) -> pd.DataFrame:
    """Turn forecast samples into the hub's submission table.

    The table meets every guarantee in `docs/submission/hub-spec.md`:
    sample ids `1..n_samples` reused in every location, one row per id for every
    week, the same weeks for every location, whole non-negative counts
    and the national value as the sum of the provinces for each sample,
    unless `national` gives the model's own national samples.
    Any national rows in `samples` are replaced.

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
    national : pandas.DataFrame, optional
        National samples from a model that fits the nation on its own
        (columns `target_end_date`, `sample`, `value`). They are written
        as they are, in place of the sum, and `unassigned` is ignored.
    zones : pandas.DataFrame, optional
        Health-zone samples (columns as `samples`), such as from
        `forecasters.zones.ZoneAllocator`. They must sum to the rounded
        province samples, sample by sample, and are not added to the
        national value.
    n_samples : int
        Number of samples each location must have.

    Returns
    -------
    pandas.DataFrame
        Columns `HUB_COLUMNS`, sorted by location, sample and week.
        Values are rounded to whole cases before the national sum, so the
        national value is exactly the sum of the written province values.

    Raises
    ------
    ValueError
        If a province or zone does not have sample ids `1..n_samples`, a
        sample lacks a week, locations cover different weeks, or the
        zones do not sum to their provinces.
    """
    ref = as_date(reference_date).isoformat()
    prov = samples.loc[samples["location"] != NATIONAL].copy()
    if prov.empty:
        raise ValueError("No province samples to write.")
    prov["target_end_date"] = prov["target_end_date"].astype(str)
    prov["location"] = prov["location"].astype(str)
    prov["sample"] = prov["sample"].astype("int64")
    prov["value"] = prov["value"].clip(lower=0).round()
    zone = _zone_rows(prov, zones)
    ids = set(range(1, n_samples + 1))
    weeks = sorted(set(prov["target_end_date"]))
    for loc, rows in pd.concat([prov, zone]).groupby("location"):
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
    fitted = national is not None
    if national is not None:
        national = national.assign(
            target_end_date=national["target_end_date"].astype(str),
            sample=national["sample"].astype("int64"),
            value=national["value"].clip(lower=0).round(),
        )[["target_end_date", "sample", "value"]]
    else:
        national = (
            prov.groupby(["target_end_date", "sample"])["value"]
            .sum()
            .reset_index()
        )
    if not fitted and unassigned is not None and not unassigned.empty:
        extra = (
            unassigned.assign(
                target_end_date=unassigned["target_end_date"].astype(str),
                sample=unassigned["sample"].astype("int64"),
                value=unassigned["value"].clip(lower=0).round(),
            )
            .groupby(["target_end_date", "sample"])["value"]
            .sum()
        )
        national = national.set_index(["target_end_date", "sample"])
        national["value"] = national["value"].add(extra, fill_value=0)
        national = national.reset_index()
    national["location"] = NATIONAL
    both = pd.concat([prov, national, zone], ignore_index=True)
    out = pd.DataFrame(
        {
            "reference_date": ref,
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
        The model id, such as `"seabbs-persistence"`.

    Returns
    -------
    pathlib.Path
        `root/model-output/<mid>/<reference_date>-<mid>.csv`.
    """
    ref = as_date(reference_date).isoformat()
    return Path(root) / "model-output" / mid / f"{ref}-{mid}.csv"


def write_submission(table: pd.DataFrame, root: Path | str, mid: str) -> Path:
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
    path = submission_path(root, str(refs[0]), mid)
    path.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(path, index=False)
    return path


def _check_file_name(file_name: str, mid: str | None) -> list[str]:
    match = re.match(r"^(\d{4}-\d{2}-\d{2})-(.+)\.csv$", file_name)
    if match is None:
        return [f"File name {file_name!r} is not `YYYY-MM-DD-team-model.csv`."]
    errors = validate_model_id(match.group(2))
    if mid is not None and match.group(2) != mid:
        errors.append(f"File name model id {match.group(2)!r} is not {mid!r}.")
    return errors


def _file_model_id(file_name: str) -> str | None:
    match = re.match(r"^\d{4}-\d{2}-\d{2}-(.+)\.csv$", file_name)
    return match.group(1) if match else None


def _check_id_scheme(df: pd.DataFrame) -> list[str]:
    """Check sample ids the way hubValidations reads them.

    hubValidations treats all rows sharing an `output_type_id` as one
    sample. With ids reused across locations, every location must use the
    same ids. With ids unique to a location, every location must cover the
    same weeks. Mixing the two fails either way
    (docs/submission/hub-spec.md, s. 4).
    """
    ids = df.groupby("location")["output_type_id"].apply(frozenset)
    if len(ids) < 2:
        return []
    shared = frozenset.intersection(*ids)
    if shared:
        if len(set(ids)) > 1:
            return [
                "Sample ids are reused across locations, so every location "
                "must use the same sample ids."
            ]
        return []
    if any(a & b for a in ids for b in ids if a is not b):
        return [
            "Sample ids are partly shared between locations; use the same "
            "sample ids in every location."
        ]
    weeks = df.groupby("location")["target_end_date"].apply(frozenset)
    if len(set(weeks)) > 1:
        return [
            "Sample ids are unique to each location, so hubValidations "
            "needs the same weeks in every location."
        ]
    return []


def validate_submission(
    table: pd.DataFrame,
    hub_round: HubRound,
    *,
    file_name: str | None = None,
    mid: str | None = None,
    check_round: bool = True,
    submitted_at: dt.datetime | None = None,
    dir_name: str | None = None,
    metadata_dir: Path | str | None = None,
    metadata_dirs: Sequence[Path | str] = (),
) -> list[str]:
    """Check a submission table against the round's rules.

    The checks mirror `hubValidations::validate_submission`: file name,
    columns, round id, task id values, output type, sample counts and
    the sample structure, value type and range, and the submission
    window.

    Parameters
    ----------
    table : pandas.DataFrame
        The submission, as written or as read back from CSV.
    hub_round : HubRound
        The round's configuration.
    file_name : str, optional
        The file's name, checked against the reference date and model id.
    mid : str, optional
        The expected model id.
    check_round : bool
        Whether to require the reference date and target end dates to be
        listed in `tasks.json`. Turn off for backtests of past dates the
        hub never opened.
    submitted_at : datetime.datetime, optional
        If given, check that this time is inside the submission window.
    dir_name : str, optional
        Name of the directory holding the file, which must equal the
        model id in the file name.
    metadata_dir, metadata_dirs : pathlib.Path or str, optional
        If given, `<model_id>.yml` (or `.yaml`) must exist in one of them.

    Returns
    -------
    list of str
        Problems found. Empty if the table is valid.
    """
    errors: list[str] = []
    cols = list(table.columns)
    if sorted(cols) != sorted(HUB_COLUMNS):
        return [f"Columns {cols} are not {list(HUB_COLUMNS)}."]
    if table.empty:
        return ["The submission has no rows."]
    if table.isna().any().any():
        errors.append("The submission has missing values.")
    df = table.astype({"reference_date": str, "target_end_date": str})

    refs = df["reference_date"].unique()
    if len(refs) != 1:
        return [*errors, f"Expected one reference date, found {list(refs)}."]
    ref = str(refs[0])
    if not is_saturday(ref):
        errors.append(f"Reference date {ref} is not a Saturday.")
    if check_round and ref not in hub_round.reference_dates:
        errors.append(f"Reference date {ref} is not an open round.")
    if file_name is not None:
        errors.extend(_check_file_name(file_name, mid))
        if not file_name.startswith(ref):
            errors.append(f"File name {file_name} does not match {ref}.")
    file_mid = _file_model_id(file_name) if file_name else mid
    if dir_name is not None and file_mid is not None and dir_name != file_mid:
        errors.append(
            f"The directory {dir_name!r} must be named after the model id "
            f"{file_mid!r}."
        )
    dirs = [*metadata_dirs, *([metadata_dir] if metadata_dir else [])]
    if dirs and file_mid is not None:
        found = any(
            (Path(d) / f"{file_mid}.{ext}").exists()
            for d in dirs
            for ext in ("yml", "yaml")
        )
        if not found:
            errors.append(f"No model metadata file {file_mid}.yml found.")

    bad_targets = sorted(set(df["target"]) - set(hub_round.targets))
    if bad_targets:
        errors.append(f"Unknown targets {bad_targets}.")
    bad_locs = sorted(set(df["location"]) - hub_round.locations)
    if bad_locs:
        errors.append(f"Unknown locations {bad_locs}.")
    if set(df["output_type"]) != {OUTPUT_TYPE}:
        errors.append(f"Output type must be {OUTPUT_TYPE!r}.")

    weeks = sorted(set(df["target_end_date"]))
    not_sat = [w for w in weeks if not is_saturday(w)]
    if not_sat:
        errors.append(f"Target end dates {not_sat} are not Saturdays.")
    last = (as_date(ref) + dt.timedelta(days=21)).isoformat()
    first = min(hub_round.target_end_dates, default="0000-00-00")
    out_of_range = [w for w in weeks if w > last or w < first]
    if out_of_range:
        errors.append(
            f"Target end dates {out_of_range} are outside {first} to {last}."
        )
    if check_round:
        unlisted = sorted(set(weeks) - set(hub_round.target_end_dates))
        if unlisted:
            errors.append(f"Target end dates {unlisted} are not accepted.")

    ids = pd.to_numeric(df["output_type_id"], errors="coerce")
    if ids.isna().any() or not (ids == ids.round()).all():
        errors.append("Sample ids (`output_type_id`) must be integers.")
        return errors
    values = pd.to_numeric(df["value"], errors="coerce")
    if values.isna().any() or not values.map(math.isfinite).all():
        errors.append("Values must be finite numbers.")
    elif (values < 0).any():
        errors.append("Values must be non-negative.")

    keys = ["reference_date", "location", "target", "target_end_date"]
    if df.duplicated([*keys, "output_type_id"]).any():
        errors.append("Duplicate rows for a task and sample id.")
    compound = list(hub_round.compound_taskid_set)
    n_per = df.groupby([*compound, "target_end_date"])["output_type_id"]
    counts = n_per.nunique()
    wrong = counts[counts != hub_round.n_samples]
    if not wrong.empty:
        errors.append(
            f"{len(wrong)} tasks do not have exactly {hub_round.n_samples} "
            "samples."
        )
    # Each sample index must cover the same weeks within a compound task.
    weeks_per_sample = df.groupby([*compound, "output_type_id"])[
        "target_end_date"
    ].apply(lambda s: tuple(sorted(s)))
    if weeks_per_sample.groupby(level=compound).nunique().max() > 1:
        errors.append(
            "Within a location, every sample must cover the same weeks."
        )
    errors.extend(_check_id_scheme(df))

    if submitted_at is not None:
        start, end = submission_window(ref)
        when = submitted_at.astimezone(HUB_TIMEZONE)
        if not (start <= when <= end):
            errors.append(
                f"Submission time {when.isoformat()} is outside the window "
                f"{start.isoformat()} to {end.isoformat()}."
            )
    return errors


def validate_submission_file(
    path: Path | str,
    hub_round: HubRound,
    *,
    metadata_dirs: Sequence[Path | str] = (),
    check_round: bool = True,
    submitted_at: dt.datetime | None = None,
) -> list[str]:
    """Check a submission file where it sits.

    Reads the CSV and runs `validate_submission` with its file name, the
    name of its directory and, if `metadata_dirs` is given, a check that
    the model's metadata exists.

    Parameters
    ----------
    path : pathlib.Path or str
        The submission, at `.../<model_id>/<reference_date>-<model_id>.csv`.
    hub_round : HubRound
        The round's configuration.
    metadata_dirs : sequence of pathlib.Path or str
        Directories searched for `<model_id>.yml`.
    check_round : bool
        As for `validate_submission`.
    submitted_at : datetime.datetime, optional
        As for `validate_submission`.

    Returns
    -------
    list of str
        Problems found. Empty if the file is valid.
    """
    p = Path(path)
    try:
        table = pd.read_csv(p, dtype={"output_type_id": str})
    except (OSError, ValueError, pd.errors.ParserError) as err:
        return [f"Could not read {p}: {err}"]
    if "output_type_id" in table.columns:
        ids = table["output_type_id"].astype(str)
        if not ids.str.fullmatch(r"-?\d+").all():
            return ["Sample ids (`output_type_id`) must be integers."]
        table["output_type_id"] = ids.astype("int64")
    return validate_submission(
        table,
        hub_round,
        file_name=p.name,
        dir_name=p.parent.name,
        metadata_dirs=metadata_dirs,
        check_round=check_round,
        submitted_at=submitted_at,
    )


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


def validate_model_metadata(
    record: Mapping[str, Any],
    schema_json: Path | str,
    file_name: str | None = None,
) -> list[str]:
    """Check a metadata record against the hub's JSON schema.

    Parameters
    ----------
    record : mapping
        The metadata, as rendered or read from YAML.
    schema_json : pathlib.Path or str
        Path to `hub-config/model-metadata-schema.json`.
    file_name : str, optional
        The YAML file name, which must be `<team_abbr>-<model_abbr>.yml`.

    Returns
    -------
    list of str
        Problems found. Empty if the record is valid.
    """
    schema = json.loads(Path(schema_json).read_text())
    validator = jsonschema.Draft202012Validator(
        schema, format_checker=jsonschema.FormatChecker()
    )
    errors = [
        f"{'/'.join(str(p) for p in e.absolute_path) or '<root>'}: {e.message}"
        for e in validator.iter_errors(dict(record))
    ]
    if file_name is not None:
        expected = f"{record.get('team_abbr')}-{record.get('model_abbr')}"
        if Path(file_name).stem != expected:
            errors.append(f"File name {file_name} is not {expected}.yml.")
    return errors


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
