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
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

import jsonschema
import pandas as pd
import yaml

from .dates import (
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
    "TEAM_ABBR",
    "HubRound",
    "load_hub_round",
    "model_id",
    "render_model_metadata",
    "submission_path",
    "to_hub_table",
    "validate_model_id",
    "validate_model_metadata",
    "validate_submission",
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

TEAM_ABBR: Final = "epiforecasts"
"""Team abbreviation, shared with earlier epiforecasts hub models."""

_ID_PART = re.compile(r"^[A-Za-z0-9_]{1,15}$")


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
        If either part is not 1 to 15 letters, digits or underscores.
    """
    for part in (team_abbr, model_abbr):
        if not _ID_PART.match(part):
            raise ValueError(
                f"{part!r} must be 1 to 15 letters, digits or underscores."
            )
    return f"{team_abbr}-{model_abbr}"


def validate_model_id(value: str) -> list[str]:
    """Check a model id against the hub's naming rules.

    Parameters
    ----------
    value : str
        A model id such as `"epiforecasts-persistence"`.

    Returns
    -------
    list of str
        Problems found. Empty if the id is valid.
    """
    parts = value.split("-")
    if len(parts) != 2:
        return [f"Model id {value!r} must be `team-model` with one hyphen."]
    return [
        f"{part!r} in {value!r} must be 1 to 15 letters, digits or underscores."
        for part in parts
        if not _ID_PART.match(part)
    ]


def to_hub_table(
    samples: pd.DataFrame,
    reference_date: dt.date | str,
    target: str = "wk inc confirmed cases",
) -> pd.DataFrame:
    """Turn forecast samples into the hub's submission table.

    Parameters
    ----------
    samples : pandas.DataFrame
        Columns `location`, `target_end_date`, `sample` (an integer
        index, shared across weeks within a location) and `value`.
    reference_date : datetime.date or str
        The round's reference date.
    target : str
        The target name.

    Returns
    -------
    pandas.DataFrame
        Columns `HUB_COLUMNS`, sorted by location, sample and week.
        Values are rounded to whole cases because the target is a count.
    """
    ref = as_date(reference_date).isoformat()
    out = pd.DataFrame(
        {
            "reference_date": ref,
            "target": target,
            "target_end_date": samples["target_end_date"].astype(str),
            "location": samples["location"].astype(str),
            "output_type": OUTPUT_TYPE,
            "output_type_id": samples["sample"].astype("int64"),
            "value": samples["value"].clip(lower=0).round().astype("int64"),
        }
    )
    return out.sort_values(
        ["location", "output_type_id", "target_end_date"]
    ).reset_index(drop=True)[list(HUB_COLUMNS)]


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
        The model id, such as `"epiforecasts-persistence"`.

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


def validate_submission(
    table: pd.DataFrame,
    hub_round: HubRound,
    *,
    file_name: str | None = None,
    mid: str | None = None,
    check_round: bool = True,
    submitted_at: dt.datetime | None = None,
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

    if submitted_at is not None:
        start, end = submission_window(ref)
        when = submitted_at.astimezone(HUB_TIMEZONE)
        if not (start <= when <= end):
            errors.append(
                f"Submission time {when.isoformat()} is outside the window "
                f"{start.isoformat()} to {end.isoformat()}."
            )
    return errors


# --------------------------------------------------------------------------
# Model metadata
# --------------------------------------------------------------------------

METADATA_DEFAULTS: Final[dict[str, Any]] = {
    "team_name": "EpiForecasts",
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
    "repo_url": "https://github.com/seabbs/bdbv-hub-baseline-forecasts",
    "website_url": "https://seabbs.github.io/bdbv-hub-baseline-forecasts/",
    "ensemble_of_hub_models": False,
}
"""Fields shared by every model's metadata. Contributors are to confirm."""

_LLM_NOTE: Final = (
    " Model code was drafted with the assistance of a language model and "
    "is under review by the named authors."
)


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
        Model abbreviation (the part after `epiforecasts-`).
    model_name : str
        Full model name, under 50 characters.
    methods : str
        Short description, at most 200 characters.
    methods_long : str
        Full description. A note that the code was drafted with an LLM is
        appended.
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
            "methods_long": methods_long + _LLM_NOTE,
        },
    }
    if model_version is not None:
        record["model_version"] = model_version
    for key in ("website_url", "repo_url", "ensemble_of_hub_models"):
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
