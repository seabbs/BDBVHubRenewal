from __future__ import annotations

import datetime as dt
import json
from pathlib import Path

import pandas as pd
import pytest
import yaml

from bdbv_hub_renewal.submission.hub import (
    HUB_COLUMNS,
    load_hub_round,
    model_id,
    render_model_metadata,
    to_hub_table,
    validate_model_id,
    validate_model_metadata,
    validate_submission,
    validate_submission_file,
    write_model_metadata,
    write_submission,
)


def _samples(n: int = 100) -> pd.DataFrame:
    weeks = ["2026-10-24", "2026-10-31", "2026-11-07"]
    rows = [
        {"location": loc, "target_end_date": w, "sample": s, "value": 10.4}
        for loc in ("CD-IT", "CD")
        for s in range(1, n + 1)
        for w in weeks
    ]
    return pd.DataFrame(rows)


@pytest.fixture
def hub_round(tasks_json: Path):  # type: ignore[no-untyped-def]
    return load_hub_round(tasks_json)


def test_load_hub_round(hub_round) -> None:  # type: ignore[no-untyped-def]
    assert hub_round.n_samples == 100
    assert hub_round.compound_taskid_set == (
        "reference_date",
        "location",
        "target",
    )
    assert "CD-IT" in hub_round.locations
    assert (hub_round.window_start, hub_round.window_end) == (-6, -4)


def test_to_hub_table_is_valid(hub_round) -> None:  # type: ignore[no-untyped-def]
    table = to_hub_table(_samples(), "2026-10-31")
    assert tuple(table.columns) == HUB_COLUMNS
    assert (table["value"] == 10).all()
    errors = validate_submission(
        table,
        hub_round,
        file_name="2026-10-31-epiforecasts-persistence.csv",
        mid="epiforecasts-persistence",
    )
    assert errors == []


def test_round_trip_through_csv(hub_round, tmp_path: Path) -> None:  # type: ignore[no-untyped-def]
    table = to_hub_table(_samples(), "2026-10-31")
    path = write_submission(table, tmp_path, "epiforecasts-persistence")
    assert path.name == "2026-10-31-epiforecasts-persistence.csv"
    assert path.parent.name == "epiforecasts-persistence"
    back = pd.read_csv(path)
    assert validate_submission(back, hub_round) == []


@pytest.mark.parametrize(
    ("change", "message"),
    [
        (lambda t: t.iloc[1:], "exactly 100 samples"),
        (lambda t: t.assign(value=-1), "non-negative"),
        (lambda t: t.assign(location="CD-XX"), "Unknown locations"),
        (lambda t: t.assign(target_end_date="2026-10-30"), "not Saturdays"),
        (lambda t: t.assign(target_end_date="2026-11-28"), "outside"),
        (lambda t: t.assign(output_type="quantile"), "Output type"),
        (lambda t: t.assign(reference_date="2026-10-24"), "not an open round"),
        (lambda t: t.drop(columns="target"), "Columns"),
        (lambda t: t.assign(output_type_id=1.5), "integers"),
    ],
)
def test_validation_catches_problems(hub_round, change, message: str) -> None:  # type: ignore[no-untyped-def]
    table = change(to_hub_table(_samples(), "2026-10-31"))
    errors = validate_submission(table, hub_round)
    assert any(message in e for e in errors), errors


def test_validation_catches_ragged_trajectories(hub_round) -> None:  # type: ignore[no-untyped-def]
    table = to_hub_table(_samples(), "2026-10-31")
    # Move one sample's week so samples cover different week sets.
    mask = (
        (table["location"] == "CD-IT")
        & (table["output_type_id"] == 1)
        & (table["target_end_date"] == "2026-11-07")
    )
    table.loc[mask, "target_end_date"] = "2026-11-14"
    errors = validate_submission(table, hub_round)
    assert any("same weeks" in e for e in errors)


def test_validation_window(hub_round) -> None:  # type: ignore[no-untyped-def]
    table = to_hub_table(_samples(), "2026-10-31")
    tz = dt.UTC
    inside = dt.datetime(2026, 10, 26, 12, tzinfo=tz)
    late = dt.datetime(2026, 10, 28, 12, tzinfo=tz)
    assert validate_submission(table, hub_round, submitted_at=inside) == []
    assert validate_submission(table, hub_round, submitted_at=late)


def test_backtest_dates_skip_round_checks(hub_round) -> None:  # type: ignore[no-untyped-def]
    samples = _samples().assign(
        target_end_date=lambda d: d["target_end_date"].map(
            {
                "2026-10-24": "2026-08-15",
                "2026-10-31": "2026-08-22",
                "2026-11-07": "2026-08-29",
            }
        )
    )
    table = to_hub_table(samples, "2026-08-22")
    assert validate_submission(table, hub_round, check_round=False) == []


def test_model_ids() -> None:
    assert model_id("persistence") == "epiforecasts-persistence"
    with pytest.raises(ValueError, match="letters"):
        model_id("bad-name")
    assert validate_model_id("epiforecasts-renewal") == []
    assert validate_model_id("a-b-c")
    assert validate_model_id("epiforecasts-waytoolongmodelname")


def test_metadata_is_valid(schema_json: Path, tmp_path: Path) -> None:
    record = render_model_metadata(
        "persistence",
        model_name="Persistence baseline",
        methods="Short.",
        methods_long="Long.",
        data_inputs="Hub data.",
    )
    path = write_model_metadata(record, tmp_path)
    assert path.name == "epiforecasts-persistence.yml"
    back = yaml.safe_load(path.read_text())
    assert validate_model_metadata(back, schema_json, path.name) == []
    assert "language model" in back["model_details"]["methods_long"]
    bad = {**back, "license": "MIT"}
    assert validate_model_metadata(bad, schema_json)
    assert validate_model_metadata(back, schema_json, "wrong.yml")


# Gaps found by the hub validation bundle -----------------------------------

FIXTURES = Path(__file__).parent / "hub_validation" / "fixtures"
SNAPSHOT_TASKS = FIXTURES / "hub-snapshot" / "hub-config" / "tasks.json"


def _provinces(n: int = 100, value: float = 10.4) -> pd.DataFrame:
    weeks = ["2026-10-24", "2026-10-31", "2026-11-07"]
    rows = [
        {"location": loc, "target_end_date": w, "sample": s, "value": value}
        for loc in ("CD-IT", "CD-NK")
        for s in range(1, n + 1)
        for w in weeks
    ]
    return pd.DataFrame(rows)


def test_validation_needs_same_ids_in_every_location(hub_round) -> None:  # type: ignore[no-untyped-def]
    table = to_hub_table(_provinces(), "2026-10-31")
    mask = (table["location"] == "CD-IT") & (table["output_type_id"] == 1)
    table.loc[mask, "output_type_id"] = 101
    errors = validate_submission(table, hub_round)
    assert any("same sample ids" in e for e in errors), errors


def test_unique_ids_need_same_weeks_everywhere(hub_round) -> None:  # type: ignore[no-untyped-def]
    table = to_hub_table(_provinces(), "2026-10-31")
    offset = (table["location"] == "CD-NK") * 100
    offset += (table["location"] == "CD") * 200
    table["output_type_id"] += offset
    assert validate_submission(table, hub_round) == []
    ragged = table.loc[
        ~(
            (table["location"] == "CD-NK")
            & (table["target_end_date"] == "2026-11-07")
        )
    ]
    errors = validate_submission(ragged, hub_round)
    assert any("same weeks in every location" in e for e in errors), errors


def test_reused_ids_allow_different_weeks_by_location(hub_round) -> None:  # type: ignore[no-untyped-def]
    table = to_hub_table(_provinces(), "2026-10-31")
    table = table.loc[
        ~(
            (table["location"] == "CD-NK")
            & (table["target_end_date"] == "2026-11-07")
        )
    ]
    assert validate_submission(table, hub_round) == []


def test_validation_checks_directory_and_metadata(
    hub_round,  # type: ignore[no-untyped-def]
    tmp_path: Path,
) -> None:
    table = to_hub_table(_provinces(), "2026-10-31")
    name = "2026-10-31-epiforecasts-persistence.csv"
    mid = "epiforecasts-persistence"
    errors = validate_submission(
        table, hub_round, file_name=name, mid=mid, dir_name="epiforecasts-x"
    )
    assert any("directory" in e for e in errors), errors
    errors = validate_submission(
        table, hub_round, file_name=name, mid=mid, metadata_dir=tmp_path
    )
    assert any("metadata" in e.lower() for e in errors), errors
    (tmp_path / f"{mid}.yml").write_text("team_abbr: epiforecasts\n")
    assert (
        validate_submission(
            table,
            hub_round,
            file_name=name,
            mid=mid,
            dir_name=mid,
            metadata_dir=tmp_path,
        )
        == []
    )


def test_model_abbreviations_at_most_14_characters() -> None:
    assert model_id("a" * 14) == f"epiforecasts-{'a' * 14}"
    with pytest.raises(ValueError, match="14"):
        model_id("a" * 15)
    assert validate_model_id(f"epiforecasts-{'a' * 15}")


def test_writer_makes_national_the_sum_of_provinces() -> None:
    samples = _provinces(value=10.4)
    samples.loc[samples["location"] == "CD-NK", "value"] = 2.6
    national = samples.loc[samples["location"] == "CD-IT"].assign(
        location="CD", value=999.0
    )
    table = to_hub_table(pd.concat([samples, national]), "2026-10-31")
    wide = table.pivot_table(
        index=["output_type_id", "target_end_date"],
        columns="location",
        values="value",
    )
    assert (wide["CD"] == wide["CD-IT"] + wide["CD-NK"]).all()
    assert (wide["CD"] == 13).all()
    assert table["output_type_id"].dtype == "int64"
    assert sorted(table["output_type_id"].unique()) == list(range(1, 101))


def test_writer_adds_an_unassigned_component() -> None:
    unassigned = pd.DataFrame(
        {
            "target_end_date": "2026-10-31",
            "sample": range(1, 101),
            "value": 2.0,
        }
    )
    table = to_hub_table(_provinces(), "2026-10-31", unassigned=unassigned)
    cd = table.loc[table["location"] == "CD"].set_index(
        ["output_type_id", "target_end_date"]
    )["value"]
    assert (cd.xs("2026-10-31", level=1) == 22).all()
    assert (cd.xs("2026-10-24", level=1) == 20).all()


@pytest.mark.parametrize(
    "change",
    [
        lambda s: s.loc[~((s["location"] == "CD-NK") & (s["sample"] == 5))],
        lambda s: s.loc[
            ~(
                (s["location"] == "CD-NK")
                & (s["target_end_date"] == "2026-11-07")
            )
        ],
        lambda s: s.assign(sample=s["sample"] + 1),
    ],
    ids=["missing-sample", "missing-week", "ids-not-1-to-100"],
)
def test_writer_refuses_inconsistent_samples(change) -> None:  # type: ignore[no-untyped-def]
    with pytest.raises(ValueError, match=r"sample|week"):
        to_hub_table(change(_provinces()), "2026-10-31")


def _round2_fixtures() -> list[dict[str, object]]:
    manifest = json.loads((FIXTURES / "manifest.json").read_text())
    return [
        f
        for f in manifest["fixtures"]
        if f["kind"] == "submission" and "2026-10-31" in f["path"]
    ]


@pytest.mark.parametrize("fx", _round2_fixtures(), ids=lambda f: str(f["case"]))
def test_in_package_check_agrees_with_bundle_fixtures(fx) -> None:  # type: ignore[no-untyped-def]
    path = FIXTURES.parent / str(fx["path"])
    hub_round = load_hub_round(SNAPSHOT_TASKS)
    meta = [
        FIXTURES / "constructed" / "model-metadata",
        FIXTURES / "real" / "model-metadata",
    ]
    errors = validate_submission_file(path, hub_round, metadata_dirs=meta)
    assert (errors == []) is bool(fx["valid_py"]), errors
