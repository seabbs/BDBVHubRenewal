from __future__ import annotations

import datetime as dt
import json
from pathlib import Path

import pandas as pd
import pytest
import yaml

from bdbvhubrenewal.submission.hub import (
    HUB_COLUMNS,
    model_id,
    render_model_metadata,
    run_date,
    to_hub_table,
    write_model_metadata,
    write_submission,
)
from bdbvhubrenewal.submission.validate import (
    Hub,
    validate_metadata,
    validate_submission,
)

ROOT = Path(__file__).resolve().parents[2]
FIXTURES = Path(__file__).parent / "hub_validation" / "fixtures"
HUB = Hub.load(FIXTURES / "hub-snapshot")
"""The pinned hub config, in which the round of 2026-10-31 is open."""
METADATA = ROOT / "hub" / "model-metadata"
MID = "seabbs-renewal"

FORECAST_DATE = "2026-10-20"
"""On or before every reference date these tests use."""


def _failures(table: pd.DataFrame, root: Path) -> list[str]:
    """Write a table to the hub layout and list the checks it fails."""
    path = write_submission(table, root, MID)
    rep = validate_submission(
        path, HUB, metadata_dirs=[METADATA], skip_window=True
    )
    return [c.check for c in rep.failed]


def _table(samples: pd.DataFrame, ref: str, **kw: object) -> pd.DataFrame:
    kw.setdefault("forecast_date", FORECAST_DATE)
    return to_hub_table(samples, ref, **kw)  # type: ignore[arg-type]


def _samples(n: int = 100) -> pd.DataFrame:
    weeks = ["2026-10-24", "2026-10-31", "2026-11-07"]
    rows = [
        {"location": loc, "target_end_date": w, "sample": s, "value": 10.4}
        for loc in ("CD-IT", "CD")
        for s in range(1, n + 1)
        for w in weeks
    ]
    return pd.DataFrame(rows)


def test_to_hub_table_is_valid(tmp_path: Path) -> None:
    table = _table(_samples(), "2026-10-31")
    assert tuple(table.columns) == HUB_COLUMNS
    assert (table["value"] == 10).all()
    assert _failures(table, tmp_path) == []


def test_submission_path(tmp_path: Path) -> None:
    table = _table(_samples(), "2026-10-31")
    path = write_submission(table, tmp_path, "seabbs-randomwalk")
    assert path.name == "2026-10-31-seabbs-randomwalk.csv"
    assert path.parent.name == "seabbs-randomwalk"


def test_model_ids() -> None:
    assert model_id("randomwalk") == "seabbs-randomwalk"
    with pytest.raises(ValueError, match="letters"):
        model_id("bad-name")


def test_metadata_is_valid(tmp_path: Path) -> None:
    record = render_model_metadata(
        "randomwalk",
        model_name="RandomWalk baseline",
        methods="Short.",
        methods_long="Long.",
        data_inputs="Hub data.",
    )
    path = write_model_metadata(record, tmp_path)
    assert path.name == "seabbs-randomwalk.yml"
    assert validate_metadata(path, HUB).valid
    back = yaml.safe_load(path.read_text())
    assert "language model" not in back["model_details"]["methods_long"]
    # Sam's HPRU funding, in the schema's `team_funding` field.
    assert "NIHR207404" in back["team_funding"]


# Gaps found by the hub validation bundle -----------------------------------


def _provinces(n: int = 100, value: float = 10.4) -> pd.DataFrame:
    weeks = ["2026-10-24", "2026-10-31", "2026-11-07"]
    rows = [
        {"location": loc, "target_end_date": w, "sample": s, "value": value}
        for loc in ("CD-IT", "CD-NK")
        for s in range(1, n + 1)
        for w in weeks
    ]
    return pd.DataFrame(rows)


def test_validation_needs_same_ids_in_every_location(
    tmp_path: Path,
) -> None:
    table = _table(_provinces(), "2026-10-31")
    mask = (table["location"] == "CD-IT") & (table["output_type_id"] == 1)
    table.loc[mask, "output_type_id"] = 101
    assert _failures(table, tmp_path) == ["spl_compound_taskid_set"]


def test_model_abbreviations_at_most_14_characters() -> None:
    assert model_id("a" * 14) == f"seabbs-{'a' * 14}"
    with pytest.raises(ValueError, match="14"):
        model_id("a" * 15)


def test_writer_makes_national_the_sum_of_provinces() -> None:
    samples = _provinces(value=10.4)
    samples.loc[samples["location"] == "CD-NK", "value"] = 2.6
    national = samples.loc[samples["location"] == "CD-IT"].assign(
        location="CD", value=999.0
    )
    table = _table(pd.concat([samples, national]), "2026-10-31")
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
    table = _table(_provinces(), "2026-10-31", unassigned=unassigned)
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
        _table(change(_provinces()), "2026-10-31")


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
    meta = [
        FIXTURES / "constructed" / "model-metadata",
        FIXTURES / "real" / "model-metadata",
    ]
    rep = validate_submission(path, HUB, metadata_dirs=meta, skip_window=True)
    assert rep.valid is bool(fx["valid_py"]), rep.failed


# forecast_date ---------------------------------------------------------------


def test_forecast_date_column_is_written_after_reference_date() -> None:
    table = _table(_samples(), "2026-10-31")
    assert list(table.columns)[:2] == ["reference_date", "forecast_date"]
    assert (table["forecast_date"] == FORECAST_DATE).all()


def test_forecast_date_defaults_to_the_run_date_in_utc() -> None:
    table = to_hub_table(_samples(), "2026-10-31")
    assert (table["forecast_date"] == run_date().isoformat()).all()
    late = dt.datetime(
        2026, 10, 26, 23, 30, tzinfo=dt.timezone(dt.timedelta(hours=-5))
    )
    assert run_date(late) == dt.date(2026, 10, 27)
