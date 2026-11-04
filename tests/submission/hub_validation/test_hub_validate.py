from __future__ import annotations

import csv
import json
import shutil
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest
from hv_fixtures import BUNDLE, FIXTURES, METADATA_DIRS, SNAPSHOT

from bdbvhubrenewal.submission import validate as hv

FILE_31 = "2026-10-31-testteam-testmodel.csv"


def fixture_path(case: str, name: str = FILE_31) -> Path:
    return (
        BUNDLE / "fixtures" / "constructed" / case / "testteam-testmodel" / name
    )


def failures(rep: hv.Report) -> list[str]:
    return sorted(c.check for c in rep.failed)


@pytest.mark.parametrize("fx", FIXTURES, ids=lambda f: f["path"])
def test_fixture_matches_manifest(fx: dict[str, Any], hub: hv.Hub) -> None:
    rep = hv.validate_file(
        BUNDLE / fx["path"],
        hub,
        metadata_dirs=METADATA_DIRS,
        skip_window=True,
    )
    assert failures(rep) == fx["py_failures"], [
        (c.check, c.message) for c in rep.failed
    ]


def test_each_invalid_fixture_has_one_distinct_cause() -> None:
    """Invalid constructed fixtures each fail for a stated reason."""
    inv = [
        f
        for f in FIXTURES
        if f["case"].startswith("invalid_") and "source" not in f
    ]
    assert len(inv) >= 20
    for f in inv:
        assert f["py_failures"], f["case"]


def test_python_and_r_verdicts_agree_except_documented() -> None:
    diffs = {f["case"] for f in FIXTURES if f["valid_py"] != f["valid_r"]}
    # The R wrapper drops `forecast_date` as the hub's workflow does, so the
    # column's rules are checked in Python only.
    assert diffs == {
        "invalid_noninteger_sample_id",
        "invalid_forecast_date_missing",
        "invalid_forecast_date_format",
        "invalid_forecast_date_after_reference",
        "invalid_forecast_dates_two",
        "2026-10-31-OHTJHU-BM.csv",
    }


def test_real_round2_submission_is_valid(hub: hv.Hub, tmp_path: Path) -> None:
    p = (
        BUNDLE
        / "fixtures"
        / "real"
        / "model-output"
        / "OHTJHU-BM"
        / "2026-10-31-OHTJHU-BM.csv"
    )
    # The file predates `forecast_date`, so the hub would now reject it.
    rep = hv.validate_submission(
        p, hub, metadata_dirs=METADATA_DIRS, skip_window=True
    )
    assert failures(rep) == ["forecast_date_column"]
    # With the column added, every other check passes.
    lines = p.read_text().splitlines()
    dated = [
        lines[0].replace("reference_date,", "reference_date,forecast_date,")
    ]
    dated += [ln.replace(",", ",2026-10-26,", 1) for ln in lines[1:]]
    q = tmp_path / "OHTJHU-BM" / p.name
    q.parent.mkdir()
    q.write_text("\n".join(dated) + "\n")
    rep = hv.validate_submission(
        q, hub, metadata_dirs=METADATA_DIRS, skip_window=True
    )
    assert rep.valid
    assert any(c.check == "spl_n" and c.status == "pass" for c in rep.checks)


# Submission window ----------------------------------------------------------


@pytest.mark.parametrize(
    ("now", "ok"),
    [
        # Window for 2026-10-31 is 2026-10-25 00:00 to 2026-10-27 23:59:59
        # Africa/Kinshasa (UTC+1).
        ("2026-10-25T00:00:00+01:00", True),
        ("2026-10-24T23:30:00+00:00", True),  # 00:30 Sunday in Kinshasa
        ("2026-10-24T22:59:59+00:00", False),  # 23:59:59 Saturday WAT
        ("2026-10-27T23:59:59+01:00", True),
        ("2026-10-27T23:30:00+00:00", False),  # 00:30 Wednesday WAT
        ("2026-10-31T12:00:00+01:00", False),  # on the reference date
    ],
)
def test_submission_window(now: str, ok: bool, hub: hv.Hub) -> None:
    rep = hv.validate_submission(
        fixture_path("valid_basic"),
        hub,
        metadata_dirs=METADATA_DIRS,
        now=datetime.fromisoformat(now),
    )
    st = next(c for c in rep.checks if c.check == "submission_time")
    assert (st.status == "pass") is ok, st.message


def test_window_bounds(hub: hv.Hub) -> None:
    rnd = hub.find_round("2026-10-31")
    assert rnd is not None
    start, end = hub.submission_window(rnd, "2026-10-31")
    assert start.astimezone(UTC).isoformat() == ("2026-10-24T23:00:00+00:00")
    assert end.astimezone(UTC).isoformat() == ("2026-10-27T22:59:59+00:00")


# forecast_date ---------------------------------------------------------------


def _write(path: Path, header: list[str], rows: list[dict[str, str]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=header, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    return path


def _read(case: str = "valid_basic") -> tuple[list[str], list[dict[str, str]]]:
    return hv.read_rows(fixture_path(case))


def _check(rep: hv.Report, name: str) -> hv.Check:
    return next(c for c in rep.checks if c.check == name)


def test_forecast_date_column_and_value_are_checked(hub: hv.Hub) -> None:
    rep = hv.validate_submission(
        fixture_path("valid_basic"),
        hub,
        metadata_dirs=METADATA_DIRS,
        skip_window=True,
    )
    assert rep.valid
    assert _check(rep, "forecast_date").status == "pass"


@pytest.mark.parametrize(
    "bad", ["", "NA", "20/10/2026", "2026-10-20T00:00", "2026-02-30"]
)
def test_forecast_date_must_parse(
    bad: str, hub: hv.Hub, tmp_path: Path
) -> None:
    cols, rows = _read()
    rows[3]["forecast_date"] = bad
    p = _write(tmp_path / "testteam-testmodel" / FILE_31, cols, rows)
    rep = hv.validate_submission(
        p, hub, metadata_dirs=METADATA_DIRS, skip_window=True
    )
    assert failures(rep) == ["forecast_date_valid"]


def test_forecast_date_may_equal_but_not_follow_reference(
    hub: hv.Hub, tmp_path: Path
) -> None:
    cols, rows = _read()
    on_day = [dict(r, forecast_date="2026-10-31") for r in rows]
    p = _write(tmp_path / "testteam-testmodel" / FILE_31, cols, on_day)
    rep = hv.validate_submission(
        p, hub, metadata_dirs=METADATA_DIRS, skip_window=True
    )
    assert rep.valid
    after = [dict(r, forecast_date="2026-11-01") for r in rows]
    p = _write(tmp_path / "other" / "testteam-testmodel" / FILE_31, cols, after)
    rep = hv.validate_submission(
        p, hub, metadata_dirs=METADATA_DIRS, skip_window=True
    )
    assert failures(rep) == ["forecast_date_le_reference"]


def _resubmission(
    tmp_path: Path, new_date: str = "2026-10-22", **changes: Any
) -> tuple[Path, Path]:
    """The hub's file (one date) and a file that appends a second."""
    cols, rows = _read()
    old = _write(tmp_path / "hub" / FILE_31, cols, rows)
    added = [dict(r, forecast_date=new_date) for r in rows[:200]]
    appended = rows + added
    if changes.get("edit_old"):
        appended = [dict(appended[0], value="999"), *appended[1:]]
    if changes.get("drop_old"):
        appended = appended[1:]
    if changes.get("two_new"):
        appended += [dict(r, forecast_date="2026-10-23") for r in rows[:5]]
    new = _write(
        tmp_path / "pr" / "testteam-testmodel" / FILE_31, cols, appended
    )
    return old, new


def test_append_keeps_original_rows_and_adds_one_date(
    hub: hv.Hub, tmp_path: Path
) -> None:
    old, new = _resubmission(tmp_path)
    rep = hv.validate_submission(
        new,
        hub,
        metadata_dirs=METADATA_DIRS,
        skip_window=True,
        previous=old,
    )
    # The new forecast is a part file, so later sample checks may fail.
    assert _check(rep, "forecast_date").status == "pass"


def test_resubmission_with_all_new_rows_is_valid(
    hub: hv.Hub, tmp_path: Path
) -> None:
    cols, rows = _read()
    old = _write(tmp_path / "hub" / FILE_31, cols, rows)
    new_rows = [dict(r, forecast_date="2026-10-22") for r in rows]
    new = _write(
        tmp_path / "pr" / "testteam-testmodel" / FILE_31, cols, rows + new_rows
    )
    rep = hv.validate_submission(
        new,
        hub,
        metadata_dirs=METADATA_DIRS,
        skip_window=True,
        previous=old,
    )
    assert rep.valid, [(c.check, c.message) for c in rep.failed]


@pytest.mark.parametrize(
    ("change", "check"),
    [
        ({"edit_old": True}, "forecast_date_preserved"),
        ({"drop_old": True}, "forecast_date_preserved"),
        ({"two_new": True}, "forecast_date_one_new"),
    ],
)
def test_resubmission_must_append_only(
    change: dict[str, bool], check: str, hub: hv.Hub, tmp_path: Path
) -> None:
    old, new = _resubmission(tmp_path, **change)
    rep = hv.validate_submission(
        new,
        hub,
        metadata_dirs=METADATA_DIRS,
        skip_window=True,
        previous=old,
    )
    assert failures(rep) == [check]


def test_resubmission_needs_a_new_date(hub: hv.Hub, tmp_path: Path) -> None:
    cols, rows = _read()
    old = _write(tmp_path / "hub" / FILE_31, cols, rows)
    same = _write(tmp_path / "pr" / "testteam-testmodel" / FILE_31, cols, rows)
    rep = hv.validate_submission(
        same, hub, metadata_dirs=METADATA_DIRS, skip_window=True, previous=old
    )
    assert failures(rep) == ["forecast_date_one_new"]
    extra = [*rows, dict(rows[0])]
    more = _write(
        tmp_path / "pr2" / "testteam-testmodel" / FILE_31, cols, extra
    )
    rep = hv.validate_submission(
        more, hub, metadata_dirs=METADATA_DIRS, skip_window=True, previous=old
    )
    assert failures(rep) == ["forecast_date_preserved"]


def test_hub_copy_without_the_column_cannot_be_appended(
    hub: hv.Hub, tmp_path: Path
) -> None:
    cols, rows = _read()
    old_cols = [c for c in cols if c != "forecast_date"]
    old = _write(tmp_path / "hub" / FILE_31, old_cols, rows)
    new = _write(tmp_path / "pr" / "testteam-testmodel" / FILE_31, cols, rows)
    rep = hv.validate_submission(
        new, hub, metadata_dirs=METADATA_DIRS, skip_window=True, previous=old
    )
    assert failures(rep) == ["forecast_date_preserved"]


@pytest.mark.parametrize(
    ("now", "ok"),
    [
        ("2026-10-19T12:00:00+01:00", True),  # a day before
        ("2026-10-20T09:00:00+01:00", True),
        ("2026-10-21T23:00:00+01:00", True),  # a day after
        ("2026-10-22T00:30:00+01:00", False),
        ("2026-10-18T12:00:00+01:00", False),
    ],
)
def test_new_forecast_date_must_be_recent(
    now: str, ok: bool, hub: hv.Hub
) -> None:
    rep = hv.validate_submission(
        fixture_path("valid_basic"),
        hub,
        metadata_dirs=METADATA_DIRS,
        now=datetime.fromisoformat(now),
        skip_window=False,
    )
    passed = "forecast_date_recent" not in {c.check for c in rep.failed}
    assert passed is ok


def test_retained_dates_are_not_checked_for_recency(
    hub: hv.Hub, tmp_path: Path
) -> None:
    old, new = _resubmission(tmp_path)
    rep = hv.validate_submission(
        new,
        hub,
        metadata_dirs=METADATA_DIRS,
        now=datetime.fromisoformat("2026-10-22T10:00:00+01:00"),
        previous=old,
    )
    assert "forecast_date_recent" not in {c.check for c in rep.failed}


# Config is read at run time --------------------------------------------------


@pytest.fixture
def advanced_hub(tmp_path: Path) -> hv.Hub:
    """Snapshot with the next weekly dates added, as the hub's bot does."""
    shutil.copytree(SNAPSHOT / "hub-config", tmp_path / "hub-config")
    p = tmp_path / "hub-config" / "tasks.json"
    cfg = json.loads(p.read_text())
    tids = cfg["rounds"][1]["model_tasks"][0]["task_ids"]
    tids["reference_date"]["optional"].append("2026-11-07")
    tids["target_end_date"]["optional"].append("2026-11-28")
    p.write_text(json.dumps(cfg))
    return hv.Hub.load(tmp_path)


def test_new_reference_date_is_picked_up(
    advanced_hub: hv.Hub, hub: hv.Hub
) -> None:
    p = (
        BUNDLE
        / "fixtures"
        / "constructed"
        / "invalid_unknown_round"
        / "testteam-testmodel"
        / "2026-11-07-testteam-testmodel.csv"
    )
    old = hv.validate_submission(
        p, hub, metadata_dirs=METADATA_DIRS, skip_window=True
    )
    new = hv.validate_submission(
        p, advanced_hub, metadata_dirs=METADATA_DIRS, skip_window=True
    )
    assert failures(old) == ["round_id_valid"]
    assert new.valid


def test_readme_horizon_warns_when_config_allows_more(
    advanced_hub: hv.Hub,
) -> None:
    """A week the config accepts but the README rule does not: warn only."""
    rep = hv.validate_submission(
        fixture_path("invalid_ted_beyond_horizon"),
        advanced_hub,
        metadata_dirs=METADATA_DIRS,
        skip_window=True,
    )
    assert rep.valid
    w = [c for c in rep.checks if c.check == "readme_horizon"]
    assert w
    assert w[0].status == "warn"


# Units ----------------------------------------------------------------------


@pytest.mark.parametrize(
    ("a", "b"),
    [
        ("1", "1.0"),
        (0.025, "0.025"),
        ("2026-10-31", "2026-10-31"),
        (100, "100"),
    ],
)
def test_norm_equal(a: Any, b: Any) -> None:
    assert hv.norm(a) == hv.norm(b)


@pytest.mark.parametrize(
    ("name", "ok"),
    [
        ("2026-10-31-team-model.csv", True),
        ("2026-10-31-team_a-model_b.csv", True),
        ("2026-10-31_team-model.csv", False),
        ("2026-10-31-team-model-extra.csv", False),
        ("team-model.csv", False),
    ],
)
def test_parse_file_name(name: str, ok: bool) -> None:
    assert (hv.parse_file_name(name) is not None) is ok


def test_hub_reads_rounds(hub: hv.Hub) -> None:
    rnd = hub.find_round("2026-10-31")
    assert rnd is not None
    assert rnd.name == "BDBV 2026 nowcasting and forecasting"
    assert rnd.task_id_names == [
        "reference_date",
        "location",
        "target",
        "target_end_date",
    ]
    ot = rnd.model_tasks[0].output_types["sample"]
    assert ot.sample_params is not None
    assert ot.sample_params["min_samples_per_task"] == 100
    assert hub.find_round("2026-10-12") is not None
    assert hub.find_round("2026-10-12") is not rnd


def test_schema_misplaced_additional_properties(hub: hv.Hub) -> None:
    """The hub schema puts additionalProperties inside contributor
    ``properties``, so extra contributor keys are allowed."""
    assert hub.metadata_schema is not None
    contrib = hub.metadata_schema["properties"]["model_contributors"]["items"]
    assert "additionalProperties" in contrib["properties"]
    assert hv._schema_errors({"name": "x", "github": "y"}, contrib) == []


def test_cli_exit_codes(hub_path: Path, tmp_path: Path) -> None:
    md = [a for d in METADATA_DIRS for a in ("--metadata-dir", str(d))]
    out = tmp_path / "res.json"
    ok = hv.main(
        [
            "--hub",
            str(hub_path),
            *md,
            "--skip-window",
            "--json",
            str(out),
            str(fixture_path("valid_basic")),
        ]
    )
    assert ok == 0
    assert json.loads(out.read_text())
    bad = hv.main(
        [
            "--hub",
            str(hub_path),
            *md,
            "--skip-window",
            str(fixture_path("invalid_samples_99")),
        ]
    )
    assert bad == 1
