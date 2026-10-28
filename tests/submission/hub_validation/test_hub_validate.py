from __future__ import annotations

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
    assert diffs == {"invalid_noninteger_sample_id"}


def test_real_round2_submission_is_valid(hub: hv.Hub) -> None:
    p = (
        BUNDLE
        / "fixtures"
        / "real"
        / "model-output"
        / "OHTJHU-BM"
        / "2026-10-31-OHTJHU-BM.csv"
    )
    rep = hv.validate_submission(
        p, hub, metadata_dirs=METADATA_DIRS, skip_window=True
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
