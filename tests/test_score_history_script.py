from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "score_history.py"


@pytest.fixture(scope="module")
def sh():
    spec = importlib.util.spec_from_file_location("score_history", SCRIPT)
    assert spec is not None
    assert spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules["score_history"] = mod
    spec.loader.exec_module(mod)
    return mod


def test_source_sha_is_read_from_main_deploys_only(sh) -> None:
    sha = "0996814" + "0" * 33
    msg = f"Deploying to gh-pages from @ seabbs/BDBVHubRenewal@{sha} 🚀"
    assert sh.source_sha(msg) == sha
    assert sh.source_sha("Deploy preview for PR 102 🛫") is None
    assert sh.source_sha("Remove preview for PR 93 🛬") is None


def _summary() -> dict[str, object]:
    # Models: randomwalk (baseline), good, "as reported".
    # Places: CD (national), CD-IT and CD-NK (provinces).
    # Rows: model, origin, release, place, week, wis, cov50.
    return {
        "models": [
            {"name": "randomwalk"},
            {"name": "good"},
            {"name": "as reported"},
        ],
        "baseline": 0,
        "metrics": [
            {"key": "wis", "relative": True},
            {"key": "coverage_50", "relative": False},
        ],
        "origins": ["2026-10-24", "2026-10-31"],
        "releases": ["2026-11-01"],
        "weeks": ["2026-10-24"],
        "places": [
            {"id": "CD", "pool": "national"},
            {"id": "CD-IT", "pool": "provinces"},
            {"id": "CD-NK", "pool": "provinces"},
        ],
        "rows": [
            [0, 0, 0, 0, 0, 10.0, 1.0],
            [0, 1, 0, 0, 0, 30.0, 0.0],
            [1, 0, 0, 0, 0, 5.0, 1.0],
            [1, 1, 0, 0, 0, 15.0, 1.0],
            [0, 0, 0, 1, 0, 4.0, 1.0],
            [1, 0, 0, 1, 0, 2.0, 0.0],
            # No baseline forecast for CD-NK: left out.
            [1, 0, 0, 2, 0, 100.0, 0.0],
            [2, 0, 0, 0, 0, 1.0, 1.0],
        ],
    }


def test_summary_scores_match_on_baseline_forecasts(sh) -> None:
    out = sh.summary_scores(_summary()).set_index(["pool", "model"])
    assert set(out.index.get_level_values("model")) == {"randomwalk", "good"}
    nat = out.loc[("national", "good")]
    assert nat["n"] == 2
    assert nat["wis"] == pytest.approx(10.0)
    assert nat["rel_wis"] == pytest.approx(0.5)
    assert nat["coverage_50"] == pytest.approx(1.0)
    prov = out.loc[("provinces", "good")]
    assert prov["n"] == 1
    assert prov["rel_wis"] == pytest.approx(0.5)
    assert out.loc[("national", "randomwalk"), "rel_wis"] == 1.0
    assert "rel_coverage_50" not in out.columns


def test_history_flags_changes_in_what_is_scored(sh) -> None:
    base = _summary()
    later = _summary()
    later["origins"] = ["2026-10-31", "2026-11-07"]
    deploys = [
        ("a" * 40, "2026-11-04T10:00:00+00:00", "p1", "latest", base),
        ("a" * 40, "2026-11-04T11:00:00+00:00", "p2", "latest", base),
        ("b" * 40, "2026-11-05T10:00:00+00:00", "p3", "latest", later),
    ]
    out = sh.history(deploys)
    first = out.loc[(out["model"] == "good") & (out["pool"] == "national")]
    assert list(first["source_sha"].str[:1]) == ["a", "a", "b"]
    assert list(first["origins"])[-1] == "2026-10-31;2026-11-07"
    assert list(first["break"]) == [True, False, True]
    assert list(first["releases"]) == ["2026-11-01"] * 3
