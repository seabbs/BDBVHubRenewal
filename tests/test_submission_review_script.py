from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "submission_review.py"


@pytest.fixture(scope="module")
def sr():
    spec = importlib.util.spec_from_file_location("submission_review", SCRIPT)
    assert spec is not None
    assert spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules["submission_review"] = mod
    spec.loader.exec_module(mod)
    return mod


def test_a_checkpoint_without_held_out_losses_still_writes_the_page(
    sr, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    gate = tmp_path / "track" / "gate"
    gate.mkdir(parents=True)
    run = {
        "reference_date": "2026-11-21",
        "checkpoint": {
            "tag": "chronos-ft-x",
            "sha256": {"adapter_model.safetensors": "a" * 64},
        },
    }
    (gate / "provenance.json").write_text(json.dumps({"run": run}))
    (gate / "provenance.md").write_text("## Gate\n")
    monkeypatch.setattr(sr, "_settings", lambda _tag: {})
    out = tmp_path / "out"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "submission_review.py",
            "--track",
            str(tmp_path / "track"),
            "--observed",
            str(tmp_path / "none.csv"),
            "--out",
            str(out),
        ],
    )
    sr.main()
    page = (out / "index.md").read_text()
    assert "chronos-ft-x" in page
    assert "Held-out loss" not in page


def test_no_locations_draw_no_chart(sr, tmp_path: Path) -> None:
    observed = pd.DataFrame(
        columns=["location", "target_end_date", "observation"]
    )
    path = tmp_path / "fans.png"
    assert not sr.fan_charts({}, observed, [], path, "Zones")
    assert not path.exists()


def _quantiles(median: float) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "location": ["CD-IT"],
            "target_end_date": ["2026-11-21"],
            0.5: [median],
            0.975: [median * 2],
        }
    )


def test_flagged_takes_the_gate_thresholds_from_the_policy(
    sr, tmp_path: Path
) -> None:
    observed = pd.DataFrame(
        {
            "location": ["CD-IT"] * 4,
            "target_end_date": [
                "2026-10-31",
                "2026-11-07",
                "2026-11-14",
                "2026-11-21",
            ],
            "observation": [10, 10, 10, 10],
        }
    )
    # A median of 150 is 7.5 times the floor of 20.
    q = _quantiles(150.0)
    assert sr._flagged(q, observed, "2026-11-21") == ["CD-IT"]
    policy = tmp_path / "policy.toml"
    policy.write_text(
        "[checks]\n"
        "horizon_weeks = 3\n"
        "plausible_recent_weeks = 4\n"
        "plausible_floor = 20\n"
        "plausible_median_ratio = 10.0\n"
        "plausible_q975_ratio = 15.0\n"
        "plausible_low_ratio = 20.0\n"
    )
    assert sr._flagged(q, observed, "2026-11-21", policy) == []
