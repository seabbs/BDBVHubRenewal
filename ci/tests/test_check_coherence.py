"""The data coherence check: national against the sum of provinces."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from typing import Any

from conftest import AS_OF, FIXTURES, write_json

from submission_ci.check_coherence import assess, main
from submission_ci.common import read_json

FAKE = FIXTURES / "fake_bdbv_forecasts.py"


def coherence(gaps: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    rows = gaps or []
    streams = {
        st: {
            "weeks_with_gap": sum(r["stream"] == st for r in rows),
            "max_abs_gap": max(
                (abs(r["gap"]) for r in rows if r["stream"] == st), default=0
            ),
            "total_gap": sum(r["gap"] for r in rows if r["stream"] == st),
        }
        for st in ("observed", "imputed", "total")
    }
    return {
        "schema_version": 1,
        "as_of": AS_OF,
        "national": "CD",
        "coherent": not rows,
        "streams": streams,
        "rows": rows,
        "handling": "The model fits the provinces and adds the gap as an "
        "unassigned national component.",
    }


GAP = {
    "target_end_date": "2026-10-24",
    "stream": "total",
    "national": 50,
    "provinces": 48,
    "gap": 2,
}


def test_coherent_data_pass() -> None:
    c = assess(coherence())
    assert c.status == "pass"
    assert "every week" in c.summary


def test_gap_warns() -> None:
    c = assess(coherence([GAP]))
    assert c.status == "warn"
    assert "1 week" in c.summary and "largest gap 2" in c.summary


def test_missing_file_is_na(tmp_path: Path) -> None:
    out = tmp_path / "o"
    assert (
        main(["--coherence", str(tmp_path / "none.json"), "--out", str(out)])
        == 0
    )
    data = read_json(out / "coherence-summary.json")
    assert data["status"] == "skip"
    assert "not checked" in data["summary"]


def test_cli_writes_report_with_gap_table(tmp_path: Path) -> None:
    src = write_json(tmp_path / "coherence.json", coherence([GAP]))
    out = tmp_path / "o"
    assert main(["--coherence", str(src), "--out", str(out)]) == 0
    data = read_json(out / "coherence-summary.json")
    assert data["status"] == "warn"
    md = (out / "coherence-summary.md").read_text()
    assert "## Data coherence: WARN" in md
    assert "<details>" in md and "| 2026-10-24 | total | 50 | 48 | 2 |" in md
    assert "unassigned national component" in md


def test_unreadable_file_warns(tmp_path: Path) -> None:
    bad = tmp_path / "coherence.json"
    bad.write_text("{not json")
    out = tmp_path / "o"
    main(["--coherence", str(bad), "--out", str(out)])
    assert read_json(out / "coherence-summary.json")["status"] == "warn"


def _fake(tmp_path: Path, gap: str) -> dict[str, Any]:
    out = tmp_path / "c.json"
    env = {**os.environ, "FAKE_COHERENCE_GAP": gap}
    cmd = [sys.executable, str(FAKE), "coherence", "--vintage", AS_OF]
    subprocess.run(
        [*cmd, "--out", str(out)],
        check=True,
        env=env,
    )
    return dict(read_json(out))


def test_fake_cli_coherence(tmp_path: Path) -> None:
    clean = _fake(tmp_path, "")
    assert clean["coherent"] and not clean["rows"]
    assert assess(clean).status == "pass"
    gapped = _fake(tmp_path, "2026-10-24:2")
    assert not gapped["coherent"]
    assert {r["stream"] for r in gapped["rows"]} == {"observed", "total"}
    assert gapped["streams"]["total"]["max_abs_gap"] == 2
    assert assess(gapped).status == "warn"
