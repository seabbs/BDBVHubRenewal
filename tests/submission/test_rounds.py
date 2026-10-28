from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from bdbv_hub_renewal.submission.rounds import latest_run, make_round_pages

REF = "2026-10-31"


def _ledger(root: Path) -> Path:
    ledger = root / "ledger"
    state = {
        "schema_version": 1,
        "rounds": {
            REF: {
                "status": "submitted",
                "produced_by": "main model (stand-in)",
                "as_of": "2026-10-24",
                "pr_url": "https://github.com/o/r/pull/1",
                "run": "https://github.com/o/r/actions/runs/9",
                "updated": "2026-10-27T18:14:00+00:00",
            },
            "2026-10-24": {"status": "skipped", "produced_by": ""},
        },
    }
    ledger.mkdir(parents=True)
    (ledger / "state.json").write_text(json.dumps(state))
    for run in ("8", "10"):
        d = ledger / "archive" / REF / run
        d.mkdir(parents=True)
        (d / "nowcast.png").write_bytes(f"png {run}".encode())
    d = ledger / "archive" / REF / "10"
    (d / "provenance.json").write_text(
        json.dumps(
            {
                "selected": "main",
                "stand_in": "the `main` config runs the persistence baseline",
                "reason": "The main model (stand-in) cleared every check.",
                "attempts": [
                    {"id": "main", "label": "main model", "status": "pass"},
                    {"id": "retry-1", "label": "r", "status": "not needed"},
                ],
            }
        )
    )
    (d / "run.json").write_text(
        json.dumps({"model": "persistence", "config": "main"})
    )
    rows = [
        {"location": loc, "target_end_date": "2026-11-07", "value": v}
        for loc, v in (("CD", 10), ("CD", 20), ("CD-IT", 5))
    ]
    pd.DataFrame(rows).to_csv(d / f"{REF}-epiforecasts-persistence.csv")
    return ledger


def test_latest_run_is_the_largest_id(tmp_path: Path) -> None:
    ledger = _ledger(tmp_path)
    run = latest_run(ledger / "archive" / REF)
    assert run is not None
    assert run.name == "10"


def test_round_pages(tmp_path: Path) -> None:
    docs, gen = tmp_path / "docs", tmp_path / "gen"
    done = make_round_pages(_ledger(tmp_path), docs, gen)
    assert done == [REF, "2026-10-24"]
    rounds = docs / "submission" / "rounds"
    page = (rounds / f"{REF}.md").read_text()
    assert page.startswith(f"# Round {REF}")
    assert f"![Nowcast and forecast submitted for {REF}]({REF}.png)" in page
    assert (rounds / f"{REF}.png").read_bytes() == b"png 10"
    assert "<https://github.com/o/r/pull/1>" in page
    assert "persistence (`main`)" in page
    assert '!!! note "Stand-in"' in page
    assert "| 2026-11-07 | 15 |" in page
    assert "| main model | submitted |  |" in page
    assert "| r |" not in page
    # A round without an archive still gets a page, without a plot.
    assert "![" not in (rounds / "2026-10-24.md").read_text()
    index = (rounds / "index.md").read_text()
    assert f"[{REF}]({REF}.md)" in index
    snippet = (gen / "latest-submission.md").read_text()
    assert f"(submission/rounds/{REF}.md)" in snippet


def test_no_ledger_yet(tmp_path: Path) -> None:
    docs, gen = tmp_path / "docs", tmp_path / "gen"
    assert make_round_pages(None, docs, gen) == []
    assert (gen / "latest-submission.md").read_text() == ""
    index = (docs / "submission/rounds/index.md").read_text()
    assert "yet" not in index
    assert make_round_pages(tmp_path / "missing", docs, gen) == []
