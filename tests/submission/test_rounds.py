from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from bdbvhubrenewal.cli import main
from bdbvhubrenewal.submission.rounds import (
    freeze_round,
    latest_run,
    make_round_pages,
)

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
    pd.DataFrame(rows).to_csv(d / f"{REF}-seabbs-persistence.csv")
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


def test_freeze_round_writes_the_page_once(tmp_path: Path) -> None:
    ledger = _ledger(tmp_path)
    assert freeze_round(ledger, REF)
    frozen = ledger / "pages" / f"{REF}.md"
    page = frozen.read_text()
    assert page.startswith(f"# Round {REF}")
    assert (ledger / "pages" / f"{REF}.png").read_bytes() == b"png 10"
    # A later run of the round, or new rendering code, never rewrites it.
    (ledger / "archive" / REF / "10" / "nowcast.png").write_bytes(b"new")
    assert not freeze_round(ledger, REF)
    assert frozen.read_text() == page
    assert (ledger / "pages" / f"{REF}.png").read_bytes() == b"png 10"


def test_frozen_pages_are_copied_not_rebuilt(tmp_path: Path) -> None:
    ledger = _ledger(tmp_path)
    pages = ledger / "pages"
    pages.mkdir()
    (pages / f"{REF}.md").write_text("# Round frozen\n")
    (pages / f"{REF}.png").write_bytes(b"frozen png")
    docs, gen = tmp_path / "docs", tmp_path / "gen"
    assert make_round_pages(ledger, docs, gen) == [REF, "2026-10-24"]
    rounds = docs / "submission" / "rounds"
    assert (rounds / f"{REF}.md").read_text() == "# Round frozen\n"
    assert (rounds / f"{REF}.png").read_bytes() == b"frozen png"
    # A round not yet frozen is still built from the ledger.
    assert (rounds / "2026-10-24.md").read_text().startswith("# Round")


def test_rounds_index_lists_rounds_by_date(tmp_path: Path) -> None:
    docs, gen = tmp_path / "docs", tmp_path / "gen"
    make_round_pages(_ledger(tmp_path), docs, gen)
    index = (docs / "submission" / "rounds" / "index.md").read_text()
    assert index.index(REF) < index.index("2026-10-24")


def test_freeze_round_command(tmp_path: Path) -> None:
    ledger = _ledger(tmp_path)
    args = ["freeze-round", "--ledger", str(ledger), "--round", REF]
    assert main(args) == 0
    assert (ledger / "pages" / f"{REF}.md").exists()
    assert main(args) == 0  # already frozen: nothing to do
