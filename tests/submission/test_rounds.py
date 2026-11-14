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


RUN = "https://github.com/o/r/actions/runs/10"
IDS = ("seabbs-renewal", "seabbs-rwnowcast", "seabbs-new")


def _model(model_id: str, i: int) -> dict[str, str]:
    return {
        "status": "submitted",
        "produced_by": f"label {model_id}",
        "pr_url": f"https://github.com/h/hub/pull/{90 + i}",
        "run": RUN,
        "code_commit": "abcdef1234",
        "archive": f"archive/{REF}/10/{model_id}",
    }


def _ledger(root: Path) -> Path:
    ledger = root / "ledger"
    state = {
        "schema_version": 1,
        "rounds": {
            REF: {
                "status": "submitted",
                "as_of": "2026-10-24",
                "models": {m: _model(m, i) for i, m in enumerate(IDS)},
            },
            "2026-10-24": {
                "status": "skipped",
                "models": {"seabbs-renewal": {"status": "skipped"}},
            },
        },
    }
    ledger.mkdir(parents=True)
    (ledger / "state.json").write_text(json.dumps(state))
    (ledger / "archive" / REF / "8").mkdir(parents=True)
    for i, model_id in enumerate(IDS):
        d = ledger / "archive" / REF / "10" / model_id
        d.mkdir(parents=True)
        (d / "nowcast.png").write_bytes(f"png {model_id}".encode())
        (d / "provenance.json").write_text(
            json.dumps(
                {
                    "selected": "main",
                    "status": "warn",
                    "reason": "The main model cleared every check.",
                    "attempts": [
                        {"id": "main", "label": "main model", "status": "pass"},
                        {"id": "r1", "label": "r", "status": "not needed"},
                    ],
                }
            )
        )
        (d / "run.json").write_text(
            json.dumps({"model": f"model{i}", "config": "main"})
        )
        (d / "checks.json").write_text(
            json.dumps(
                {
                    "checks": [
                        {"name": "schema", "status": "pass", "summary": "ok"}
                    ]
                }
            )
        )
        rows = [
            {"location": loc, "target_end_date": "2026-11-07", "value": v}
            for loc, v in (("CD", 10), ("CD", 20), ("CD-IT", 5))
        ]
        pd.DataFrame(rows).to_csv(d / f"{REF}-{model_id}.csv")
    return ledger


def _state(url: str) -> str:
    return "merged" if url.endswith("90") else "open"


def test_latest_run_is_the_largest_id(tmp_path: Path) -> None:
    ledger = _ledger(tmp_path)
    run = latest_run(ledger / "archive" / REF)
    assert run is not None
    assert run.name == "10"


def test_round_page_has_a_tab_per_model_id(tmp_path: Path) -> None:
    docs, gen = tmp_path / "docs", tmp_path / "gen"
    done = make_round_pages(_ledger(tmp_path), docs, gen, _state)
    assert done == [REF, "2026-10-24"]
    rounds = docs / "submission" / "rounds"
    page = (rounds / f"{REF}.md").read_text()
    assert page.startswith(f"# Round {REF}")
    for i, model_id in enumerate(IDS):
        assert f'=== "{model_id}"' in page
        image = f"{REF}-{model_id}.png"
        assert f"({image})" in page
        assert (rounds / image).read_bytes() == f"png {model_id}".encode()
        assert f"https://github.com/h/hub/pull/{90 + i}" in page
        assert f"model{i} (`main`)" in page
    assert "    | 2026-11-07 | 15 |" in page
    assert "    | schema | pass | ok |" in page
    assert "    | main model | submitted |  |" in page
    assert "| r |" not in page
    assert "[`abcdef1`](https://github.com/o/r/commit/abcdef1234)" in page
    # A round with no model submitted still gets a page, without a plot.
    assert "![" not in (rounds / "2026-10-24.md").read_text()
    snippet = (gen / "latest-submission.md").read_text()
    assert f"(submission/rounds/{REF}.md)" in snippet
    assert all(m in snippet for m in IDS)


def test_log_has_a_table_per_model_id(tmp_path: Path) -> None:
    docs, gen = tmp_path / "docs", tmp_path / "gen"
    make_round_pages(_ledger(tmp_path), docs, gen, _state)
    log = (docs / "submission" / "rounds" / "index.md").read_text()
    assert log.startswith("# Submission log")
    for i, model_id in enumerate(IDS):
        table = log.split(f"## {model_id}")[1].split("##")[0]
        assert f"| [{REF}]({REF}.md) | [10]({RUN}) |" in table
        assert f"| model{i} | warn |" in table
        assert f"[#{90 + i}](https://github.com/h/hub/pull/{90 + i})" in table
        row = next(x for x in table.splitlines() if x.startswith(f"| [{REF}]"))
        assert row.endswith("| merged |" if i == 0 else "| open |")


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
    assert (
        ledger / "pages" / f"{REF}-seabbs-new.png"
    ).read_bytes() == b"png seabbs-new"
    # A later run of the round, or new rendering code, never rewrites it.
    (
        ledger / "archive" / REF / "10" / "seabbs-new" / "nowcast.png"
    ).write_bytes(b"new")
    assert not freeze_round(ledger, REF)
    assert frozen.read_text() == page
    assert (
        ledger / "pages" / f"{REF}-seabbs-new.png"
    ).read_bytes() == b"png seabbs-new"


def test_frozen_pages_are_copied_not_rebuilt(tmp_path: Path) -> None:
    ledger = _ledger(tmp_path)
    pages = ledger / "pages"
    pages.mkdir()
    (pages / f"{REF}.md").write_text("# Round frozen\n")
    (pages / f"{REF}-seabbs-renewal.png").write_bytes(b"frozen png")
    docs, gen = tmp_path / "docs", tmp_path / "gen"
    assert make_round_pages(ledger, docs, gen, _state) == [
        REF,
        "2026-10-24",
    ]
    rounds = docs / "submission" / "rounds"
    assert (rounds / f"{REF}.md").read_text() == "# Round frozen\n"
    assert (rounds / f"{REF}-seabbs-renewal.png").read_bytes() == b"frozen png"
    # A round not yet frozen is still built from the ledger.
    assert (rounds / "2026-10-24.md").read_text().startswith("# Round")


def test_rounds_index_lists_rounds_by_date(tmp_path: Path) -> None:
    docs, gen = tmp_path / "docs", tmp_path / "gen"
    make_round_pages(_ledger(tmp_path), docs, gen, _state)
    index = (docs / "submission" / "rounds" / "index.md").read_text()
    renewal = index.split("## seabbs-renewal")[1].split("##")[0]
    assert renewal.index(REF) < renewal.index("2026-10-24")


def test_freeze_round_command(tmp_path: Path) -> None:
    ledger = _ledger(tmp_path)
    args = ["freeze-round", "--ledger", str(ledger), "--round", REF]
    assert main(args) == 0
    assert (ledger / "pages" / f"{REF}.md").exists()
    assert main(args) == 0  # already frozen: nothing to do
