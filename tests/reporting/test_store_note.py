from __future__ import annotations

from pathlib import Path

from bdbvhubrenewal.reporting.store_note import store_note, write_store_note


def _row(origin: str, used: str | None, current: bool, failed: bool = False):
    return {
        "model": "renewal",
        "origin": origin,
        "key": "k",
        "used": used,
        "commit": "a" * 40 if used else None,
        "current": current,
        "failed": failed,
    }


def test_current_fits_name_their_commit() -> None:
    note = store_note([_row("2026-10-24", "k", True)], "b" * 40)
    assert "aaaaaaa" in note
    assert "!!!" not in note


def test_fits_not_made_are_listed() -> None:
    status = [
        _row("2026-11-01", None, False),
        _row("2026-10-24", "old", False),
        _row("2026-10-17", "old", False, failed=True),
    ]
    note = store_note(status, "b" * 40)
    assert '!!! note "Fits not yet made for this code"' in note
    assert "bbbbbbb" in note
    assert "renewal` at 2026-10-17 and 2026-10-24 and 2026-11-01." in note
    assert "The fit failed for `renewal` at 2026-10-17." in note
    assert "older code are shown for `renewal` at 2026-10-17 and" in note
    assert "Not shown: `renewal` at 2026-11-01." in note
    assert "run" not in note
    running = store_note(status, "b" * 40, "https://run")
    assert "A [Fit store run](https://run) is fitting them." in running


def test_no_record_says_no_fits(tmp_path: Path) -> None:
    write_store_note(tmp_path, tmp_path / "gen")
    text = (tmp_path / "gen" / "fit-store.md").read_text()
    assert "No fitted model has a stored fit yet." in text
