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
    assert '!!! note "Fits not made for this code"' in note
    assert "bbbbbbb" in note
    assert "`renewal` at 2026-10-24 and 2026-11-01." in note
    assert "failed for `renewal` at 2026-10-17." in note
    assert "older code are shown for `renewal` at 2026-10-17 and" in note
    assert "Not shown: `renewal` at 2026-11-01." in note
    assert "run" not in note
    running = store_note(status, "b" * 40, "https://run")
    assert "A [Fit store run](https://run) is fitting them." in running


def test_no_record_says_no_fits(tmp_path: Path) -> None:
    write_store_note(tmp_path, tmp_path / "gen")
    text = (tmp_path / "gen" / "fit-store.md").read_text()
    assert "No fitted model has a stored fit yet." in text


def test_a_preview_says_in_one_line_what_it_lacks() -> None:
    status = [
        _row("2026-11-01", None, False),
        _row("2026-10-24", "old", False),
        _row("2026-10-17", "k", True),
    ]
    note = store_note(status, "b" * 40, preview="c" * 40)
    assert "!!!" not in note
    assert "Not yet fitted" not in note
    extra = [line for line in note.splitlines() if line][1:]
    assert len(extra) == 1
    (line,) = extra
    # The preview's own head, not the merge commit, and the counts.
    assert "ccccccc" in line
    assert "bbbbbbb" not in line
    assert "2 fits" in line
    assert "older fits are shown for 1" in line
    assert "1 has none" in line


def test_a_preview_with_current_fits_adds_nothing() -> None:
    note = store_note([_row("2026-10-24", "k", True)], "b" * 40, preview="c")
    assert note == store_note([_row("2026-10-24", "k", True)], "b" * 40)


def test_the_preview_head_comes_from_the_environment(
    tmp_path: Path, monkeypatch
) -> None:
    import json

    (tmp_path / "fit-store.json").write_text(
        json.dumps([_row("2026-11-01", None, False)])
    )
    monkeypatch.setenv("GITHUB_SHA", "b" * 40)
    monkeypatch.setenv("PREVIEW_HEAD", "c" * 40)
    write_store_note(tmp_path, tmp_path / "gen")
    text = (tmp_path / "gen" / "fit-store.md").read_text()
    assert "ccccccc" in text
    assert "!!!" not in text


def test_pages_passes_the_pull_request_head() -> None:
    import yaml

    pages = Path(__file__).parents[2] / ".github" / "workflows" / "pages.yml"
    steps = yaml.safe_load(pages.read_text())["jobs"]["build"]["steps"]
    (step,) = [
        s
        for s in steps
        if s.get("name")
        == "Compare with hub forecasts, then the docs figures and tables"
    ]
    assert step["env"]["PREVIEW_HEAD"] == (
        "${{ github.event.pull_request.head.sha }}"
    )
