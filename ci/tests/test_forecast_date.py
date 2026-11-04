from __future__ import annotations

import datetime as dt
import io
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

import pytest

from submission_ci.forecast_date import (
    append,
    fetch,
    main,
    parse_date,
    problems,
    read_rows,
    write_rows,
)

COLS = ["reference_date", "forecast_date", "location", "value"]
REF = "2026-11-07"


def table(date: str, n: int = 3) -> tuple[list[str], list[dict[str, str]]]:
    rows = [
        {
            "reference_date": REF,
            "forecast_date": date,
            "location": f"CD-{i}",
            "value": f"{i}.0",
        }
        for i in range(n)
    ]
    return COLS, rows


@pytest.mark.parametrize(
    ("value", "ok"),
    [
        ("2026-11-02", True),
        ("2026-02-30", False),
        ("NA", False),
        ("", False),
        ("2026-11-02T10:00", False),
        ("02/11/2026", False),
        ("0000-01-01", False),
    ],
)
def test_parse_date(value: str, ok: bool) -> None:
    assert (parse_date(value) is not None) is ok


def test_new_file_with_one_date_is_valid() -> None:
    cols, rows = table("2026-11-02")
    assert problems(cols, rows, None, dt.date(2026, 11, 2)) == []


def test_column_must_be_present() -> None:
    cols, rows = table("2026-11-02")
    assert problems([c for c in cols if c != "forecast_date"], rows)


def test_date_must_parse() -> None:
    cols, rows = table("2026-11-02")
    rows[1]["forecast_date"] = "NA"
    assert "YYYY-MM-DD" in problems(cols, rows)[0]


def test_date_must_not_follow_reference_date() -> None:
    cols, rows = table(REF)
    assert problems(cols, rows) == []
    cols, rows = table("2026-11-08")
    assert "on or before" in problems(cols, rows)[0]


def test_new_file_has_one_date() -> None:
    cols, rows = table("2026-11-02")
    rows[0]["forecast_date"] = "2026-11-01"
    assert "exactly one" in problems(cols, rows)[0]


@pytest.mark.parametrize(
    ("today", "ok"),
    [
        (dt.date(2026, 11, 1), True),
        (dt.date(2026, 11, 3), True),
        (dt.date(2026, 11, 4), False),
        (dt.date(2026, 10, 31), False),
    ],
)
def test_new_date_must_be_recent(today: dt.date, ok: bool) -> None:
    cols, rows = table("2026-11-02")
    assert (problems(cols, rows, None, today) == []) is ok


def test_append_keeps_original_rows_and_adds_one_date() -> None:
    old = table("2026-11-01")
    new = table("2026-11-03", n=2)
    cols, rows = append(old, new)
    assert rows[:3] == old[1]
    assert rows[3:] == new[1]
    # Retained dates are not subject to the recency check.
    assert problems(cols, rows, old, dt.date(2026, 11, 3)) == []


def test_append_refuses_a_reused_date() -> None:
    with pytest.raises(ValueError, match="kept unchanged"):
        append(table("2026-11-01"), table("2026-11-01", n=2))


def test_append_refuses_two_new_dates() -> None:
    new = table("2026-11-03", n=2)
    new[1][0]["forecast_date"] = "2026-11-04"
    with pytest.raises(ValueError, match="exactly one"):
        append(table("2026-11-01"), new)


def test_append_refuses_a_hub_copy_without_the_column() -> None:
    old = (["reference_date", "location", "value"], [{"reference_date": REF}])
    with pytest.raises(ValueError, match="columns differ"):
        append(old, table("2026-11-03"))


@pytest.mark.parametrize("change", ["edit", "drop", "duplicate"])
def test_changed_existing_rows_fail(change: str) -> None:
    old = table("2026-11-01")
    cols, rows = append(old, table("2026-11-03"))
    if change == "edit":
        rows[0] = {**rows[0], "value": "1"}
    elif change == "drop":
        rows = rows[1:]
    else:
        rows = [*rows, dict(rows[0])]
    assert "kept unchanged" in problems(cols, rows, old)[0]


def test_fetch_from_a_checkout(tmp_path: Path) -> None:
    hub = tmp_path / "hub"
    name = f"{REF}-seabbs-renewal.csv"
    (hub / "model-output" / "seabbs-renewal").mkdir(parents=True)
    (hub / "model-output" / "seabbs-renewal" / name).write_text("a\n1\n")
    out = tmp_path / "prev.csv"
    assert fetch("seabbs-renewal", REF, out, checkout=hub)
    assert out.read_text() == "a\n1\n"
    assert not fetch("seabbs-other", REF, out, checkout=hub)


def test_fetch_reads_raw_github_and_treats_404_as_absent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    seen: list[str] = []

    def opener(url: str, timeout: float = 0.0) -> Any:  # noqa: ARG001
        seen.append(url)
        if "missing" in url:
            raise urllib.error.HTTPError(url, 404, "Not Found", None, None)  # type: ignore[arg-type]
        return io.BytesIO(b"a\n1\n")

    monkeypatch.setattr(urllib.request, "urlopen", opener)
    out = tmp_path / "p.csv"
    assert fetch("seabbs-renewal", REF, out)
    assert seen == [
        "https://raw.githubusercontent.com/InsightNet-US/BDBV-Modeling-Hub/"
        f"main/model-output/seabbs-renewal/{REF}-seabbs-renewal.csv"
    ]
    assert not fetch("missing", REF, out)


def test_cli_append_and_check(tmp_path: Path) -> None:
    old = tmp_path / "old.csv"
    new = tmp_path / "new.csv"
    out = tmp_path / "out.csv"
    write_rows(old, *table("2026-11-01"))
    write_rows(new, *table("2026-11-03"))
    assert main(["append", "--previous", str(old), "--new", str(new),
                 "--out", str(out)]) == 0  # fmt: skip
    assert len(read_rows(out)[1]) == 6
    assert main(["check", str(out), "--previous", str(old),
                 "--today", "2026-11-03"]) == 0  # fmt: skip
    assert main(["check", str(out), "--previous", str(old),
                 "--today", "2026-11-10"]) == 1  # fmt: skip
    assert main(["append", "--previous", str(out), "--new", str(new),
                 "--out", str(tmp_path / "x.csv")]) == 1  # fmt: skip
