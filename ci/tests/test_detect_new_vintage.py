from __future__ import annotations

import datetime as dt
import json
import subprocess
from pathlib import Path

import pytest
from conftest import HUB, FakeRunner

from submission_ci.common import read_json
from submission_ci.detect_new_vintage import (
    DataVersion,
    GhHubSource,
    code_changed,
    decide,
    empty_state,
    main,
    open_reference_dates,
    record,
    record_model,
)
from submission_ci.window import KINSHASA, current_window, window_for

TASKS = read_json(HUB / "tasks.json")
V = DataVersion("d" * 40, "2026-10-31T20:00:00Z", "h" * 40, "2026-10-31")
MONDAY = dt.datetime(2026, 11, 2, 10, tzinfo=dt.UTC)


def state_with(**rounds: dict[str, str]) -> dict[str, object]:
    s = empty_state()
    s["rounds"] = {k.replace("_", "-"): v for k, v in rounds.items()}
    return s


def test_window_is_sunday_to_tuesday_kinshasa() -> None:
    w = window_for(dt.date(2026, 11, 7))
    assert w.opens == dt.datetime(2026, 11, 1, tzinfo=KINSHASA)
    assert w.closes == dt.datetime(2026, 11, 4, tzinfo=KINSHASA)
    # Saturday 23:30 UTC is already Sunday 00:30 in Kinshasa.
    assert w.is_open(dt.datetime(2026, 10, 31, 23, 30, tzinfo=dt.UTC))
    assert not w.is_open(dt.datetime(2026, 10, 31, 22, 59, tzinfo=dt.UTC))
    # Tuesday 22:59 UTC is 23:59 in Kinshasa, the last minute.
    assert w.is_open(dt.datetime(2026, 11, 3, 22, 59, tzinfo=dt.UTC))
    assert not w.is_open(dt.datetime(2026, 11, 3, 23, 0, tzinfo=dt.UTC))


def test_window_rejects_non_saturday() -> None:
    with pytest.raises(ValueError, match="Saturday"):
        window_for(dt.date(2026, 11, 6))


@pytest.mark.parametrize(
    ("now", "ref"),
    [
        (dt.datetime(2026, 11, 2, 10, tzinfo=dt.UTC), "2026-11-07"),
        (dt.datetime(2026, 11, 5, 10, tzinfo=dt.UTC), "2026-11-14"),
        (dt.datetime(2026, 11, 7, 10, tzinfo=dt.UTC), "2026-11-14"),
    ],
)
def test_current_window_is_open_or_next(now: dt.datetime, ref: str) -> None:
    assert current_window(now).reference_date.isoformat() == ref


def test_open_reference_dates_reads_the_sample_round() -> None:
    assert open_reference_dates(TASKS) == {"2026-10-31", "2026-11-07"}


def test_new_vintage_runs() -> None:
    d = decide(state_with(), V, TASKS, MONDAY)
    assert d.should_run and d.fresh and d.reference_date == "2026-11-07"


def test_same_data_as_this_round_does_not_rerun() -> None:
    s = state_with(
        **{"2026_11_07": {"data_commit": V.data_commit, "as_of": V.as_of}}
    )
    d = decide(s, V, TASKS, MONDAY)
    assert not d.should_run and "no new hub data" in d.reason


def test_data_fix_during_window_updates() -> None:
    s = state_with(
        **{"2026_11_07": {"data_commit": "e" * 40, "as_of": V.as_of}}
    )
    d = decide(s, V, TASKS, MONDAY)
    assert d.should_run and "updating" in d.reason


def test_last_weeks_vintage_waits_then_falls_back_late() -> None:
    s = state_with(
        **{"2026_10_31": {"data_commit": "c" * 40, "as_of": V.as_of}}
    )
    d = decide(s, V, TASKS, MONDAY)
    assert not d.should_run and not d.fresh
    late = dt.datetime(2026, 11, 3, 15, tzinfo=dt.UTC)  # 8 h left
    d = decide(s, V, TASKS, late)
    assert d.should_run and "latest data" in d.reason


def test_submitted_round_is_left_to_a_manual_dispatch() -> None:
    s = state_with(
        **{
            "2026_11_07": {
                "data_commit": "e" * 40,
                "as_of": V.as_of,
                "status": "submitted",
            }
        }
    )
    d = decide(s, V, TASKS, MONDAY)
    assert not d.should_run and "already has a hub PR" in d.reason
    assert decide(s, V, TASKS, MONDAY, force=True).should_run


SUBMITTED = state_with(
    **{
        "2026_11_07": {
            "data_commit": V.data_commit,
            "as_of": V.as_of,
            "status": "submitted",
            "code_commit": "f" * 40,
        }
    }
)


def test_model_code_change_updates_a_submitted_round() -> None:
    d = decide(SUBMITTED, V, TASKS, MONDAY, changed_since=lambda _: True)
    assert d.should_run and d.code_changed and "model code" in d.reason
    d = decide(SUBMITTED, V, TASKS, MONDAY, changed_since=lambda _: False)
    assert not d.should_run and "already has a hub PR" in d.reason


def test_model_code_change_outside_window_does_not_run() -> None:
    friday = dt.datetime(2026, 11, 6, 10, tzinfo=dt.UTC)
    d = decide(SUBMITTED, V, TASKS, friday, changed_since=lambda _: True)
    assert not d.should_run


def test_code_changed_compares_model_paths(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)

    def git(*args: str) -> str:
        return subprocess.run(
            ["git", "-c", "user.name=t", "-c", "user.email=t@t", *args],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()

    git("init", "-q")
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "m.py").write_text("a = 1\n")
    (tmp_path / "README.md").write_text("x\n")
    git("add", "-A")
    git("commit", "-q", "-m", "one")
    first = git("rev-parse", "HEAD")
    (tmp_path / "README.md").write_text("y\n")
    git("commit", "-q", "-am", "docs")
    assert not code_changed(first)
    (tmp_path / "src" / "m.py").write_text("a = 2\n")
    git("commit", "-q", "-am", "model")
    assert code_changed(first)


def test_tuesday_morning_check_falls_back_to_the_latest_data() -> None:
    s = state_with(
        **{"2026_10_31": {"data_commit": "c" * 40, "as_of": V.as_of}}
    )
    tuesday = dt.datetime(2026, 11, 3, 6, tzinfo=dt.UTC)  # 17 h left
    d = decide(s, V, TASKS, tuesday)
    assert d.should_run and "latest data" in d.reason


def test_outside_window_never_runs_even_forced() -> None:
    friday = dt.datetime(2026, 11, 6, 10, tzinfo=dt.UTC)
    assert not decide(state_with(), V, TASKS, friday, force=True).should_run
    d = decide(state_with(), V, TASKS, friday, force=True, ignore_window=True)
    assert d.should_run


def test_round_missing_from_tasks_does_not_run() -> None:
    later = dt.datetime(2026, 11, 9, 10, tzinfo=dt.UTC)
    d = decide(state_with(), V, TASKS, later)
    assert not d.should_run and "tasks.json" in d.reason


def test_record_keeps_history() -> None:
    s = record(
        empty_state(),
        "2026-11-07",
        data_commit="a",
        as_of="x",
        status="failed",
    )
    s = record(
        s,
        "2026-11-07",
        data_commit="b",
        as_of="x",
        status="submitted",
        produced_by="main model",
        code_commit="c",
    )
    entry = s["rounds"]["2026-11-07"]
    assert entry["data_commit"] == "b" and entry["code_commit"] == "c"
    assert [h["status"] for h in entry["history"]] == ["failed", "submitted"]


def test_record_model_keeps_each_model_across_round_records() -> None:
    s = record(
        empty_state(),
        "2026-11-07",
        data_commit="a",
        as_of="x",
        status="submitted",
    )
    for model_id, pr in (("seabbs-renewal", "95"), ("seabbs-rwnowcast", "94")):
        s = record_model(
            s,
            "2026-11-07",
            model_id,
            status="submitted",
            archive=f"archive/2026-11-07/1/{model_id}",
            pr_url=f"https://h/pull/{pr}",
        )
    s = record(s, "2026-11-07", data_commit="b", as_of="x", status="submitted")
    models = s["rounds"]["2026-11-07"]["models"]
    assert sorted(models) == ["seabbs-renewal", "seabbs-rwnowcast"]
    assert models["seabbs-rwnowcast"]["pr_url"] == "https://h/pull/94"


def test_gh_source_picks_newest_commit(monkeypatch: pytest.MonkeyPatch) -> None:
    r = FakeRunner(real_git=False)
    r.on(["gh", "api", "repos/o/h/commits/main"], json.dumps({"sha": "HEAD"}))

    def commits(args: list[str]) -> object:
        from submission_ci.common import CommandResult

        stamp = (
            "2026-10-31T00:00:00Z"
            if "time-series" in args[2]
            else "2026-10-01T00:00:00Z"
        )
        sha = "new" if "time-series" in args[2] else "old"
        return CommandResult(
            0,
            json.dumps(
                [{"sha": sha, "commit": {"committer": {"date": stamp}}}]
            ),
        )

    r.on(["gh", "api"], commits)  # type: ignore[arg-type]
    r.on(["gh", "api", "repos/o/h/commits/main"], json.dumps({"sha": "HEAD"}))
    src = GhHubSource("o/h", runner=r)
    monkeypatch.setattr(
        src, "_raw", lambda _ref, _path: (HUB / "time-series.csv").read_text()
    )
    v = src.data_version()
    assert (v.data_commit, v.hub_ref, v.as_of) == ("new", "HEAD", "2026-10-31")


def test_cli_writes_outputs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    out = tmp_path / "out"
    monkeypatch.setenv("GITHUB_OUTPUT", str(out))
    assert (
        main(
            [
                "detect",
                "--state",
                str(HUB / "state.json"),
                "--fixture",
                str(HUB),
                "--now",
                MONDAY.isoformat(),
            ]
        )
        == 0
    )
    text = out.read_text()
    assert "should_run=true" in text and "reference_date=2026-11-07" in text
