"""Tests of scripts/germany_forecast.py: German 1 to 4 week forecasts."""

from __future__ import annotations

import datetime as dt
import importlib.util
import json
import sys
from pathlib import Path

import pandas as pd
import pytest

from bdbvhubrenewal.data.germany import nowcast_tasks, read_triangle
from bdbvhubrenewal.evaluation.germany import load_germany_config

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
FIXTURE = (
    ROOT / "tests" / "fixtures" / "germany" / "germany_covid19_hosp.csv.gz"
)
DATE = "2021-09-15"
"""A Wednesday: its hub week ends on Saturday 2021-09-18."""
RUN_DATE = "2021-09-25"
"""The first fixture date the random walk has a spread at in both tasks."""
SPLIT = "2021-10-01"


def _load(name: str):
    sys.path.insert(0, str(SCRIPTS))
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
    assert spec is not None
    assert spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def gf():
    return _load("germany_forecast")


@pytest.fixture(scope="module")
def cs():
    return _load("checkpoint_scores")


@pytest.fixture(scope="module")
def poll():
    return _load("checkpoint_poll")


@pytest.fixture(scope="module")
def triangle() -> pd.DataFrame:
    return read_triangle(FIXTURE)


@pytest.fixture(scope="module")
def forecasts(gf, triangle) -> pd.DataFrame:
    return gf.forecasts(
        ["randomwalknowcast"], triangle, RUN_DATE, 30, 1, split_date=SPLIT
    )


# Tasks -----------------------------------------------------------------


def test_forecast_tasks_extend_four_weeks_past_the_nowcast(gf, triangle):
    now = nowcast_tasks(triangle, DATE, 10, 1)
    ahead = gf.forecast_tasks(triangle, DATE, 10, 1, split_date=SPLIT)
    assert len(ahead) == len(now) == 2
    for task, base in zip(ahead, now, strict=True):
        last = dt.date.fromisoformat(max(base.target_end_dates))
        assert task.target_end_dates == tuple(
            (last + dt.timedelta(weeks=k)).isoformat() for k in (1, 2, 3, 4)
        )
        # Same data and places as the nowcast: only the weeks change.
        assert task.locations == base.locations
        assert task.reference_date == base.reference_date
        pd.testing.assert_frame_equal(task.daily_history, base.daily_history)


def test_forecasts_end_7_to_28_days_after_the_nowcast_date(forecasts):
    assert set(forecasts["horizon"]) == {7, 14, 21, 28}
    assert set(forecasts["horizon_weeks"]) == {1, 2, 3, 4}
    ends = {
        (dt.date.fromisoformat(RUN_DATE) + dt.timedelta(days=h)).isoformat()
        for h in (7, 14, 21, 28)
    }
    assert set(forecasts["target_end_date"]) == ends
    assert (forecasts["origin"] == RUN_DATE).all()
    # Both tasks: Germany and its two states, and the six age groups.
    assert forecasts["location"].nunique() == 9
    assert set(forecasts["model"]) == {"randomwalknowcast"}
    assert forecasts["quantile"].nunique() == 7


def test_models_are_randomwalknowcast_zero_shot_and_the_checkpoint(gf, cs):
    assert gf.label_models(cs.REFERENCE) == (
        "randomwalknowcast",
        "fm_chronos_daily_ragged",
        "fm_chronos_weekly_ragged",
    )
    assert gf.label_models("slots-abc") == cs.FINETUNED


# The split guard -------------------------------------------------------


@pytest.mark.parametrize("date", [SPLIT, "2021-10-05"])
def test_dates_on_or_after_the_split_are_refused(gf, triangle, date):
    with pytest.raises(ValueError, match="held-out"):
        gf.forecast_tasks(triangle, date, 10, 1, split_date=SPLIT)
    with pytest.raises(ValueError, match="held-out"):
        gf.forecasts(["randomwalknowcast"], triangle, date, 10, 1, SPLIT)


def test_the_unit_refuses_held_out_dates_before_running(gf):
    config = load_germany_config(ROOT / "config" / "germany.toml")
    with pytest.raises(ValueError, match="held-out"):
        gf.check_dates(["2022-02-09", config.split_date], config.split_date)
    gf.check_dates(["2022-02-09"], config.split_date)
    assert gf.dev_dates()[-1] == "2022-02-09"
    assert len(gf.dev_dates()) == 80


# Truth -----------------------------------------------------------------


def test_every_horizon_gets_the_40_day_rule_s_truth(gf, triangle):
    # The fixture's reports end on 2021-10-20, 7 days after DATE's 4-week
    # sum, so a 7-day delay stands in for the 40-day one here.
    ends = [
        (dt.date.fromisoformat(DATE) + dt.timedelta(days=h)).isoformat()
        for h in (7, 14, 21, 28)
    ]
    forecasts = pd.DataFrame(
        {
            "model": "randomwalknowcast",
            "origin": DATE,
            "location": ["DE", "DE-BY", "00-04", "80+"],
            "target_end_date": ends,
            "horizon": [7, 14, 21, 28],
            "horizon_weeks": [1, 2, 3, 4],
            "quantile": 0.5,
            "value": 1.0,
        }
    )
    fc = gf.with_truth(forecasts, triangle, truth_delay=7)
    assert fc["truth"].notna().all()
    lag = pd.to_datetime(fc["truth_vintage"]) - pd.to_datetime(
        fc["target_end_date"]
    )
    assert (lag.dt.days >= 7).all()


def test_the_triangle_holds_truth_for_the_last_date_s_fourth_week(gf):
    config = load_germany_config(ROOT / "config" / "germany.toml")
    last = dt.date.fromisoformat(gf.dev_dates()[-1])
    due = last + dt.timedelta(days=28 + config.truth_delay)
    # The hub triangle is as of 2022-08-08 (config/germany.toml).
    assert due < dt.date(2022, 8, 8)


# Rows ------------------------------------------------------------------


def _scores(model, checkpoint, wis, horizon=1):
    end = (dt.date(2021, 12, 1) + dt.timedelta(weeks=horizon)).isoformat()
    return pd.DataFrame(
        {
            "model": model,
            "checkpoint": checkpoint,
            "origin": ["2021-12-01"] * 2,
            "truth_vintage": ["2022-01-30"] * 2,
            "location": ["DE", "DE-BY"],
            "target_end_date": [end] * 2,
            "horizon_weeks": horizon,
            "level": ["national", "states"],
            "wis": [wis, wis],
            "log_wis": [wis / 10, wis / 10],
            "coverage_50": [1.0, 0.0],
            "coverage_95": [1.0, 1.0],
        }
    )


@pytest.fixture(scope="module")
def rows(gf):
    cells = pd.concat(
        [
            _scores(m, c, w, h)
            for h in (1, 4)
            for m, c, w in [
                ("randomwalknowcast", "zero-shot", 2.0),
                ("fm_chronos_daily_ragged", "zero-shot", 4.0),
                ("fm_chronos_daily_ragged", "slots-abc", 1.0),
            ]
        ]
    )
    return gf.relative_rows(cells)


def test_rows_are_a_study_of_their_own_with_the_evidence_columns(cs, rows):
    assert list(rows.columns) == cs.COLUMNS
    assert set(rows["study"]) == {"germany_forecast"}
    assert set(rows["target"]) == {"7day"}
    assert set(rows["horizon_weeks"]) == {1, 4}
    assert set(rows["level"]) == {"national", "states"}
    assert (rows["rule"] == cs.RULE).all()


def test_rows_are_relative_to_randomwalknowcast_and_to_zero_shot(rows):
    by = rows.set_index(
        ["rel_to", "horizon_weeks", "level", "model", "checkpoint"]
    )["rel_wis"]
    rw = ("randomwalknowcast", 1, "national")
    assert by.loc[(*rw, "randomwalknowcast", "zero-shot")] == 1
    assert by.loc[(*rw, "fm_chronos_daily_ragged", "slots-abc")] == 0.5
    zs = ("zero-shot", 4, "states")
    assert by.loc[(*zs, "fm_chronos_daily_ragged", "zero-shot")] == 1
    assert by.loc[(*zs, "fm_chronos_daily_ragged", "slots-abc")] == 0.25
    # randomwalknowcast has no zero-shot of its own.
    assert not (
        (rows["rel_to"] == "zero-shot") & (rows["model"] == "randomwalknowcast")
    ).any()


def test_the_evidence_keeps_the_horizon(cs, rows):
    merged = cs.merge_evidence(pd.DataFrame(columns=cs.COLUMNS), rows)
    assert "horizon_weeks" in cs.COLUMNS
    assert set(merged["horizon_weeks"].dropna()) == {1, 4}


# The unit's cache and the scorer's hooks -------------------------------


def _cache(tmp_path, label, dates):
    root = tmp_path / "germany_forecast" / label
    root.mkdir(parents=True, exist_ok=True)
    for d in dates:
        (root / f"{d}.csv.gz").write_text("")


def test_a_unit_is_done_when_every_date_has_its_file(gf, cs, tmp_path):
    _cache(tmp_path, "reference", ["2021-11-22"])
    assert gf.unit_done(tmp_path, "reference", ["2021-11-22"])
    assert not gf.unit_done(tmp_path, "reference", ["2021-11-22", "x"])
    _cache(tmp_path, "slots-abc", gf.dev_dates())
    assert cs.unit_done(tmp_path, "germany_forecast", "slots-abc")
    assert not cs.unit_done(tmp_path, "germany_forecast", "slots-abc", "daily")


def test_the_scorer_runs_the_unit_with_the_7day_target_only(
    cs, gf, monkeypatch, tmp_path
):
    ran = []
    monkeypatch.setattr(gf, "unit", lambda c, lb: ran.append((c, lb)))
    monkeypatch.setattr(cs, "_germany_forecast", lambda: gf)
    cs.main(["germany_forecast", str(tmp_path), "slots-abc", "7day"])
    assert ran == [(tmp_path, "slots-abc")]
    with pytest.raises(SystemExit):
        cs.main(["germany_forecast", str(tmp_path), "slots-abc", "daily"])


# The poller ------------------------------------------------------------


def _settings(cache: Path, label: str, run: str, n: int) -> None:
    held = cache / "heldout"
    held.mkdir(exist_ok=True)
    settings = {
        "one-model": {
            "series_bias": {"init": "zero"},
            "sources": ["daily", "daily+weekly"],
        },
        "no-slots": {},
    }[run]
    (held / f"{label}.json").write_text(
        json.dumps({**settings, "n_outbreaks": n})
    )


@pytest.fixture
def live(poll, tmp_path):
    _settings(tmp_path, "slots-a", "one-model", 125)
    _settings(tmp_path, "slots-b", "one-model", 1000)
    _settings(tmp_path, "slots-c", "one-model", 750)
    _settings(tmp_path, "nnnn", "no-slots", 2000)
    return tmp_path, poll.runs_of(tmp_path)


def _done(monkeypatch, poll, done):
    monkeypatch.setattr(
        poll.cs,
        "unit_done",
        lambda _c, study, label, _target="7day": (study, label) in done,
    )


def test_forecasts_wait_for_every_live_german_nowcast(poll, live, monkeypatch):
    cache, runs = live
    # slots-b, the newest, is still nowcasting: nothing runs yet, not
    # even the reference, and not the older slots-a that is done.
    _done(monkeypatch, poll, {("germany", "slots-a"), ("germany", "slots-c")})
    assert poll.forecast_units(cache, runs) == []


def test_forecasts_run_the_reference_then_the_newest_checkpoint_only(
    poll, live, monkeypatch
):
    cache, runs = live
    done = {("germany", lb) for lb in ("slots-a", "slots-b", "slots-c")}
    done.add(("germany", "nnnn"))
    _done(monkeypatch, poll, done)
    assert poll.forecast_units(cache, runs) == [
        ("germany_forecast", "reference"),
        ("germany_forecast", "slots-b"),
    ]
    done.add(("germany_forecast", "reference"))
    assert poll.forecast_units(cache, runs) == [("germany_forecast", "slots-b")]
    done.add(("germany_forecast", "slots-b"))
    assert poll.forecast_units(cache, runs) == []


def test_the_forecast_job_comes_after_the_zones_before_the_daily_target(
    poll,
):
    order = poll.IDLE_JOBS
    assert order.index(poll.ZONE_JOB) < order.index(poll.FORECAST_JOB)
    assert order.index(poll.FORECAST_JOB) < order.index(poll.DAILY_JOB)


def test_forecast_units_name_their_study_on_the_command_line(poll, tmp_path):
    cmd = poll.unit_command(tmp_path, ("germany_forecast", "slots-a"))
    assert cmd[-4:] == ["germany_forecast", str(tmp_path), "slots-a", "7day"]
    assert "MemoryMax=6G" in cmd
    assert cmd[cmd.index("nice") + 1 : cmd.index("nice") + 3] == ["-n", "19"]
