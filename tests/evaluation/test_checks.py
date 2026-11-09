from __future__ import annotations

import pandas as pd

from bdbvhubrenewal.evaluation.checks import check_summary, flags, place_types
from bdbvhubrenewal.evaluation.scoring import SCORE_COLUMNS

ORIGIN = "2026-11-01"  # a Sunday, so the last week ends 2026-10-31
WEEKS = ["2026-10-03", "2026-10-10", "2026-10-17", "2026-10-24", "2026-10-31"]


def _counts(series: dict[str, list[int]]) -> pd.DataFrame:
    rows = [
        (ORIGIN, loc, week, n)
        for loc, values in series.items()
        for week, n in zip(WEEKS, values, strict=True)
        if n > 0
    ]
    return pd.DataFrame(
        rows, columns=["vintage", "location", "target_end_date", "total"]
    )


COUNTS = _counts(
    {
        "wamba": [13, 8, 19, 7, 0],
        "rungu": [1, 1, 1, 0, 1],
        "steady": [20, 22, 21, 19, 9],
        "quiet": [0, 0, 0, 0, 0],
    }
)


def _quantiles(rows: dict[str, tuple[float, float]]) -> pd.DataFrame:
    return pd.DataFrame(
        [("rw", ORIGIN, loc, q50, q95) for loc, (q50, q95) in rows.items()],
        columns=["model", "origin", "location", "q50", "q95"],
    )


def test_place_types_split_on_the_recent_maximum() -> None:
    origins = pd.DataFrame(
        {"origin": ORIGIN, "location": ["wamba", "rungu", "steady", "quiet"]}
    )
    got = place_types(COUNTS, origins).set_index("location")["place_type"]
    assert got.to_dict() == {
        "wamba": "larger",
        "rungu": "small",
        "steady": "larger",
        "quiet": "silent",
    }


def test_flags_catch_collapse_and_blow_up_only() -> None:
    q = _quantiles(
        {
            "wamba": (0.0, 1.0),  # level 13.5, median 0
            "rungu": (8.0, 57.0),  # recent maximum 1
            "steady": (20.0, 40.0),
            "quiet": (0.0, 2.0),
        }
    )
    got = flags(COUNTS, q).set_index("location")["flag"]
    assert got.to_dict() == {
        "wamba": "collapse",
        "rungu": "blow-up",
        "steady": "no flag",
        "quiet": "no flag",
    }


def test_flags_use_only_the_counts_reported_at_the_origin() -> None:
    later = pd.DataFrame(
        {
            "vintage": "2026-11-08",
            "location": "wamba",
            "target_end_date": "2026-10-31",
            "total": [20],
        }
    )
    q = _quantiles({"wamba": (0.0, 1.0), "rungu": (8.0, 57.0)})
    before = flags(COUNTS, q)
    after = flags(pd.concat([COUNTS, later], ignore_index=True), q)
    pd.testing.assert_frame_equal(before, after)


def test_check_summary_averages_by_place_type_and_flag() -> None:
    scores = pd.DataFrame(
        {
            "model": "rw",
            "origin": ORIGIN,
            "truth_vintage": "2026-11-08",
            "location": ["wamba", "wamba", "rungu"],
            "target_end_date": ["2026-10-31", "2026-11-07", "2026-11-07"],
            "scale": "zones",
            **dict.fromkeys(SCORE_COLUMNS, 0.0),
            "wis": [4.0, 2.0, 9.0],
            "log_wis": [1.0, 1.0, 2.0],
            "coverage_90": [0.0, 1.0, 0.0],
        }
    )
    types = pd.DataFrame(
        {
            "origin": ORIGIN,
            "location": ["wamba", "rungu"],
            "place_type": ["larger", "small"],
        }
    )
    flagged = pd.DataFrame(
        {
            "model": "rw",
            "origin": ORIGIN,
            "location": ["wamba", "rungu"],
            "flag": ["collapse", "blow-up"],
        }
    )
    got = check_summary(scores, types, flagged).set_index(["split", "group"])
    assert got.loc[("place type", "larger"), "WIS"] == 3.0
    assert got.loc[("place type", "larger"), "n"] == 2
    assert got.loc[("flag", "blow-up"), "90% coverage"] == 0.0
    assert got.loc[("flag", "collapse"), "log WIS"] == 1.0


def test_check_summary_divides_by_the_baseline_on_the_same_weeks() -> None:
    week = {
        "origin": ORIGIN,
        "truth_vintage": "2026-11-08",
        "location": "wamba",
        "target_end_date": "2026-11-07",
        "scale": "zones",
    }
    scores = pd.DataFrame(
        [
            {
                **week,
                **dict.fromkeys(SCORE_COLUMNS, 0.0),
                "model": "rw",
                "wis": 2.0,
            },
            {
                **week,
                **dict.fromkeys(SCORE_COLUMNS, 0.0),
                "model": "rwn",
                "wis": 6.0,
            },
        ]
    )
    types = pd.DataFrame(
        {"origin": [ORIGIN], "location": ["wamba"], "place_type": ["larger"]}
    )
    # The baseline's forecast is not flagged, the model's is.
    flagged = pd.DataFrame(
        {
            "model": ["rw", "rwn"],
            "origin": ORIGIN,
            "location": "wamba",
            "flag": ["no flag", "collapse"],
        }
    )
    got = check_summary(scores, types, flagged, baseline="rw").set_index(
        ["split", "group", "model"]
    )
    assert got.loc[("flag", "collapse", "rwn"), "relative WIS"] == 3.0
    assert got.loc[("place type", "larger", "rw"), "relative WIS"] == 1.0
