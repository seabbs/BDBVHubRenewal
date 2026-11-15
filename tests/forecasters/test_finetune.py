"""Training blocks for fine-tuning Chronos-2 on simulated outbreaks."""

from __future__ import annotations

import numpy as np
import pandas as pd

from bdbvhubrenewal.forecasters import NATIONAL
from bdbvhubrenewal.forecasters.finetune import (
    final_counts,
    releases,
    training_block,
    weekly_cutoffs,
)
from bdbvhubrenewal.forecasters.foundation import hub_triangle


def _counts() -> pd.DataFrame:
    """Two places' triangle: onsets 10-05 to 10-14, reported 0 to 9 days on."""
    rows = []
    for place, scale in (("A", 1), ("B", 2)):
        for onset in pd.date_range("2026-10-05", "2026-10-14"):
            for delay in (0, 3, 9):
                report = onset + pd.Timedelta(days=delay)
                rows.append((place, onset, report, scale))
    return pd.DataFrame(
        rows, columns=["location", "reference_date", "report_date", "count"]
    )


def test_hub_triangle_takes_max_lag() -> None:
    rows = releases(_counts(), "weekly", ["2026-10-17", "2026-10-24"])
    tri = hub_triangle(rows, "weekly", (NATIONAL,), "2026-10-31", max_lag=2)
    assert tri.partial.shape == (1, 4, 2)
    assert tri.n_context == tri.last - 1


def test_releases_count_what_was_reported_by_each_cutoff() -> None:
    rows = releases(_counts(), "daily", ["2026-10-07", "2026-10-30"])
    first = rows.loc[
        (rows["vintage"] == "2026-10-07") & (rows["location"] == "A")
    ]
    # Onset 10-05 had its delay-0 report by 10-07, 10-07 its delay-0.
    assert first.set_index("onset_date")["observed"].to_dict() == {
        "2026-10-05": 1,
        "2026-10-06": 1,
        "2026-10-07": 1,
    }
    last = rows.loc[rows["vintage"] == "2026-10-30"]
    # The country is the sum of the places, every report in.
    country = last.loc[last["location"] == NATIONAL, "observed"]
    assert (country == 9).all()
    assert (rows["imputed"] == 0).all()


def test_weekly_releases_are_by_onset_week() -> None:
    rows = releases(_counts(), "weekly", ["2026-10-30"])
    a = rows.loc[rows["location"] == "A"].set_index("target_end_date")
    # Onsets 10-05 to 10-10 fall in the week ending Saturday 10-10.
    assert a["total"].to_dict() == {"2026-10-10": 18, "2026-10-17": 12}


def test_final_counts_line_up_with_the_grid() -> None:
    out = final_counts(
        _counts(), "daily", (NATIONAL, "A", "B"), ["2026-10-04", "2026-10-05"]
    )
    assert out.tolist() == [[0.0, 9.0], [0.0, 3.0], [0.0, 6.0]]


def test_cutoffs_fall_on_the_weekday() -> None:
    assert weekly_cutoffs("2026-10-05", "2026-10-22", 3) == [
        "2026-10-08",
        "2026-10-15",
        "2026-10-22",
    ]


def test_block_is_the_ragged_layout_with_final_counts_ahead() -> None:
    places = (NATIONAL, "A", "B")
    cutoffs = ["2026-10-07", "2026-10-09"]
    block = training_block(_counts(), "daily", places, cutoffs, max_lag=3)
    start, horizon = block["start"], block["horizon"]
    context = block["context"]
    assert block["n_targets"] == 3
    assert block["n_covariates"] == block["n_future_covariates"] == 9
    assert context.shape == (12, start + horizon)
    # Cut-off 10-09 with D = 3 gives context to 10-06 from onset 10-05.
    assert start == 2
    # The horizon runs to the Saturday four weeks after the cut-off's week.
    assert (
        horizon
        == (pd.Timestamp("2026-11-07") - pd.Timestamp("2026-10-07")).days + 1
    )
    # Targets past the context are the final counts.
    assert context[0, start : start + 8].tolist() == [9.0] * 8
    assert np.all(context[0, start + 8 :] == 0.0)
    # Lag 0 of the country at 10-09 is the cut-off's same-day report.
    assert context[3, 4] == 3.0
    # Lag 0 at 10-08 is missing: no release had its cut-off on 10-08.
    assert np.isnan(context[3, 3])
    assert np.array_equal(
        block["future_covariates"], context[:, start:], equal_nan=True
    )


def test_releases_add_each_place_total_of_its_strata() -> None:
    counts = _counts()
    counts["location"] = counts["location"].map({"A": "P1-A1", "B": "P1-A2"})
    rows = releases(counts, "daily", ["2026-10-30"], totals=True)
    by = rows.groupby("location")["observed"].sum()
    # P1 is the sum of its two strata, and the country counts them once.
    assert by["P1"] == by["P1-A1"] + by["P1-A2"]
    assert by[NATIONAL] == by["P1"]


def test_three_level_block_has_country_place_and_strata() -> None:
    counts = _counts()
    counts["location"] = counts["location"].map({"A": "P1-A1", "B": "P1-A2"})
    cutoffs = weekly_cutoffs("2026-10-05", "2026-10-21", 2)
    places = (NATIONAL, "P1", "P1-A1", "P1-A2")
    block = training_block(
        counts, "daily", places, cutoffs, max_lag=3, totals=True
    )
    assert block["n_targets"] == 4
    start = block["start"]
    target = block["context"][:4, start:]
    # The country and P1 are both the sum of the strata.
    np.testing.assert_array_equal(target[0], target[1])
    np.testing.assert_array_equal(target[1], target[2] + target[3])


def test_totals_sum_strata_by_place_and_by_age_exactly() -> None:
    counts = _counts()
    counts = pd.concat(
        [
            counts.assign(
                location=counts["location"].map({"A": "P1-A1", "B": "P1-A2"})
            ),
            counts.assign(
                location=counts["location"].map({"A": "P2-A1", "B": "P2-A2"})
            ),
        ]
    )
    rows = releases(counts, "daily", ["2026-10-09", "2026-10-30"], totals=True)
    wide = rows.pivot_table(
        index=["vintage", "onset_date"],
        columns="location",
        values="observed",
        aggfunc="sum",
        fill_value=0,
    )
    leaves = ["P1-A1", "P1-A2", "P2-A1", "P2-A2"]
    assert (wide[NATIONAL] == wide[leaves].sum(axis=1)).all()
    assert (wide["P1"] == wide["P1-A1"] + wide["P1-A2"]).all()
    assert (wide["A1"] == wide["P1-A1"] + wide["P2-A1"]).all()


def test_regions_of_places_total_their_places_exactly() -> None:
    counts = _counts()
    counts["location"] = counts["location"].map({"A": "R1-P01", "B": "R1-P02"})
    rows = releases(counts, "weekly", ["2026-10-30"], totals=True)
    wide = rows.pivot_table(
        index="target_end_date",
        columns="location",
        values="total",
        aggfunc="sum",
        fill_value=0,
    )
    assert (wide["R1"] == wide["R1-P01"] + wide["R1-P02"]).all()
    # A region's places are not age groups: no total by their second part.
    assert "P01" not in wide.columns


def test_block_of_places_with_no_release_rows_is_none() -> None:
    # Only place A had cases, so its releases hold the country alone.
    counts = _counts().query("location == 'A'")
    cutoffs = ["2026-10-17", "2026-10-24"]
    assert training_block(counts, "daily", ("A",), cutoffs, max_lag=3) is None
    assert training_block(counts, "daily", ("B",), cutoffs, max_lag=3) is None
