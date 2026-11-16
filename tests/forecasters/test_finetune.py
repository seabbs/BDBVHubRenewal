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
