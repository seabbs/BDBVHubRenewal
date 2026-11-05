from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from bdbvhubrenewal.reporting.lag_comparison import (
    calibration_table,
    late_reports,
    newest_weeks,
    plot_learnt_lag,
)


def test_late_reports_counts_rows_reported_in_time_but_published_later():
    # Cut-offs 10-07 and 10-16. A row reported on 10-07 (week ending
    # 10-03, delay 4) is in the first file, another only in the second.
    rows = {"location": "CD-IT", "onset_week": "2026-10-03", "delay": 4}
    delays = pd.DataFrame(
        [
            {**rows, "vintage": "2026-10-09"},
            {**rows, "vintage": "2026-10-17"},
            {**rows, "vintage": "2026-10-17"},
        ]
    )
    daily = pd.DataFrame(
        {
            "vintage": ["2026-10-09", "2026-10-17"],
            "onset_date": ["2026-10-07", "2026-10-16"],
        }
    )
    out = late_reports(delays, daily).set_index(["pair", "days"])
    cell = out.loc[("2026-10-09 to 2026-10-17", "0")]
    assert (cell["published"], cell["late"]) == (1, 1)
    assert cell["share_late"] == 0.5
    assert out.loc[("All", "0"), "late"] == 1


def test_newest_weeks_flags_truths_above_the_interval():
    quantiles = pd.DataFrame(
        {
            "model": "renewal",
            "origin": "2026-10-17",
            "target": "reported",
            "truth_vintage": "2026-10-24",
            "location": "CD",
            "target_end_date": ["2026-10-10", "2026-10-03"],
            "weeks_back": [2, 3],
            "q5": 10.0,
            "q25": 15.0,
            "q50": 20.0,
            "q75": 25.0,
            "q95": 30.0,
            "reported": 12.0,
            "truth": [40.0, 20.0],
        }
    )
    rows = newest_weeks(quantiles)
    assert rows["above_90"].tolist() == [True, False]
    table = calibration_table(rows, ())
    assert table.loc[0, "coverage_90"] == 0.5
    assert table.loc[0, "added"] == 36
    assert table.loc[0, "added_median"] == 16
    assert table.loc[0, "width_90"] == 1.0


def test_learnt_lag_is_drawn_for_each_origin_beside_the_fixed_lag():
    lags = np.arange(15)
    pmf = np.exp(-lags / 2) / np.exp(-lags / 2).sum()
    table = pd.concat(
        pd.DataFrame(
            {"vintage": v, "lag": lags, "q5": pmf / 2, "q50": pmf, "q95": pmf}
        )
        for v in ("2026-10-17", "2026-10-24")
    )
    fig = plot_learnt_lag(table)
    labels = [line.get_label() for line in fig.axes[0].lines]
    assert labels == [
        "renewallearntlag, origin 2026-10-17",
        "renewallearntlag, origin 2026-10-24",
        "renewal, uniform 0 to 3",
    ]
    plt.close(fig)
