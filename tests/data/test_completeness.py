from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from bdbvhubrenewal.data.completeness import (
    complete_weeks,
    reported_share,
    share_from_cdf,
    truncated_delay_cdf,
)
from bdbvhubrenewal.evaluation.backtest import load_config


def _delays(delay: int, n: int = 20) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "vintage": "2026-10-24",
            "location": "CD-IT",
            "onset_week": "2026-06-06",
            "delay": [delay] * n,
        }
    )


def test_reported_share_with_a_fixed_delay() -> None:
    # Onset is spread over the 7 days of the week, so a 10-day delay is
    # reported 4 to 10 days after the week ends.
    d = _delays(10)
    share = reported_share(d, "2026-10-24", [3, 7, 10, 14], lag=0)
    assert share == pytest.approx([0.0, 4 / 7, 1.0, 1.0])
    # A publication lag delays every case by that many days.
    assert reported_share(d, "2026-10-24", [10], lag=3)[0] == pytest.approx(
        4 / 7
    )


def test_reported_share_uses_settled_onset_weeks_only() -> None:
    # Recent onset weeks only show short delays so far, so they are left out.
    old = _delays(30)
    recent = _delays(0).assign(onset_week="2026-10-17")
    share = reported_share(pd.concat([old, recent]), "2026-10-24", [7], lag=0)
    assert share[0] == 0.0


def test_complete_weeks() -> None:
    assert complete_weeks(_delays(0), "2026-10-24", lag=0) == 0
    assert complete_weeks(_delays(10), "2026-10-24", lag=0) == 2
    assert complete_weeks(_delays(10), "2026-10-24", lag=3) == 2


def test_complete_weeks_matches_the_config(delays: pd.DataFrame) -> None:
    # The configured K is the one the latest delay data give.
    latest = str(delays["vintage"].max())
    assert complete_weeks(delays, latest) == load_config().complete_after_weeks


def _truncated(rng: np.random.Generator, vintage: str) -> pd.DataFrame:
    """Delay rows of a steady outbreak, right truncated at the vintage."""
    end = pd.Timestamp(vintage)
    onset = end - pd.to_timedelta(rng.integers(0, 120, 40000), unit="D")
    delay = rng.geometric(1 / 8, onset.size) - 1
    seen = onset + pd.to_timedelta(delay + 3, unit="D") <= end
    week = onset + pd.to_timedelta((5 - onset.dayofweek) % 7, unit="D")
    return pd.DataFrame(
        {
            "vintage": vintage,
            "location": "CD-IT",
            "onset_week": week[seen].strftime("%Y-%m-%d"),
            "delay": delay[seen],
        }
    )


def test_truncated_delay_cdf_with_a_fixed_delay() -> None:
    d = _delays(10).assign(onset_week="2026-09-26")
    cdf = truncated_delay_cdf(d, "2026-10-24", lag=0)
    assert cdf is not None
    assert cdf[9] == 0.0
    assert cdf[10] == 1.0
    share = share_from_cdf(cdf, [3, 7, 10, 14], lag=0)
    assert share == pytest.approx([0.0, 4 / 7, 1.0, 1.0])


def test_truncated_delay_cdf_corrects_right_truncation() -> None:
    rng = np.random.default_rng(2)
    vintage = "2026-10-24"
    rows = _truncated(rng, vintage)
    cdf = truncated_delay_cdf(rows, vintage, window=84, lag=3)
    assert cdf is not None
    true = 1 - (7 / 8) ** (np.arange(cdf.size) + 1)
    # Within a point of the true cdf at the delays that matter.
    assert np.abs(cdf[:21] - true[:21]).max() < 0.01
    # The naive cdf of the same rows is biased short.
    recent = pd.to_datetime(rows["onset_week"]) > pd.Timestamp(vintage) - (
        pd.Timedelta(days=84)
    )
    naive = np.bincount(rows.loc[recent, "delay"].clip(upper=60), minlength=61)
    assert (naive.cumsum() / naive.sum())[7] > true[7] + 0.03


def test_truncated_delay_cdf_needs_rows_in_the_window() -> None:
    assert truncated_delay_cdf(_delays(5), "2026-10-24", window=84) is None
