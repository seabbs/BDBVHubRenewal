from __future__ import annotations

import pandas as pd
import pytest

from bdbvhubrenewal.data.completeness import complete_weeks, reported_share
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
