"""Realism summaries of simulated outbreaks against real data."""

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))

from finetune_realism import weekly_summary


def test_zero_share_counts_strata_with_no_cases():
    weeks = pd.date_range("2026-01-03", periods=10, freq="7D")
    weekly = pd.DataFrame(
        {"location": "P01", "week": weeks, "count": [1, 2, 3, 0, 5] * 2}
    )
    # Only one stratum is in the file: its zero weeks are 2 of 10.
    assert weekly_summary(weekly)["zero_share"] == 0.2
    # Four strata, three with no case: 2 + 30 zero weeks of 40.
    places = ["P01", "P02", "P03", "P04"]
    out = weekly_summary(weekly, places)
    assert out["zero_share"] == 32 / 40
