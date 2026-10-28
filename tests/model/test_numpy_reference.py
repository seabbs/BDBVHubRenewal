from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from bdbv_hub_renewal.data.delays import ParametricDelay, hub_delay_pmf

from .numpy_reference import (
    delay_matrix,
    expected_imputed_onsets,
    expected_known_onsets,
    expected_missing_by_report,
    to_onset_weeks,
)


def test_known_plus_missing_equals_reports() -> None:
    n = 60
    confirmed = np.linspace(5, 50, n)
    pmfs = np.array(
        [ParametricDelay("gamma", 5 + s / 20, 4).pmf(20) for s in range(n)]
    )
    missing = np.linspace(0.05, 0.3, n)
    cutoff = 45
    known = expected_known_onsets(confirmed, pmfs, missing, cutoff)
    miss = expected_missing_by_report(confirmed, pmfs, missing)
    reports = confirmed @ delay_matrix(pmfs, n)
    assert known.sum() + miss[: cutoff + 1].sum() == pytest.approx(
        reports[: cutoff + 1].sum()
    )


def test_hub_imputation_conserves_cases(delays: pd.DataFrame) -> None:
    pmf = hub_delay_pmf(delays.loc[delays["vintage"] == "2026-10-24"])
    missing = np.zeros(200)
    missing[150:180] = 10
    imputed = expected_imputed_onsets(missing, pmf)
    assert imputed.sum() == pytest.approx(missing.sum())
    # Onsets move earlier than the report days.
    days = np.arange(200)
    assert np.sum(days * imputed) < np.sum(days * missing)


def test_to_onset_weeks() -> None:
    weekly = to_onset_weeks(np.ones(14), "2026-10-04")
    assert list(weekly.index) == ["2026-10-10", "2026-10-17"]
    assert list(weekly) == [7, 7]
