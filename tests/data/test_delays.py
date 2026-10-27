from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from bdbv_hub_renewal.data.delays import (
    EmpiricalDelayModel,
    ParametricDelay,
    delay_summary,
    empirical_pmf,
)


def test_empirical_pmf_drops_out_of_range() -> None:
    pmf = empirical_pmf([-1, 0, 1, 1, 100], max_delay=5)
    assert pmf.sum() == pytest.approx(1)
    assert pmf[1] == pytest.approx(2 / 3)


def test_empirical_model_shrinks_small_provinces(delays: pd.DataFrame) -> None:
    latest = delays.loc[delays["vintage"] == "2026-10-24"]
    fit = EmpiricalDelayModel(window_weeks=5, prior_weight=50).fit(
        latest, "2026-10-24"
    )
    small = fit.pmf(
        "CD-BU",
        sorted(latest.loc[latest["location"] == "CD-BU", "onset_week"])[0],
    )
    assert small.sum() == pytest.approx(1)
    # Unknown cells fall back to the pooled pmf.
    assert np.array_equal(fit.pmf("CD-KN", "2026-01-03"), fit.pooled)
    assert 3 <= fit.quantile(0.5) <= 7


def test_delay_summary(delays: pd.DataFrame) -> None:
    s = delay_summary(delays, by=("vintage",))
    assert list(s.columns) == ["vintage", "n", "mean", "median", "q90"]


@pytest.mark.parametrize("family", ["gamma", "lognormal", "weibull"])
def test_parametric_delay_mean(family: str) -> None:
    pmf = ParametricDelay(family, 6.0, 4.0).pmf(max_delay=80)  # type: ignore[arg-type]
    assert pmf.sum() == pytest.approx(1)
    # Daily censoring of a uniform onset adds about half a day.
    mean = np.sum(np.arange(pmf.size) * pmf)
    assert mean == pytest.approx(6.0, abs=0.7)
    weekly = ParametricDelay(family, 6.0, 4.0).pmf(80, primary_window=7)  # type: ignore[arg-type]
    assert np.sum(np.arange(weekly.size) * weekly) > mean
