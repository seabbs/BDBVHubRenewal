from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from bdbvhubrenewal.data.delays import (
    HUB_GENGAMMA,
    EmpiricalDelayModel,
    delay_summary,
    empirical_pmf,
    hub_delay_pmf,
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


@pytest.mark.parametrize(
    ("release", "fit"),
    [("2026-10-24", "exact"), ("2026-11-01", "double_censored")],
)
def test_hub_delay_pmf_matches_hub_sampling(release: str, fit: str) -> None:
    # The hub's draw: report hour Unif(0, 24) minus a generalised gamma
    # delay in hours (flexsurv's rgengamma for Q > 0), floored to days.
    mu, sigma, q = HUB_GENGAMMA[fit]
    rng = np.random.default_rng(1)
    n = 1_000_000
    gamma = rng.gamma(q**-2, size=n)
    hours = np.exp(mu + sigma * np.log(q**2 * gamma) / q)
    days = -np.floor((rng.uniform(0, 24, n) - hours) / 24).astype(int)
    pmf = hub_delay_pmf(release)
    np.testing.assert_allclose(pmf, empirical_pmf(days), atol=2e-3)
    assert pmf.sum() == pytest.approx(1)


def test_hub_delay_pmf_switches_at_double_censored_fit() -> None:
    # The double censored fit no longer counts day rounding twice, so
    # fewer imputed onsets fall on the report day.
    assert hub_delay_pmf("2026-10-24")[0] == pytest.approx(0.0717, abs=1e-3)
    assert hub_delay_pmf("2026-11-01")[0] == pytest.approx(0.0408, abs=1e-3)
