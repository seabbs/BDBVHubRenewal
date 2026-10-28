from __future__ import annotations

import dataclasses

import numpy as np
import pandas as pd
import pytest

from bdbv_hub_renewal.data.delays import ParametricDelay
from bdbv_hub_renewal.data.reconstruct import (
    compare_reconstruction,
    reconstruct_delays,
    reconstruct_vintage,
    reconstruct_vintages,
)
from bdbv_hub_renewal.data.simulate import (
    SimulationSettings,
    gravity_kernel,
    renewal,
    simulate_outbreak,
)


def _latest(df: pd.DataFrame) -> pd.DataFrame:
    return df.loc[df["vintage"] == "2026-10-24"]


def test_reconstruction_far_ahead_recovers_latest(
    target: pd.DataFrame, delays: pd.DataFrame
) -> None:
    # Weeks well before the date are fully reported, so totals match.
    rec = reconstruct_vintage(_latest(target), _latest(delays), "2026-10-24")
    rec = rec.set_index(["location", "target_end_date"])["total"]
    real = _latest(target).set_index(["location", "target_end_date"])["total"]
    old = [i for i in rec.index if i[1] <= "2026-08-01"]
    pd.testing.assert_series_equal(
        rec.loc[old], real.loc[old], check_names=False
    )


def test_reconstruction_is_close_to_real_vintages(
    target: pd.DataFrame, delays: pd.DataFrame
) -> None:
    rec = reconstruct_vintages(target, delays, ["2026-10-09", "2026-10-17"])
    assert set(rec["kind"]) == {"reconstructed"}
    comp = compare_reconstruction(target, rec)
    nat = comp.loc[comp["location"] == "CD"]
    rel = nat["total_difference"].sum() / nat["total_real"].sum()
    assert abs(rel) < 0.06


def test_reconstructed_delays_are_truncated(delays: pd.DataFrame) -> None:
    kept = reconstruct_delays(_latest(delays), "2026-09-05")
    assert (kept["vintage"] == "2026-09-05").all()
    assert kept["onset_week"].max() <= "2026-09-05"
    assert kept["onset_week"].max() >= "2026-08-29"
    assert len(kept) < len(_latest(delays))


def test_renewal_constant_r_one_is_flat() -> None:
    gi = ParametricDelay("gamma", 15, 9).pmf(60)
    days = 300
    seed = np.zeros((1, days))
    seed[0, :60] = 10
    inf, sus = renewal(np.ones((1, days)), gi, seed, np.array([1e12]))
    assert inf[0, 200] == pytest.approx(inf[0, 299], rel=0.02)
    assert sus.min() > 0.999


def test_renewal_depletion_bounds_outbreak() -> None:
    gi = ParametricDelay("gamma", 15, 9).pmf(60)
    seed = np.zeros((1, 400))
    seed[0, 0] = 10
    inf, sus = renewal(np.full((1, 400), 2.0), gi, seed, np.array([1000.0]))
    assert inf.sum() < 1000
    assert sus[0, -1] < 0.5


def test_gravity_kernel_rows_sum_to_one() -> None:
    k = gravity_kernel(("CD-IT", "CD-NK", "CD-HU"))
    assert np.allclose(k.sum(axis=1), 1)
    assert np.allclose(np.diag(k), 0)


@pytest.fixture(scope="module")
def outbreak():  # type: ignore[no-untyped-def]
    return simulate_outbreak(dataclasses.replace(SimulationSettings(), seed=7))


def test_simulated_vintage_is_consistent(outbreak) -> None:  # type: ignore[no-untyped-def]
    tab, dly = outbreak.vintage("2026-09-05")
    assert set(tab["kind"]) == {"simulated"}
    assert (tab["observed"] + tab["imputed"] == tab["total"]).all()
    nat = tab.loc[tab["location"] == "CD"]
    assert nat["observed"].sum() == len(dly)
    assert tab["target_end_date"].max() == "2026-09-05"


def test_simulated_final_vintage_matches_truth_total(outbreak) -> None:  # type: ignore[no-untyped-def]
    tab, _ = outbreak.vintage("2027-06-05")
    truth = outbreak.truth()
    nat_tab = tab.loc[tab["location"] == "CD", "total"].sum()
    nat_truth = truth.loc[truth["location"] == "CD", "truth"].sum()
    # Back-sampling can push a few onsets before the first week.
    assert nat_tab == pytest.approx(nat_truth, rel=0.01)


def test_simulation_parameters_have_turn(outbreak) -> None:  # type: ignore[no-untyped-def]
    p = outbreak.parameters
    it = p.loc[p["location"] == "CD-IT"]
    assert it["rt"].iloc[0] > 1.5
    assert it["rt"].iloc[-1] < 1
    assert it["missing"].max() == pytest.approx(0.25, abs=0.01)


def test_simulated_vintage_publishes_after_a_lag(outbreak) -> None:  # type: ignore[no-untyped-def]
    cases = outbreak.cases
    lag = (cases["publish_date"] - cases["report_date"]).dt.days
    assert lag.between(0, 3).all()
    assert lag.nunique() == 4
    day = pd.Timestamp("2026-09-05")
    _, dly = outbreak.vintage(day.date())
    published = cases.loc[cases["onset_known"] & (cases["publish_date"] <= day)]
    reported = cases.loc[cases["onset_known"] & (cases["report_date"] <= day)]
    assert len(dly) == len(published) < len(reported)


def test_simulated_missing_onsets_follow_the_report_day() -> None:
    s = dataclasses.replace(SimulationSettings(), missing=(0.0, 0.0, 0.9))
    cases = simulate_outbreak(s).cases
    # The missing share rises over the last two thirds, by report day,
    # so cases reported late have no onset more often than their onset
    # day alone implies.
    day = (cases["report_date"] - pd.Timestamp(s.start)).dt.days
    day = day.clip(upper=s.n_days - 1)
    frac = np.clip(day - s.n_days / 3, 0, None) / (s.n_days * 2 / 3)
    expected = np.where(day >= s.n_days / 3, 0.9 * frac, 0.0).mean()
    assert (~cases["onset_known"]).mean() == pytest.approx(expected, abs=0.01)
