from __future__ import annotations

import dataclasses

import numpy as np
import pandas as pd
import pytest

from bdbv_hub_renewal.data.reconstruct import (
    compare_reconstruction,
    reconstruct_delays,
    reconstruct_vintage,
    reconstruct_vintages,
)
from bdbv_hub_renewal.data.simulate import (
    SimulationSettings,
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
    assert lag.between(0, 60).all()
    # Most cases appear within 3 days, a tail of late entries up to 60.
    assert (lag > 3).mean() == pytest.approx(0.15, abs=0.03)
    assert lag.max() > 40
    day = pd.Timestamp("2026-09-05")
    _, dly = outbreak.vintage(day.date())
    known = cases["onset_known_date"] <= day
    reported = cases["onset_known"] & (cases["report_date"] <= day)
    assert len(dly) == known.sum() < reported.sum()


def test_simulated_missing_onsets_are_backfilled(outbreak) -> None:  # type: ignore[no-untyped-def]
    cases = outbreak.cases
    missing = ~cases["onset_known"]
    filled = missing & cases["onset_known_date"].notna()
    # About a third of cases reported with no onset get one later.
    assert filled.sum() / missing.sum() == pytest.approx(0.3, abs=0.05)
    gap = (
        cases.loc[filled, "onset_known_date"]
        - cases.loc[filled, "publish_date"]
    )
    assert gap.dt.days.between(7, 42).all()
    # So imputed counts of a past week fall and known counts rise.
    early, _ = outbreak.vintage("2026-09-05")
    late, _ = outbreak.vintage("2026-10-10")
    week = (early["location"] == "CD") & (
        early["target_end_date"] == "2026-08-15"
    )
    later = (late["location"] == "CD") & (
        late["target_end_date"] == "2026-08-15"
    )
    assert (
        late.loc[later, "observed"].item() > early.loc[week, "observed"].item()
    )


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
    assert (~cases["onset_known"]).mean() == pytest.approx(expected, abs=0.025)


def test_simulated_onsets_follow_the_model_renewal(outbreak) -> None:  # type: ignore[no-untyped-def]
    import jax.numpy as jnp

    from bdbv_hub_renewal.data import PROVINCE_POPULATIONS
    from bdbv_hub_renewal.model.inputs import gravity_kernel as model_kernel
    from bdbv_hub_renewal.model.model import generation_interval
    from bdbv_hub_renewal.model.renewal import coupled_renewal, mixing_matrix

    s = outbreak.settings
    p = outbreak.parameters
    rt = p.pivot(index="date", columns="location", values="rt")[
        list(s.locations)
    ].to_numpy()
    pop = np.array([PROVINCE_POPULATIONS[q] for q in s.locations])
    table = pd.DataFrame({"location": s.locations, "population": pop})
    w = mixing_matrix(
        jnp.asarray(model_kernel(s.locations, table)),
        s.importation,
        jnp.asarray(pop / pop.sum()),
    )
    seeding = np.zeros_like(rt)
    for i, day in enumerate(s.seed_day):
        if day >= 0:
            seeding[day, i] = s.seed_size / pop[i]
    g = generation_interval(s.gi_mean, s.gi_sd)
    nu = s.effective_fraction
    x, _ = coupled_renewal(
        jnp.zeros((g.size, len(pop))),
        jnp.asarray(rt),
        jnp.asarray(g),
        w,
        jnp.full(len(pop), nu),
        pool=nu,
        seeding=jnp.asarray(seeding),
    )
    onsets = p.pivot(index="date", columns="location", values="onsets")[
        list(s.locations)
    ].to_numpy()
    np.testing.assert_allclose(onsets, np.asarray(x) * pop, rtol=1e-4)


def _rows(
    vintage: str, onset_weeks: list[str], delays: list[int]
) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "vintage": vintage,
            "location": "CD-IT",
            "location_name": "Ituri",
            "onset_week": onset_weeks,
            "delay": delays,
        }
    )


def test_publication_lag_pmf_from_rows_new_in_later_vintages() -> None:
    from bdbv_hub_renewal.data.reconstruct import publication_lag_pmf

    old = _rows("2026-10-10", ["2026-09-05"] * 5, [3] * 5)
    # Two rows reported long ago appear late, three recent ones on time.
    new = pd.concat(
        [
            old.assign(vintage="2026-10-17"),
            _rows("2026-10-17", ["2026-07-04", "2026-07-04"], [2, 2]),
            _rows("2026-10-17", ["2026-10-17"] * 3, [0, 0, 0]),
        ]
    )
    later = _rows("2026-10-24", ["2026-10-24"], [0])
    pmf = publication_lag_pmf(pd.concat([old, new, later]), "2026-10-17")
    assert pmf is not None
    assert pmf.shape == (181,)
    assert pmf.sum() == pytest.approx(1.0)
    # Report dates about 100 days old keep their lag: the real files have
    # 8-9% of new rows more than 60 days late.
    assert pmf[90:110].sum() == pytest.approx(0.4, abs=0.05)
    assert pmf[-1] == 0
    assert pmf[:7].sum() == pytest.approx(0.6, abs=0.05)
    # The 10-24 file is after the forecast date, so it is not used.
    assert publication_lag_pmf(pd.concat([old, new]), "2026-10-17") == (
        pytest.approx(pmf)
    )
    # With no earlier pair there is no estimate.
    assert publication_lag_pmf(pd.concat([old, new]), "2026-10-10") is None
