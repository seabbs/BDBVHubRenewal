from __future__ import annotations

import dataclasses

import numpy as np
import pandas as pd
import pytest

from bdbvhubrenewal.data.dates import epiweek_end
from bdbvhubrenewal.data.releases import cutoff_lags
from bdbvhubrenewal.data.simulate import (
    SimulationSettings,
    release_dates,
    simulate_outbreak,
)


def test_cutoff_lags_follow_real_releases() -> None:
    daily = pd.DataFrame(
        {
            "vintage": ["2026-10-09", "2026-10-09", "2026-10-17"],
            "onset_date": ["2026-10-06", "2026-10-07", "2026-10-16"],
        }
    )
    lags = cutoff_lags(daily)
    assert lags[4] == 2
    assert lags[5] == 1
    assert lags[1] == 1


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

    from bdbvhubrenewal.data import PROVINCE_POPULATIONS
    from bdbvhubrenewal.model.inputs import gravity_kernel as model_kernel
    from bdbvhubrenewal.model.model import generation_interval
    from bdbvhubrenewal.model.renewal import coupled_renewal, mixing_matrix

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
    x, _, _ = coupled_renewal(
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


def test_unused_options_leave_the_default_outbreak_unchanged() -> None:
    # The model recovery tests are calibrated on this realisation, so
    # options that are off (removals, zones) must not draw random numbers.
    outbreak = simulate_outbreak()
    assert len(outbreak.cases) == 1797
    target, _ = outbreak.vintage("2026-10-28")
    assert int(target["imputed"].sum()) == 488


def test_release_dates_are_the_real_weekdays() -> None:
    days = release_dates("2026-10-05", "2026-10-11")
    assert days == ["2026-10-06", "2026-10-07", "2026-10-09", "2026-10-10"]


@pytest.mark.parametrize("cutoff_lag", [0, 2])
def test_simulated_daily_vintage_sums_to_weekly(
    outbreak,
    cutoff_lag: int,  # type: ignore[no-untyped-def]
) -> None:
    day = "2026-09-04"
    weekly, _ = outbreak.vintage(day, cutoff_lag=cutoff_lag)
    daily = outbreak.daily_vintage(day, cutoff_lag=cutoff_lag)
    assert set(daily["kind"]) == {"simulated"}
    cut = (pd.Timestamp(day) - pd.Timedelta(days=cutoff_lag)).date()
    assert daily["cutoff"].iloc[0] == cut.isoformat()
    assert daily["onset_date"].max() <= daily["cutoff"].iloc[0]
    week = daily["onset_date"].map(lambda x: epiweek_end(x).isoformat())
    got = (
        daily.assign(target_end_date=week)
        .groupby(["location", "target_end_date"])[["observed", "imputed"]]
        .sum()
    )
    want = weekly.set_index(["location", "target_end_date"])[
        ["observed", "imputed"]
    ]
    want = want.loc[want.sum(axis=1) > 0]
    pd.testing.assert_frame_equal(
        got.sort_index(), want.sort_index(), check_dtype=False
    )


def test_simulated_cutoffs_follow_weekday(outbreak) -> None:  # type: ignore[no-untyped-def]
    dates = release_dates("2026-09-01", "2026-09-05")
    out = outbreak.daily_vintages(dates, cutoff_lags={1: 0, 2: 1, 4: 2, 5: 1})
    cut = out.groupby("vintage")["cutoff"].first().to_dict()
    assert cut == {
        "2026-09-01": "2026-09-01",
        "2026-09-02": "2026-09-01",
        "2026-09-04": "2026-09-02",
        "2026-09-05": "2026-09-04",
    }
