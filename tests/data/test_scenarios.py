"""Simulated scenarios calibrated to the real data."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from bdbvhubrenewal.data import load_health_zones
from bdbvhubrenewal.data.simulate import (
    calibrated_settings,
    simulate_outbreak,
)

DATA = Path(__file__).resolve().parents[2] / "data"
LATEST = "2026-10-24"


@pytest.fixture(scope="module")
def zones() -> pd.DataFrame:
    return load_health_zones(DATA)


@pytest.fixture(scope="module")
def outbreak(target, delays, zones):  # type: ignore[no-untyped-def]
    s = calibrated_settings(target, delays, zones, LATEST, name="flat")
    return simulate_outbreak(s)


def _weekly_real(target: pd.DataFrame) -> pd.DataFrame:
    rows = target.loc[
        (target["vintage"] == LATEST)
        & (target["kind"] == "real")
        & (target["location"] != "CD")
    ]
    return rows.pivot_table(
        index="target_end_date", columns="location", values="total"
    ).fillna(0)


def test_calibrated_onsets_follow_the_real_provinces(outbreak, target):  # type: ignore[no-untyped-def]
    real = _weekly_real(target)
    s = outbreak.settings
    assert set(s.locations) == set(real.columns[real.sum() > 0])
    truth = outbreak.truth().pivot_table(
        index="target_end_date", columns="location", values="truth"
    )
    # Weeks reported in full (6+ weeks old) match the real totals: sizes
    # and timing of every province.
    old = [w for w in real.index if w <= "2026-09-12" and w in truth.index]
    for loc in s.locations:
        sim, obs = truth.loc[old, loc].fillna(0), real.loc[old, loc]
        assert sim.sum() == pytest.approx(obs.sum(), rel=0.15, abs=10)
        if obs.sum() > 50:
            assert np.argmax(sim.to_numpy()) == pytest.approx(
                np.argmax(obs.to_numpy()), abs=2
            )


def test_calibrated_delays_and_missingness(outbreak, delays, target):  # type: ignore[no-untyped-def]
    cases = outbreak.cases
    delay = (cases["report_date"] - cases["onset_date"]).dt.days
    real = delays.loc[delays["vintage"] == LATEST, "delay"].clip(0, 60)
    assert delay.mean() == pytest.approx(real.mean(), rel=0.2)
    rows = target.loc[
        (target["vintage"] == LATEST) & (target["location"] == "CD")
    ]
    share = rows["imputed"].sum() / rows["total"].sum()
    sim, _ = outbreak.vintage(LATEST)
    nat = sim.loc[sim["location"] == "CD"]
    assert nat["imputed"].sum() / nat["total"].sum() == pytest.approx(
        share, abs=0.05
    )


def test_scenarios_differ_after_the_data(target, delays, zones):  # type: ignore[no-untyped-def]
    ends = {}
    for name in ("falling", "flat", "rising"):
        s = calibrated_settings(target, delays, zones, LATEST, name=name)
        truth = simulate_outbreak(s).truth()
        nat = truth.loc[truth["location"] == "CD"].set_index("target_end_date")[
            "truth"
        ]
        ends[name] = nat.iloc[-4:].sum()
    assert ends["falling"] < ends["flat"] < ends["rising"]


def test_downward_revisions(outbreak):  # type: ignore[no-untyped-def]
    a, _ = outbreak.vintage("2026-10-10")
    b, _ = outbreak.vintage("2026-10-24")
    key = ["location", "target_end_date"]
    both = a.merge(b, on=key, suffixes=("_a", "_b"))
    old = both.loc[both["target_end_date"] <= "2026-09-12"]
    assert (old["total_b"] < old["total_a"]).any()
    assert outbreak.cases["removed_date"].notna().mean() == pytest.approx(
        outbreak.settings.removal_share, abs=0.01
    )


def test_zone_totals_add_up_to_provinces(outbreak, zones):  # type: ignore[no-untyped-def]
    prov, _ = outbreak.vintage(LATEST)
    zone = outbreak.zone_vintage(LATEST)
    assert set(zone.columns) >= {"location", "province", "target_end_date"}
    summed = zone.groupby(["province", "target_end_date"])["total"].sum()
    p = prov.loc[prov["location"] != "CD"].set_index(
        ["location", "target_end_date"]
    )["total"]
    p.index.names = ["province", "target_end_date"]
    joined = pd.concat([summed, p], axis=1, keys=["zones", "prov"]).fillna(0)
    np.testing.assert_array_equal(joined["zones"], joined["prov"])
    # Zones come from the real zones of each province.
    real = set(zones["location"])
    assert set(zone["location"]) <= real
