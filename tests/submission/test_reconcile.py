from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from bdbvhubrenewal.forecasters.zones import check_zone_coherence
from bdbvhubrenewal.submission.reconcile import (
    Reconciliation,
    reconcile,
    reconciliation_settings,
)

ROOT = Path(__file__).resolve().parents[2]
WEEKS = ["2026-10-31", "2026-11-07", "2026-11-14"]


def _samples(n: int = 50, seed: int = 1) -> pd.DataFrame:
    """Province trajectories with each province's draws independent."""
    rng = np.random.default_rng(seed)
    rows = []
    for loc, scale in (("CD-IT", 200.0), ("CD-NK", 100.0), ("CD-HU", 30.0)):
        level = rng.lognormal(np.log(scale), 0.5, n)
        for s in range(n):
            for w, week in enumerate(WEEKS):
                value = round(level[s] * (1 + 0.1 * w))
                rows.append((loc, week, s + 1, float(value)))
    rows += [("CD-BU", w, s + 1, 0.0) for s in range(n) for w in WEEKS]
    return pd.DataFrame(
        rows, columns=["location", "target_end_date", "sample", "value"]
    )


def _zones(samples: pd.DataFrame) -> pd.DataFrame:
    """Split Ituri into two zones, 60:40 in whole numbers."""
    it = samples.loc[samples["location"] == "CD-IT"]
    a = it.assign(location="CD-IT-AA", value=np.floor(it["value"] * 0.6))
    b = it.assign(
        location="CD-IT-BB", value=it["value"].to_numpy() - a["value"]
    )
    return pd.concat([a, b], ignore_index=True)


def _trajectories(samples: pd.DataFrame, loc: str) -> set[tuple[float, ...]]:
    rows = samples.loc[samples["location"] == loc]
    wide = rows.pivot(index="sample", columns="target_end_date", values="value")
    return {tuple(r) for r in wide.to_numpy()}


def _totals(samples: pd.DataFrame, loc: str) -> pd.Series:
    rows = samples.loc[samples["location"] == loc]
    return rows.groupby("sample")["value"].sum()


def test_the_default_is_the_sum() -> None:
    settings = reconciliation_settings(ROOT / "config" / "hub.toml")
    assert settings.method == "sum"
    assert settings.zone_pairing == "sample"


def test_sum_leaves_the_samples_unchanged() -> None:
    samples = _samples()
    zones = _zones(samples)
    out, out_zones = reconcile(samples, zones, Reconciliation("sum"), seed=1)
    pd.testing.assert_frame_equal(out, samples)
    pd.testing.assert_frame_equal(out_zones, zones)


def test_copula_keeps_each_province_trajectories() -> None:
    samples = _samples()
    out, _ = reconcile(samples, None, Reconciliation("copula", 0.5), seed=1)
    for loc in ("CD-IT", "CD-NK", "CD-HU", "CD-BU"):
        assert _trajectories(out, loc) == _trajectories(samples, loc)
    assert len(out) == len(samples)


def test_copula_of_one_makes_provinces_comonotone() -> None:
    samples = _samples()
    out, _ = reconcile(samples, None, Reconciliation("copula", 1.0), seed=1)
    it, nk = _totals(out, "CD-IT"), _totals(out, "CD-NK")
    assert it.rank().corr(nk.rank()) > 0.99
    before = _totals(samples, "CD-IT").corr(_totals(samples, "CD-NK"))
    assert before < 0.5


def test_copula_widens_the_national_sum() -> None:
    samples = _samples(n=400)
    out, _ = reconcile(samples, None, Reconciliation("copula", 0.8), seed=2)

    def national_sd(table: pd.DataFrame) -> float:
        week = table.loc[table["target_end_date"] == WEEKS[0]]
        return float(week.groupby("sample")["value"].sum().std())

    assert national_sd(out) > 1.2 * national_sd(samples)


def test_zones_move_with_their_province() -> None:
    samples = _samples()
    zones = _zones(samples)
    out, out_zones = reconcile(
        samples, zones, Reconciliation("copula", 0.7), seed=3
    )
    assert not check_zone_coherence(out, out_zones)
    for z in ("CD-IT-AA", "CD-IT-BB"):
        assert _trajectories(out_zones, z) == _trajectories(zones, z)


def test_the_same_seed_gives_the_same_order() -> None:
    samples = _samples()
    one, _ = reconcile(samples, None, Reconciliation("copula", 0.3), seed=4)
    two, _ = reconcile(samples, None, Reconciliation("copula", 0.3), seed=4)
    pd.testing.assert_frame_equal(one, two)


@pytest.mark.parametrize(
    "bad",
    [
        {"method": "mint"},
        {"method": "copula", "rho": 1.5},
        {"zone_pairing": "random"},
    ],
)
def test_bad_settings_raise(bad: dict[str, object]) -> None:
    with pytest.raises(ValueError, match="must be"):
        Reconciliation(**bad)  # type: ignore[arg-type]
