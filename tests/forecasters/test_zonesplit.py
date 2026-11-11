from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from bdbvhubrenewal.data.zones import hub_zones, province_of, weekly_zone_counts
from bdbvhubrenewal.evaluation.backtest import BacktestConfig, make_tasks
from bdbvhubrenewal.forecasters import ForecastTask
from bdbvhubrenewal.forecasters import zonerenewal as zr
from bdbvhubrenewal.forecasters import zones as fz
from bdbvhubrenewal.forecasters.zonesplit import ZoneShares

DATA = Path(__file__).resolve().parents[1] / "fixtures" / "data"
ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def task(target: pd.DataFrame, delays: pd.DataFrame) -> ForecastTask:
    rows = target.loc[target["vintage"] == "2026-11-01"]
    return make_tasks(rows, delays, BacktestConfig(n_samples=20, seed=3))[0]


def _provinces(task: ForecastTask) -> pd.DataFrame:
    """Province samples: the latest count, Poisson noise, every week."""
    rng = np.random.default_rng(2)
    rows = []
    for loc in ("CD-IT", "CD-NK", "CD-SK"):
        series = task.target.loc[task.target["location"] == loc, "total"]
        level = float(series.iloc[-1]) if len(series) else 0.0
        for week in task.target_end_dates:
            for k in range(1, task.n_samples + 1):
                rows.append((loc, week, k, float(rng.poisson(level))))
    return pd.DataFrame(
        rows, columns=["location", "target_end_date", "sample", "value"]
    )


def test_zone_mode_reads_booleans_and_modes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv(zr.ZONES_ENV, raising=False)
    assert zr.zone_mode(ROOT / "config" / "hub.toml") == "province"
    config = tmp_path / "hub.toml"
    for value, mode in (
        ("false", "province"),
        ("true", "joint"),
        ('"conditional"', "conditional"),
        ('"joint"', "joint"),
    ):
        config.write_text(f"[renewal]\nzones = {value}\n")
        assert zr.zone_mode(config) == mode
    assert not zr.zone_fits(config.with_name("x.toml"))
    monkeypatch.setenv(zr.ZONES_ENV, "conditional")
    assert zr.zone_mode(config) == "conditional"
    monkeypatch.setenv(zr.ZONES_ENV, "zones")
    with pytest.raises(ValueError, match="must be one of"):
        zr.zone_mode(config)


def test_shares_split_each_province_sample_across_its_zones(
    task: ForecastTask, daily_zones: pd.DataFrame
) -> None:
    counts = weekly_zone_counts(daily_zones, task.vintage)
    zones = tuple(
        z
        for z in hub_zones(DATA / "hub-config" / "tasks.json")
        if province_of(z) in {"CD-IT", "CD-NK", "CD-SK"}
    )
    prov = _provinces(task)
    shares = ZoneShares(num_warmup=60, num_samples=60, num_chains=1)
    out, fits = shares.allocate(prov, task, counts, zones)
    assert set(out["location"]) == set(zones)
    # Zones sum to their province, sample by sample and week by week.
    keys = ["p", "target_end_date", "sample"]
    summed = (
        out.assign(p=out["location"].map(province_of))
        .groupby(keys)["value"]
        .sum()
    )
    want = prov.set_index(["location", "target_end_date", "sample"])["value"]
    np.testing.assert_array_equal(
        summed.reindex(want.index).to_numpy(), want.to_numpy()
    )
    # Ituri and Nord-Kivu have cases, so their shares are fitted, with
    # deviations of their own for the zones with at least 30 cases.
    fitted = {f["province"]: f for f in fits}
    assert {"CD-IT", "CD-NK"} <= set(fitted)
    assert fitted["CD-IT"]["own"] >= 2
    # Most of Ituri goes to the zones with most cases.
    it = out.loc[out["location"].map(province_of) == "CD-IT"]
    top = it.groupby("location")["value"].sum().nlargest(3).index
    big = counts.loc[counts["location"].str.startswith("CD-IT")]
    assert set(top) & set(
        big.groupby("location")["total"].sum().nlargest(5).index
    )


def test_conditional_mode_splits_renewal_samples_by_shares(
    task: ForecastTask,
    daily_zones: pd.DataFrame,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []

    def fake(
        _self: ZoneShares, samples: pd.DataFrame, *_: object
    ) -> tuple[pd.DataFrame, list[dict[str, object]]]:
        calls.append("shares")
        return samples.iloc[0:0], []

    monkeypatch.setattr(ZoneShares, "allocate", fake)
    counts = weekly_zone_counts(daily_zones, task.vintage)
    zones = hub_zones(DATA / "hub-config" / "tasks.json")
    prov = _provinces(task)
    monkeypatch.setenv(zr.ZONES_ENV, "conditional")
    fz.zones_for_model("renewalbvd", task, prov, counts, zones, daily_zones)
    assert calls == ["shares"]
    # Other models keep the allocator.
    fz.zones_for_model("renewalsp", task, prov, counts, zones, daily_zones)
    assert calls == ["shares"]
    monkeypatch.setenv(zr.ZONES_ENV, "province")
    fz.zones_for_model("renewalbvd", task, prov, counts, zones, daily_zones)
    assert calls == ["shares"]
