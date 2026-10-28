from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from bdbvhubrenewal.data import load_daily_zones, load_health_zones
from bdbvhubrenewal.data.zones import (
    hub_zones,
    province_of,
    weekly_zone_counts,
)
from bdbvhubrenewal.evaluation.backtest import BacktestConfig, make_tasks
from bdbvhubrenewal.forecasters import ForecastTask, Persistence
from bdbvhubrenewal.forecasters.zones import (
    ZoneAllocator,
    check_zone_coherence,
    zone_task,
)
from bdbvhubrenewal.submission.hub import (
    load_hub_round,
    to_hub_table,
    validate_submission_file,
    write_submission,
)

DATA = Path(__file__).resolve().parents[2] / "data"


@pytest.fixture(scope="module")
def daily_zones() -> pd.DataFrame:
    return load_daily_zones(DATA)


@pytest.fixture(scope="module")
def zones() -> tuple[str, ...]:
    return hub_zones(DATA / "hub-config" / "tasks.json")


@pytest.fixture
def task(target: pd.DataFrame, delays: pd.DataFrame) -> ForecastTask:
    rows = target.loc[target["vintage"] == "2026-10-17"]
    return make_tasks(rows, delays, BacktestConfig(seed=3))[0]


@pytest.fixture
def counts(daily_zones: pd.DataFrame) -> pd.DataFrame:
    return weekly_zone_counts(daily_zones, "2026-10-17")


def _alloc(
    task: ForecastTask,
    counts: pd.DataFrame,
    zones: tuple[str, ...],
    samples: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    prov = Persistence().forecast(task) if samples is None else samples
    return prov, ZoneAllocator().allocate(prov, task, counts, zones)


def test_hub_zones_are_the_519_zone_codes(zones: tuple[str, ...]) -> None:
    assert len(zones) == 519
    assert all(len(z.split("-")) == 3 for z in zones)
    assert province_of("CD-IT-10") == "CD-IT"


def test_weekly_zone_counts_match_the_hub_time_series(
    daily_zones: pd.DataFrame,
) -> None:
    hz = load_health_zones(DATA)
    for vintage in hz["vintage"].unique():
        got = weekly_zone_counts(daily_zones, str(vintage))
        want = hz.loc[hz["vintage"] == vintage]
        both = got.merge(
            want, on=["location", "target_end_date"], how="outer"
        ).fillna(0)
        assert (both["total_x"] == both["total_y"]).all()
        assert set(got.columns) == {"location", "target_end_date", "total"}


def test_zone_samples_sum_to_the_province_samples(
    task: ForecastTask, counts: pd.DataFrame, zones: tuple[str, ...]
) -> None:
    prov, alloc = _alloc(task, counts, zones)
    assert check_zone_coherence(prov, alloc) == []
    summed = (
        alloc.assign(location=alloc["location"].map(province_of))
        .groupby(["location", "target_end_date", "sample"])["value"]
        .sum()
    )
    want = prov.set_index(["location", "target_end_date", "sample"])["value"]
    assert (summed.reindex(want.index) == want.clip(lower=0).round()).all()


def test_every_zone_of_a_forecast_province_is_covered(
    task: ForecastTask, counts: pd.DataFrame, zones: tuple[str, ...]
) -> None:
    _, alloc = _alloc(task, counts, zones)
    want = {z for z in zones if province_of(z) in task.locations}
    assert set(alloc["location"]) == want
    assert len(want) > 150  # 8 provinces with cases hold 178 zones
    per = alloc.groupby("location")["sample"].nunique()
    assert (per == task.n_samples).all()
    weeks = alloc.groupby("location")["target_end_date"].nunique()
    assert (weeks == len(task.target_end_dates)).all()
    assert (alloc["value"] >= 0).all()
    assert (alloc["value"] == alloc["value"].round()).all()


def test_published_zone_counts_are_kept(
    task: ForecastTask, counts: pd.DataFrame, zones: tuple[str, ...]
) -> None:
    prov, alloc = _alloc(task, counts, zones)
    seen = counts.set_index(["location", "target_end_date"])["total"]
    both = alloc.join(seen, on=["location", "target_end_date"]).fillna(0)
    # Persistence copies complete weeks, so the zones copy theirs too.
    old = both.loc[both["target_end_date"] == "2026-09-13"]
    assert (old["value"] == old["total"]).all()
    # A zone never falls below what it has published, unless its
    # province sample is below the province's published total.
    published = (
        counts.assign(location=counts["location"].map(province_of))
        .groupby(["location", "target_end_date"])["total"]
        .sum()
        .rename("published")
    )
    keys = ["location", "target_end_date"]
    prov = prov.join(published, on=keys).fillna({"published": 0})
    covered = prov.loc[prov["value"] >= prov["published"]]
    both["province"] = both["location"].map(province_of)
    both = both.merge(
        covered[[*keys, "sample"]].rename(columns={"location": "province"}),
        on=["province", "target_end_date", "sample"],
    )
    assert len(both) > 0.5 * len(alloc)
    assert (both["value"] >= both["total"]).all()


def test_zones_with_no_recent_cases_get_near_zero_draws(
    task: ForecastTask, counts: pd.DataFrame, zones: tuple[str, ...]
) -> None:
    _, alloc = _alloc(task, counts, zones)
    recent = counts.loc[counts["target_end_date"] >= "2026-08-01"]
    active = set(recent.loc[recent["total"] > 0, "location"])
    future = alloc.loc[alloc["target_end_date"] > task.vintage]
    quiet = future.loc[~future["location"].isin(active)]
    busy = future.loc[future["location"].isin(active)]
    assert quiet["value"].mean() < 0.1
    # Near zero, not exactly zero: a quiet zone can still get a case.
    assert quiet["value"].max() >= 1
    assert busy["value"].mean() > 20 * quiet["value"].mean()


def test_shares_shrink_recent_towards_longer_run() -> None:
    # One province, three zones. A had cases long ago only; B has the
    # recent cases; C never had any.
    weeks = pd.date_range("2026-08-01", "2026-10-10", freq="7D")
    rows = []
    for i, w in enumerate(weeks):
        recent = i >= len(weeks) - 3
        rows += [
            ("CD-XX-01", w, 0 if recent else 20),
            ("CD-XX-02", w, 10 if recent else 0),
        ]
    counts = pd.DataFrame(
        rows, columns=["location", "target_end_date", "total"]
    )
    counts["target_end_date"] = counts["target_end_date"].dt.date.astype(str)
    zones = ("CD-XX-01", "CD-XX-02", "CD-XX-03")
    shares = ZoneAllocator().shares(counts, zones, "2026-10-17")
    a, b, c = (shares[z] for z in zones)
    assert b > a > c > 0
    assert sum(shares.values()) == pytest.approx(1)
    # Stronger shrinkage moves the split towards the longer run.
    strong = ZoneAllocator(prior_strength=1e4).shares(
        counts, zones, "2026-10-17"
    )
    assert strong["CD-XX-01"] > a


def test_provinces_below_their_published_zone_total_are_thinned() -> None:
    counts = pd.DataFrame(
        {
            "location": ["CD-XX-01", "CD-XX-02"],
            "target_end_date": ["2026-10-10", "2026-10-10"],
            "total": [6, 4],
        }
    )
    prov = pd.DataFrame(
        {
            "location": "CD-XX",
            "target_end_date": "2026-10-10",
            "sample": [1, 2],
            "value": [3.0, 12.4],
        }
    )
    task = ForecastTask(
        vintage="2026-10-17",
        reference_date=pd.Timestamp("2026-10-24").date(),
        kind="real",
        target=pd.DataFrame(),
        delays=None,
        locations=("CD-XX",),
        target_end_dates=("2026-10-10",),
        n_samples=2,
    )
    alloc = ZoneAllocator().allocate(
        prov, task, counts, ("CD-XX-01", "CD-XX-02")
    )
    by = alloc.set_index(["sample", "location"])["value"]
    assert by.loc[1].sum() == 3
    assert (by.loc[1] <= [6, 4]).all()
    assert by.loc[2].sum() == 12
    assert (by.loc[2] >= [6, 4]).all()


def test_allocation_is_reproducible(
    task: ForecastTask, counts: pd.DataFrame, zones: tuple[str, ...]
) -> None:
    prov = Persistence().forecast(task)
    a = ZoneAllocator().allocate(prov, task, counts, zones)
    b = ZoneAllocator().allocate(prov, task, counts, zones)
    pd.testing.assert_frame_equal(a, b)


def test_check_zone_coherence_flags_a_gap(
    task: ForecastTask, counts: pd.DataFrame, zones: tuple[str, ...]
) -> None:
    prov, alloc = _alloc(task, counts, zones)
    bad = alloc.copy()
    bad.loc[bad.index[0], "value"] += 1
    assert check_zone_coherence(prov, bad)


def test_zone_task_is_a_zero_filled_persistence_task(
    task: ForecastTask, counts: pd.DataFrame, zones: tuple[str, ...]
) -> None:
    zt = zone_task(task, counts, zones)
    assert set(zt.locations) == {
        z for z in zones if province_of(z) in task.locations
    }
    one = zt.target.loc[zt.target["location"] == "CD-IT-10"]
    assert one["target_end_date"].is_monotonic_increasing
    assert one["target_end_date"].max() <= task.vintage
    gaps = pd.to_datetime(one["target_end_date"]).diff().dropna()
    assert (gaps == pd.Timedelta(days=7)).all()
    samples = Persistence().forecast(zt)
    assert set(samples["location"]) == set(zt.locations)


def test_hub_table_with_zones_validates(
    task: ForecastTask,
    counts: pd.DataFrame,
    zones: tuple[str, ...],
    tasks_json: Path,
    tmp_path: Path,
) -> None:
    prov, alloc = _alloc(task, counts, zones)
    table = to_hub_table(prov, task.reference_date, zones=alloc)
    nat = table.loc[table["location"] == "CD"]
    provs = table.loc[table["location"].str.len() == 5]
    summed = provs.groupby(["output_type_id", "target_end_date"])["value"].sum()
    got = nat.set_index(["output_type_id", "target_end_date"])["value"]
    # The national value is the sum of the provinces, not of the zones.
    assert (got.reindex(summed.index) == summed).all()
    assert set(alloc["location"]) <= set(table["location"])
    hub_round = load_hub_round(tasks_json)
    assert set(table["location"]) <= hub_round.locations
    path = write_submission(table, tmp_path, "epiforecasts-seabbsrenewal")
    assert validate_submission_file(path, hub_round, check_round=False) == []


def test_hub_table_rejects_incoherent_zones(
    task: ForecastTask, counts: pd.DataFrame, zones: tuple[str, ...]
) -> None:
    prov, alloc = _alloc(task, counts, zones)
    bad = alloc.copy()
    bad.loc[bad.index[0], "value"] += 1
    with pytest.raises(ValueError, match="zone"):
        to_hub_table(prov, task.reference_date, zones=bad)


def test_allocation_reads_final_rows_only(
    target: pd.DataFrame,
    delays: pd.DataFrame,
    daily_zones: pd.DataFrame,
    zones: tuple[str, ...],
) -> None:
    rows = target.loc[target["vintage"] == "2026-10-09"]
    task = make_tasks(rows, delays, BacktestConfig(seed=3), ("2026-10-24",))[0]
    prov = Persistence().forecast(task)
    assert "as_of" in prov
    counts = weekly_zone_counts(daily_zones, task.vintage)
    alloc = ZoneAllocator().allocate(prov, task, counts, zones)
    assert "as_of" not in alloc
    final = prov.loc[prov["as_of"].isna()].drop(columns="as_of")
    assert check_zone_coherence(final, alloc) == []
    assert np.isfinite(alloc["value"]).all()
