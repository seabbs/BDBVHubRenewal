from __future__ import annotations

import dataclasses
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from bdbvhubrenewal.data import load_health_zones
from bdbvhubrenewal.data.zones import (
    hub_zones,
    province_of,
    weekly_zone_counts,
)
from bdbvhubrenewal.evaluation.backtest import BacktestConfig, make_tasks
from bdbvhubrenewal.forecasters import (
    ForecastTask,
    RandomWalk,
    RandomWalkNowcast,
)
from bdbvhubrenewal.forecasters.zones import (
    ZoneAllocator,
    _apportion,
    check_zone_coherence,
    reconcile_zones,
    zone_forecast,
    zone_task,
)
from bdbvhubrenewal.submission.hub import to_hub_table, write_submission
from bdbvhubrenewal.submission.validate import Hub, validate_submission

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
SNAPSHOT = ROOT / "tests/submission/hub_validation/fixtures/hub-snapshot"
HUB = Hub.load(SNAPSHOT)
"""The pinned hub config."""
OPEN_VINTAGE = "2026-10-24"
"""A vintage whose reference date, 2026-10-31, is open in `HUB`."""


@pytest.fixture(scope="module")
def zones() -> tuple[str, ...]:
    return hub_zones(DATA / "hub-config" / "tasks.json")


def _task(
    target: pd.DataFrame,
    delays: pd.DataFrame,
    daily: pd.DataFrame,
    vintage: str,
) -> ForecastTask:
    rows = target.loc[target["vintage"] == vintage]
    return make_tasks(rows, delays, BacktestConfig(seed=3), daily=daily)[0]


@pytest.fixture(scope="module")
def task(
    target: pd.DataFrame, delays: pd.DataFrame, daily: pd.DataFrame
) -> ForecastTask:
    return _task(target, delays, daily, "2026-10-17")


@pytest.fixture(scope="module")
def counts(daily_zones: pd.DataFrame) -> pd.DataFrame:
    return weekly_zone_counts(daily_zones, "2026-10-17")


@pytest.fixture(scope="module")
def open_task(
    target: pd.DataFrame, delays: pd.DataFrame, daily: pd.DataFrame
) -> ForecastTask:
    return _task(target, delays, daily, OPEN_VINTAGE)


@pytest.fixture(scope="module")
def open_counts(daily_zones: pd.DataFrame) -> pd.DataFrame:
    return weekly_zone_counts(daily_zones, OPEN_VINTAGE)


def _hub_failures(path: Path) -> list[str]:
    """The hub's checks a written submission fails."""
    rep = validate_submission(
        path,
        HUB,
        metadata_dirs=[ROOT / "hub" / "model-metadata"],
        skip_window=True,
    )
    return [f"{c.check}: {c.message}" for c in rep.failed]


def _alloc(
    task: ForecastTask,
    counts: pd.DataFrame,
    zones: tuple[str, ...],
    samples: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    prov = RandomWalk().forecast(task) if samples is None else samples
    return prov, ZoneAllocator().allocate(prov, task, counts, zones)


@pytest.fixture(scope="module")
def alloc_run(
    task: ForecastTask, counts: pd.DataFrame, zones: tuple[str, ...]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Random walk province samples at 2026-10-17 and their zone split."""
    return _alloc(task, counts, zones)


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
    alloc_run: tuple[pd.DataFrame, pd.DataFrame],
) -> None:
    prov, alloc = alloc_run
    assert check_zone_coherence(prov, alloc) == []
    summed = (
        alloc.assign(location=alloc["location"].map(province_of))
        .groupby(["location", "target_end_date", "sample"])["value"]
        .sum()
    )
    want = prov.set_index(["location", "target_end_date", "sample"])["value"]
    assert (summed.reindex(want.index) == want.clip(lower=0).round()).all()


def test_every_zone_of_a_forecast_province_is_covered(
    task: ForecastTask,
    zones: tuple[str, ...],
    alloc_run: tuple[pd.DataFrame, pd.DataFrame],
) -> None:
    _, alloc = alloc_run
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
    counts: pd.DataFrame,
    alloc_run: tuple[pd.DataFrame, pd.DataFrame],
) -> None:
    prov, alloc = alloc_run
    seen = counts.set_index(["location", "target_end_date"])["total"]
    both = alloc.join(seen, on=["location", "target_end_date"]).fillna(0)
    # Random walk copies complete weeks, so the zones copy theirs too.
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
    task: ForecastTask,
    counts: pd.DataFrame,
    alloc_run: tuple[pd.DataFrame, pd.DataFrame],
) -> None:
    _, alloc = alloc_run
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
    split = ZoneAllocator(recent_weeks=4, prior_strength=20)
    shares = split.shares(counts, zones, "2026-10-17")
    a, b, c = (shares[z] for z in zones)
    assert b > a > c > 0
    assert sum(shares.values()) == pytest.approx(1)
    # Stronger shrinkage moves the split towards the longer run.
    strong = ZoneAllocator(recent_weeks=4, prior_strength=1e4).shares(
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
        cutoff="2026-10-16",
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
    task: ForecastTask,
    counts: pd.DataFrame,
    zones: tuple[str, ...],
    alloc_run: tuple[pd.DataFrame, pd.DataFrame],
) -> None:
    prov = alloc_run[0]
    a = ZoneAllocator().allocate(prov, task, counts, zones)
    b = ZoneAllocator().allocate(prov, task, counts, zones)
    pd.testing.assert_frame_equal(a, b)


def test_zone_task_is_a_zero_filled_randomwalk_task(
    task: ForecastTask,
    counts: pd.DataFrame,
    zones: tuple[str, ...],
    daily: pd.DataFrame,
    daily_zones: pd.DataFrame,
) -> None:
    zt = zone_task(task, counts, zones, pd.concat([daily, daily_zones]))
    assert set(zt.locations) == {
        z for z in zones if province_of(z) in task.locations
    }
    one = zt.target.loc[zt.target["location"] == "CD-IT-10"]
    assert one["target_end_date"].is_monotonic_increasing
    assert one["target_end_date"].max() <= task.vintage
    gaps = pd.to_datetime(one["target_end_date"]).diff().dropna()
    assert (gaps == pd.Timedelta(days=7)).all()
    # The zone task carries snapshots of the country, provinces and zones.
    assert zt.daily_history is not None
    assert {"CD", "CD-IT", "CD-IT-10"} <= set(zt.daily_history["location"])
    assert zt.daily_history["vintage"].max() == task.vintage
    assert zone_task(task, counts, zones).daily_history is None


def test_hub_table_with_zones_validates(
    open_task: ForecastTask,
    open_counts: pd.DataFrame,
    zones: tuple[str, ...],
    tmp_path: Path,
) -> None:
    prov, alloc = _alloc(open_task, open_counts, zones)
    table = to_hub_table(
        prov,
        open_task.reference_date,
        zones=alloc,
        forecast_date=open_task.reference_date,
    )
    nat = table.loc[table["location"] == "CD"]
    provs = table.loc[table["location"].str.len() == 5]
    summed = provs.groupby(["output_type_id", "target_end_date"])["value"].sum()
    got = nat.set_index(["output_type_id", "target_end_date"])["value"]
    # The national value is the sum of the provinces, not of the zones.
    assert (got.reindex(summed.index) == summed).all()
    assert set(alloc["location"]) <= set(table["location"])
    path = write_submission(table, tmp_path, "seabbs-renewal")
    assert _hub_failures(path) == []


def test_hub_table_rejects_incoherent_zones(
    task: ForecastTask,
    alloc_run: tuple[pd.DataFrame, pd.DataFrame],
) -> None:
    prov, alloc = alloc_run
    bad = alloc.copy()
    bad.loc[bad.index[0], "value"] += 1
    assert check_zone_coherence(prov, bad)
    with pytest.raises(ValueError, match="zone"):
        to_hub_table(prov, task.reference_date, zones=bad)


def test_allocation_reads_final_rows_only(
    task: ForecastTask,
    counts: pd.DataFrame,
    zones: tuple[str, ...],
    alloc_run: tuple[pd.DataFrame, pd.DataFrame],
) -> None:
    final = alloc_run[0]
    prov = pd.concat(
        [final.assign(as_of=np.nan), final.assign(as_of="2026-10-24")],
        ignore_index=True,
    )
    alloc = ZoneAllocator().allocate(prov, task, counts, zones)
    assert "as_of" not in alloc
    assert check_zone_coherence(final, alloc) == []
    assert np.isfinite(alloc["value"]).all()


def test_apportion_sums_to_the_total_in_whole_numbers() -> None:
    total = np.array([[7, 0], [3, 10]])
    weight = np.array(
        [[[1.0, 1.0, 1.0], [2.0, 1.0, 1.0]], [[0.0, 5.0, 0.0], [1.0, 3.0, 6.0]]]
    )
    out = _apportion(total, weight)
    assert (out.sum(axis=-1) == total).all()
    assert (out >= 0).all()
    assert out[1, 0].tolist() == [0, 3, 0]
    assert out[1, 1].tolist() == [1, 3, 6]
    assert out[0, 1].tolist() == [0, 0, 0]
    assert out[0, 0].tolist() == [3, 2, 2]


def _zone_model_inputs(
    task: ForecastTask,
    counts: pd.DataFrame,
    zones: tuple[str, ...],
    daily_zones: pd.DataFrame,
    model: RandomWalk,
    n_provinces: int = 2,
) -> tuple[ForecastTask, pd.DataFrame, pd.DataFrame]:
    sub = dataclasses.replace(task, locations=task.locations[:n_provinces])
    prov = model.forecast(sub)
    got = zone_forecast(model, sub, prov, counts, zones, daily_zones)
    return sub, prov, got


@pytest.mark.parametrize("model", [RandomWalk(), RandomWalkNowcast()])
def test_zone_walk_models_sum_to_their_provinces(
    task: ForecastTask,
    counts: pd.DataFrame,
    zones: tuple[str, ...],
    daily_zones: pd.DataFrame,
    model: RandomWalk,
) -> None:
    sub, prov, got = _zone_model_inputs(task, counts, zones, daily_zones, model)
    assert check_zone_coherence(prov, got) == []
    want = {z for z in zones if province_of(z) in sub.locations}
    assert set(got["location"]) == want
    assert (got["value"] == got["value"].round()).all()
    assert (got["value"] >= 0).all()


def test_zone_values_follow_the_zone_model_not_the_shares(
    task: ForecastTask,
    counts: pd.DataFrame,
    zones: tuple[str, ...],
    daily_zones: pd.DataFrame,
) -> None:
    model = RandomWalkNowcast()
    sub, prov, got = _zone_model_inputs(task, counts, zones, daily_zones, model)
    allocated = ZoneAllocator().allocate(prov, sub, counts, zones)
    keys = ["location", "target_end_date"]
    mine = got.groupby(keys)["value"].mean()
    theirs = allocated.groupby(keys)["value"].mean()
    assert not np.allclose(mine, theirs.reindex(mine.index))


def test_reconcile_falls_back_to_the_allocator_shares(
    task: ForecastTask, counts: pd.DataFrame, zones: tuple[str, ...]
) -> None:
    sub = dataclasses.replace(task, locations=task.locations[:1])
    prov = RandomWalk().forecast(sub)
    empty = pd.DataFrame(columns=list(prov.columns))
    got = reconcile_zones(empty, prov, sub, counts, zones)
    assert check_zone_coherence(prov, got) == []
    assert got["value"].sum() > 0


def test_zone_hub_table_validates_and_sums_up_the_hierarchy(
    open_task: ForecastTask,
    open_counts: pd.DataFrame,
    zones: tuple[str, ...],
    daily_zones: pd.DataFrame,
    tmp_path: Path,
) -> None:
    model = RandomWalkNowcast()
    prov = model.forecast(open_task)
    got = zone_forecast(model, open_task, prov, open_counts, zones, daily_zones)
    ref = open_task.reference_date
    table = to_hub_table(prov, ref, zones=got, forecast_date=ref)
    zone_rows = table.loc[table["location"].str.count("-") == 2]
    by_prov = (
        zone_rows.assign(location=zone_rows["location"].str[:5])
        .groupby(["location", "output_type_id", "target_end_date"])["value"]
        .sum()
    )
    prov_rows = table.loc[table["location"].str.len() == 5]
    want = prov_rows.set_index(
        ["location", "output_type_id", "target_end_date"]
    )
    assert (by_prov == want["value"].reindex(by_prov.index)).all()
    path = write_submission(table, tmp_path, "seabbs-rwnowcast")
    assert _hub_failures(path) == []
