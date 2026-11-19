"""Province-level models return their zones, split as submission did."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd
import pytest

from bdbvhubrenewal.data import load_daily_zones
from bdbvhubrenewal.data.zones import (
    hub_zones,
    is_zone,
    load_zone_data,
    province_of,
    weekly_zone_counts,
)
from bdbvhubrenewal.evaluation.backtest import BacktestConfig, make_tasks
from bdbvhubrenewal.forecasters import ForecastTask, get_model
from bdbvhubrenewal.forecasters import renewal as rn
from bdbvhubrenewal.forecasters.zones import ZoneAllocator, with_zones

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "tests" / "fixtures" / "data"
VINTAGE = "2026-10-24"
KEYS = ["location", "target_end_date", "sample"]


@pytest.fixture(scope="module")
def task(
    target: pd.DataFrame, delays: pd.DataFrame, daily: pd.DataFrame
) -> ForecastTask:
    rows = target.loc[target["vintage"] == VINTAGE]
    return make_tasks(rows, delays, BacktestConfig(seed=3), daily=daily)[0]


@pytest.fixture(scope="module")
def zone_task(
    target: pd.DataFrame, delays: pd.DataFrame, daily: pd.DataFrame
) -> ForecastTask:
    rows = target.loc[target["vintage"] == VINTAGE]
    return make_tasks(
        rows,
        delays,
        BacktestConfig(seed=3),
        daily=daily,
        zones=load_zone_data(DATA),
    )[0]


def _submitted_split(samples: pd.DataFrame, task: ForecastTask) -> pd.DataFrame:
    """The split as the submission made it: the full zone file, hub codes."""
    counts = weekly_zone_counts(load_daily_zones(DATA), task.vintage)
    codes = hub_zones(DATA / "hub-config" / "tasks.json")
    return ZoneAllocator().allocate(samples, task, counts, codes)


def _sorted(rows: pd.DataFrame) -> pd.DataFrame:
    return rows[[*KEYS, "value"]].sort_values(KEYS).reset_index(drop=True)


def _provinces(task: ForecastTask) -> pd.DataFrame:
    return get_model("randomwalk").forecast(task)


def test_with_zones_adds_the_split_submission_made(
    zone_task: ForecastTask,
) -> None:
    prov = _provinces(zone_task)
    got = with_zones(prov, zone_task)
    zones = got.loc[got["location"].map(is_zone)]
    pd.testing.assert_frame_equal(
        _sorted(zones), _sorted(_submitted_split(prov, zone_task))
    )
    # The province and national rows are untouched.
    pd.testing.assert_frame_equal(
        _sorted(got.loc[~got["location"].map(is_zone)]), _sorted(prov)
    )
    mine = {
        z for z in zone_task.zone_codes if province_of(z) in zone_task.locations
    }
    assert set(zones["location"]) == mine


def test_with_zones_leaves_samples_without_zone_data_or_with_zones(
    zone_task: ForecastTask, task: ForecastTask
) -> None:
    prov = _provinces(task)
    assert task.zone_codes == ()
    pd.testing.assert_frame_equal(with_zones(prov, task), prov)
    # A model with zones of its own keeps them.
    own = with_zones(_provinces(zone_task), zone_task)
    pd.testing.assert_frame_equal(with_zones(own, zone_task), own)


@pytest.mark.parametrize("name", ["renewal", "renewalsp", "renewalbvd"])
def test_renewal_models_read_zones(name: str) -> None:
    assert getattr(get_model(name), "reads_zones", False)


def _fake_renewal(monkeypatch: pytest.MonkeyPatch) -> None:
    # The renewal tests' stand-in for a NUTS run.
    spec = importlib.util.spec_from_file_location(
        "renewal_tests", Path(__file__).with_name("test_renewal.py")
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    _fake_run = module._fake_run
    monkeypatch.setattr(
        rn.RenewalModel, "run", lambda self, t: _fake_run(t, self.variant)
    )


def test_renewal_forecast_and_fit_dir_carry_the_submitted_zones(
    zone_task: ForecastTask,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # renewalsp fits provinces, so its zones are split at submission.
    _fake_renewal(monkeypatch)
    out = get_model("renewalsp").forecast(zone_task)
    own = out["location"].map(is_zone)
    assert own.any()
    pd.testing.assert_frame_equal(
        _sorted(out.loc[own]),
        _sorted(_submitted_split(out.loc[~own], zone_task)),
    )
    # The fit directory the submission reads holds the same rows. Its
    # silent provinces' walk draws with the sampler seed, so it is the
    # task's here.
    sampler = {"num_chains": 2, "num_samples": 50, "seed": zone_task.seed}
    rn.fit_renewal(
        zone_task, sampler, tmp_path,
        variant="renewalsp", config_name="renewalsp",
    )  # fmt: skip
    read = rn.forecast_renewal(zone_task, tmp_path).samples
    pd.testing.assert_frame_equal(_sorted(read), _sorted(out))


def test_the_submitted_file_is_unchanged(
    zone_task: ForecastTask,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from bdbvhubrenewal.cli import _hub_forecast
    from bdbvhubrenewal.submission.fitting import get_config, run_fit
    from bdbvhubrenewal.submission.hub import to_hub_table

    _fake_renewal(monkeypatch)
    # The CI's sampler seed is not the task's, as here.
    sampler = {"num_chains": 2, "num_samples": 50, "seed": 7}
    run_fit(
        get_config("renewalsp"), zone_task, sampler, tmp_path,
        model="renewalsp",
    )  # fmt: skip
    table, result, *_ = _hub_forecast(
        zone_task, data=str(DATA), model=None, config="renewalsp",
        fit=tmp_path,
        forecast_date=VINTAGE,
    )  # fmt: skip
    # As renewalsp wrote it: province samples, split at submission.
    prov = result.samples.loc[~result.samples["location"].map(is_zone)]
    want = to_hub_table(
        prov, zone_task.reference_date, unassigned=result.unassigned,
        zones=_submitted_split(prov, zone_task), forecast_date=VINTAGE,
    )  # fmt: skip
    pd.testing.assert_frame_equal(table, want)
