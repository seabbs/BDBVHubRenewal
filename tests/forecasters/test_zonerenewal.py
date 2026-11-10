from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from bdbvhubrenewal.data.zones import province_of
from bdbvhubrenewal.evaluation.backtest import BacktestConfig, make_tasks
from bdbvhubrenewal.forecasters import ForecastTask, base, get_model
from bdbvhubrenewal.forecasters import renewal as rn
from bdbvhubrenewal.forecasters import zonerenewal as zr

pytest.importorskip("pyrenew")

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def task(target: pd.DataFrame, delays: pd.DataFrame) -> ForecastTask:
    rows = target.loc[target["vintage"] == "2026-10-17"]
    return make_tasks(rows, delays, BacktestConfig(n_samples=20, seed=3))[0]


@pytest.fixture
def zoned(monkeypatch: pytest.MonkeyPatch) -> None:
    """Register the zone variants for one test."""
    for name in zr.ZONE_VARIANTS:
        monkeypatch.setitem(base._REGISTRY, name, base._REGISTRY[name])
    zr.enable_zones()


@pytest.fixture
def quiet_zones(
    task: ForecastTask, monkeypatch: pytest.MonkeyPatch
) -> set[str]:
    """Make every zone silent but the four with most cases, with 0 draws."""
    zones = zr.task_zones(task)
    counts = zr.zone_target(task).groupby("location")["total"].sum()
    busy = set(counts.sort_values().index[-4:])
    silent = {z for z in zones if z not in busy}
    rows = pd.DataFrame(
        [
            (z, date, k, 0)
            for z in sorted(silent)
            for date in task.target_end_dates
            for k in range(1, task.n_samples + 1)
        ],
        columns=["location", "target_end_date", "sample", "value"],
    )
    monkeypatch.setattr(zr, "_silent_walk", lambda _: (rows, silent))
    return silent


def _fake_run(task: ForecastTask, variant: rn.Variant) -> rn.RenewalRun:
    data = zr.zone_vintage_inputs(task, rn.model_config(variant))
    rng = np.random.default_rng(4)
    weeks = data.n_weeks + data.horizon_weeks
    zone = rng.poisson(5.0, size=(task.n_samples, weeks, len(data.names)))
    draws: dict[str, Any] = {
        "province": zone.astype(float),
        "insample": {"known": zone * 0.7, "imputed": zone * 0.3},
        "as_reported": {
            cut: {"province": np.ones_like(zone, dtype=float)}
            for cut in rn.report_cutoffs(task, data)
        },
    }
    n = 50
    extra = {
        "diverging": np.zeros((2, n), dtype=bool),
        "num_steps": np.full((2, n), 15),
        "accept_prob": np.full((2, n), 0.9),
        "energy": rng.normal(size=(2, n)),
        "adapt_state.step_size": np.full((2, n), 0.1),
    }
    delay = pd.concat(
        rn.delay_summary(
            rng.normal(-1, 0.1, size=(n, 20)),
            rng.normal(0, 0.1, size=(n, 14)),
            data.start_date,
            13,
        ).assign(place=place)
        for place in data.parent_names
    )
    return rn.RenewalRun(
        variant=variant,
        data=data,
        draws=draws,
        trace={"sd": rng.normal(size=(2, n))},
        extra=extra,
        seconds=1.5,
        cpu_seconds=2.9,
        delay=delay,
    )


def test_zone_fits_are_opt_in(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv(zr.ZONES_ENV, raising=False)
    # The shipped config keeps the province fits.
    assert zr.zone_fits(ROOT / "config" / "hub.toml") is False
    assert zr.zone_fits(tmp_path / "missing.toml") is False
    on = tmp_path / "hub.toml"
    on.write_text("[renewal]\nzones = true\n")
    assert zr.zone_fits(on) is True
    monkeypatch.setenv(zr.ZONES_ENV, "0")
    assert zr.zone_fits(on) is False
    monkeypatch.setenv(zr.ZONES_ENV, "maybe")
    with pytest.raises(ValueError, match=zr.ZONES_ENV):
        zr.zone_fits(on)


def test_province_models_stay_unless_zones_are_enabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert type(get_model("renewalbvd")) is rn.RenewalModel
    for name in zr.ZONE_VARIANTS:
        monkeypatch.setitem(base._REGISTRY, name, base._REGISTRY[name])
    zr.enable_zones()
    model = get_model("renewalbvd")
    assert isinstance(model, zr.ZoneRenewalModel)
    assert model.variant.base == "BVD"
    assert type(get_model("renewalsp")) is rn.RenewalModel
    # The zone fit's representation differs, so it keys a stored fit.
    assert repr(model) != repr(rn.RenewalModel(rn.VARIANTS["renewalbvd"]))


@pytest.mark.usefixtures("zoned")
def test_zone_fit_sums_fitted_and_silent_zones_to_provinces(
    task: ForecastTask,
    monkeypatch: pytest.MonkeyPatch,
    quiet_zones: set[str],
    tmp_path: Path,
) -> None:
    seen: list[tuple[str, ...]] = []

    def fake(self: zr.ZoneRenewalModel, t: ForecastTask) -> rn.RenewalRun:
        seen.append(t.locations)
        return _fake_run(t, self.variant)

    monkeypatch.setattr(zr.ZoneRenewalModel, "run", fake)
    model = get_model("renewal")
    assert isinstance(model, zr.ZoneRenewalModel)
    samples = model.forecast(task)
    # One joint fit of the zones the random walks do not find silent.
    (fitted,) = seen
    assert len(fitted) == 4
    assert all(len(z.split("-")) == 3 for z in fitted)
    assert not set(fitted) & quiet_zones
    zones = model.zone_samples
    assert zones is not None
    assert set(zones["location"]) == set(fitted) | quiet_zones
    assert set(samples["location"]) == set(task.locations)
    # Provinces are the sums of their zones, sample by sample, and
    # national is their sum, with no unassigned part.
    assert model.last_fit is not None
    assert model.last_fit.unassigned is None
    keys = ["location", "target_end_date", "sample"]
    summed = (
        zones.assign(location=zones["location"].map(province_of))
        .groupby(keys)["value"]
        .sum()
    )
    pd.testing.assert_series_equal(
        samples.set_index(keys)["value"].sort_index(),
        summed.sort_index(),
        check_names=False,
    )
    # The delay is by province.
    assert model.delay_summary is not None
    assert set(model.delay_summary["location"]) <= {
        *task.locations,
        "shared",
    }
    # The fit directory carries the zone samples for the submission.
    model.write_fit_dir(task, tmp_path, "main")
    written = pd.read_csv(tmp_path / zr.ZONES_FILE)
    assert set(written["location"]) == set(zones["location"])


def test_zone_inputs_take_the_zone_counts_and_province_delays(
    task: ForecastTask,
) -> None:
    rows = zr.zone_target(task)
    assert {"observed", "imputed", "total", "cutoff"} <= set(rows.columns)
    zones = tuple(rows.groupby("location")["total"].sum().nlargest(6).index)
    zt = replace(task, target=rows, locations=zones)
    data = zr.zone_vintage_inputs(zt, rn.model_config(rn.VARIANTS["renewal"]))
    assert data.names == list(zones)
    assert [data.parent_names[i] for i in data.parents] == [
        province_of(z) for z in zones
    ]
    assert data.delay_counts.shape[1] == len(data.parent_names)
    # Zone weekly counts sum to the province counts of the release.
    prov = task.target.loc[task.target["location"].isin(data.parent_names)]
    wide = prov.pivot_table(
        index="target_end_date", columns="location", values="total"
    )
    zone_totals = rows.assign(p=rows["location"].map(province_of)).pivot_table(
        index="target_end_date", columns="p", values="total", aggfunc="sum"
    )[data.parent_names]
    pd.testing.assert_frame_equal(
        wide.reindex_like(zone_totals).fillna(0),
        zone_totals.fillna(0),
        check_names=False,
        check_dtype=False,
    )


def test_zone_code_does_not_key_the_province_fits(tmp_path: Path) -> None:
    import shutil

    from bdbvhubrenewal.evaluation import store

    for name in ("pyproject.toml", "uv.lock"):
        shutil.copy(ROOT / name, tmp_path / name)
    shutil.copytree(
        ROOT / "src",
        tmp_path / "src",
        ignore=shutil.ignore_patterns("__pycache__"),
    )
    models = ("renewal", "renewalbvd", "renewalsp", "pprandomwalknowcast")
    before = {m: store.code_digest(m, tmp_path) for m in models}
    for rel in ("model/zones.py", "forecasters/zonerenewal.py"):
        path = tmp_path / "src/bdbvhubrenewal" / rel
        path.write_text(path.read_text() + "\n_EDITED = 1\n")
    assert {m: store.code_digest(m, tmp_path) for m in models} == before
