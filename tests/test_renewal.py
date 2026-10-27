from __future__ import annotations

import datetime as dt
import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from bdbv_hub_renewal.backtest import BacktestConfig, make_tasks
from bdbv_hub_renewal.fitting import Forecast, get_config
from bdbv_hub_renewal.models import (
    NATIONAL,
    ForecastModel,
    ForecastTask,
    get_model,
    list_models,
)
from bdbv_hub_renewal.models import renewal as rn

pytest.importorskip("pyrenew")


@pytest.fixture
def task(target: pd.DataFrame, delays: pd.DataFrame) -> ForecastTask:
    rows = target.loc[target["vintage"] == "2026-10-17"]
    return make_tasks(rows, delays, BacktestConfig(n_samples=20, seed=3))[0]


def test_variants_are_registered_as_fitted_models() -> None:
    specs = {s.name: s for s in list_models()}
    for name in ("hubrenewal", "hubrenewalsp"):
        assert specs[name].fitted
        model = get_model(name)
        assert isinstance(model, ForecastModel)
        assert model.name == name
    # The single patch fits the national sum, so cannot mask a province.
    assert get_model("hubrenewal").supports_mask
    assert not get_model("hubrenewalsp").supports_mask
    assert not specs["persistence"].fitted
    # Hub model abbreviations: letters, digits and underscores, under 16.
    for name in specs:
        assert len(name) < 16


def test_fit_configs_run_the_renewal_model() -> None:
    assert get_config("main").stand_in is None
    assert get_config("fallback").stand_in is None
    assert rn.VARIANTS["hubrenewal"].base == "MAIN"
    assert rn.FIT_VARIANTS == {"main": "hubrenewal", "fallback": "fallback"}


def test_model_config_applies_overrides() -> None:
    single = rn.model_config(rn.VARIANTS["hubrenewalsp"])
    assert single.national
    assert rn.model_config(rn.VARIANTS["hubrenewal"]) == rn.engine().MAIN


def test_sampler_settings_read_the_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(rn.SAMPLER_ENV, '{"num_chains": 1, "num_warmup": 5}')
    settings = rn.sampler_settings()
    assert settings["num_chains"] == 1
    assert settings["num_warmup"] == 5
    assert settings["num_samples"] == rn.DEFAULT_SAMPLER["num_samples"]
    assert rn.sampler_settings({"num_chains": 3})["num_chains"] == 3


def test_chain_diagnostics() -> None:
    rng = np.random.default_rng(1)
    n = 200
    extra = {
        "diverging": np.zeros((2, n), dtype=bool),
        "num_steps": np.full((2, n), 7),
        "accept_prob": np.full((2, n), 0.8),
        "energy": rng.normal(size=(2, n)).cumsum(axis=1),
        "adapt_state.step_size": np.full((2, n), 0.05),
    }
    extra["diverging"][1, :3] = True
    extra["num_steps"][0, :50] = 2**10 - 1
    chains = rn.chain_diagnostics(extra, max_tree_depth=10)
    assert [c["chain"] for c in chains] == [0, 1]
    assert chains[0]["draws"] == n
    assert chains[1]["divergences"] == 3
    assert chains[0]["tree_depth_hits"] == 50
    assert chains[0]["tree_depth_share"] == pytest.approx(0.25)
    assert chains[1]["tree_depth_share"] == 0.0
    assert chains[0]["mean_accept"] == pytest.approx(0.8)
    assert chains[0]["step_size"] == pytest.approx(0.05)
    # A random walk in energy has a small E-BFMI; white noise near 2.
    assert chains[0]["ebfmi"] < 0.3
    white = rn.ebfmi(rng.normal(size=4000))
    assert white == pytest.approx(2.0, rel=0.1)


def test_parameter_diagnostics_and_worst_rhat() -> None:
    rng = np.random.default_rng(2)
    trace = {
        "good": rng.normal(size=(2, 300, 3)),
        "bad": np.stack([rng.normal(size=300), rng.normal(5.0, 1.0, size=300)]),
        "fixed": np.ones((2, 300)),
    }
    params = rn.parameter_diagnostics(trace)
    names = [(p["name"], p["index"]) for p in params]
    assert names == [
        ("bad", 0),
        ("fixed", 0),
        ("good", 0),
        ("good", 1),
        ("good", 2),
    ]
    fixed = params[1]
    assert fixed["rhat"] is None
    assert fixed["ess_bulk"] is None
    assert params[0]["rhat"] > 1.5
    worst = rn.worst_rhat(params, n=2)
    assert [w["name"] for w in worst] == ["bad", "good"]
    assert all(set(p) >= {"rhat", "ess_bulk", "ess_tail"} for p in params)


def test_vintage_inputs_choose_the_active_set(task: ForecastTask) -> None:
    config = rn.model_config(rn.VARIANTS["hubrenewal"])
    data = rn.vintage_inputs(task, config)
    rows = task.target.loc[task.target["location"].isin(task.locations)]
    with_cases = set(
        rows.groupby("location")["total"].sum().loc[lambda s: s > 0].index
    )
    assert set(data.names) == with_cases
    assert set(data.inactive_names) == set(task.locations) - with_cases
    assert data.start_date.weekday() == 6
    assert data.known.shape == data.imputed.shape
    assert data.fit_mask is not None
    assert data.fit_mask.all()
    assert data.horizon_weeks == rn.horizon_weeks(task)


def test_vintage_inputs_apply_the_mask(task: ForecastTask) -> None:
    config = rn.model_config(rn.VARIANTS["hubrenewal"])
    full = rn.vintage_inputs(task, config)
    masked = rn.vintage_inputs(replace(task, mask=("CD-NK",)), config)
    j = masked.names.index("CD-NK")
    assert masked.fit_mask is not None
    assert not masked.fit_mask[j]
    assert masked.fit_mask.sum() == len(masked.names) - 1
    # Its delays leave the delay data; it stays in the latent process.
    assert masked.delay_counts.sum() < full.delay_counts.sum()
    assert masked.names == full.names


def test_single_patch_inputs_are_national(task: ForecastTask) -> None:
    config = rn.model_config(rn.VARIANTS["hubrenewalsp"])
    data = rn.vintage_inputs(task, config)
    assert data.names == [NATIONAL]
    assert data.known.shape[1] == 1


def _fake_draws(data: Any, n: int, rng: np.random.Generator) -> dict[str, Any]:
    weeks = data.n_weeks + data.horizon_weeks
    province = rng.poisson(5.0, size=(n, weeks, len(data.names))).astype(float)
    inactive = rng.poisson(0.2, size=(n, weeks, len(data.inactive_names)))
    insample = {
        "known": province * 0.7,
        "imputed": province * 0.3,
    }
    return {
        "province": province,
        "inactive": inactive.astype(float),
        "national": province.sum(2) + inactive.sum(2),
        "insample": insample,
    }


def test_samples_table_is_coherent(task: ForecastTask) -> None:
    # Every province, so some have no cases and get importation draws.
    rows = task.target.loc[task.target["location"] != NATIONAL]
    task = replace(task, locations=tuple(sorted(set(rows["location"]))))
    config = rn.model_config(rn.VARIANTS["hubrenewal"])
    data = rn.vintage_inputs(task, config)
    draws = _fake_draws(data, task.n_samples, np.random.default_rng(1))
    samples, unassigned = rn.samples_table(task, data, draws, report_all=True)
    assert unassigned is None
    assert set(samples["location"]) == set(task.locations)
    assert set(samples["sample"]) == set(range(1, task.n_samples + 1))
    # Report only the active provinces: the rest go to unassigned, so
    # national sample k is unchanged.
    some, extra = rn.samples_table(task, data, draws, report_all=False)
    assert extra is not None
    assert set(some["location"]) == set(data.names)
    full = samples.groupby(["target_end_date", "sample"])["value"].sum()
    part = some.groupby(["target_end_date", "sample"])["value"].sum()
    back = part.add(
        extra.set_index(["target_end_date", "sample"])["value"], fill_value=0
    )
    pd.testing.assert_series_equal(full, back, check_names=False)


def test_single_patch_apportions_national_draws(task: ForecastTask) -> None:
    config = rn.model_config(rn.VARIANTS["hubrenewalsp"])
    data = rn.vintage_inputs(task, config)
    draws = _fake_draws(data, task.n_samples, np.random.default_rng(2))
    samples, unassigned = rn.samples_table(task, data, draws, report_all=True)
    assert unassigned is None
    assert set(samples["location"]) == set(task.locations)
    first = data.start_date + dt.timedelta(days=6)
    week = (first + dt.timedelta(weeks=data.n_weeks)).isoformat()
    if week in task.target_end_dates:
        got = samples.loc[samples["target_end_date"] == week]
        total = got.groupby("sample")["value"].sum().to_numpy()
        np.testing.assert_allclose(total, draws["national"][:, data.n_weeks])


def test_ppc_rows_add_national(task: ForecastTask) -> None:
    config = rn.model_config(rn.VARIANTS["hubrenewal"])
    data = rn.vintage_inputs(task, config)
    draws = _fake_draws(data, 30, np.random.default_rng(3))
    observed, ppc = rn.ppc_rows(data, draws["insample"])
    assert set(observed["stream"]) == {"observed", "imputed"}
    assert NATIONAL in set(observed["location"])
    assert ppc.shape == (30, len(observed))
    nat = observed.loc[observed["location"] == NATIONAL]
    assert len(nat) == 2 * data.n_weeks


def _fake_run(task: ForecastTask, variant: rn.Variant) -> rn.RenewalRun:
    config = rn.model_config(variant)
    data = rn.vintage_inputs(task, config)
    draws = _fake_draws(data, task.n_samples, np.random.default_rng(4))
    rng = np.random.default_rng(5)
    n = 50
    trace = {"sd": rng.normal(size=(2, n)), "z": rng.normal(size=(2, n, 2))}
    extra = {
        "diverging": np.zeros((2, n), dtype=bool),
        "num_steps": np.full((2, n), 15),
        "accept_prob": np.full((2, n), 0.9),
        "energy": rng.normal(size=(2, n)),
        "adapt_state.step_size": np.full((2, n), 0.1),
    }
    return rn.RenewalRun(
        variant=variant,
        data=data,
        draws=draws,
        trace=trace,
        extra=extra,
        seconds=1.5,
    )


def test_fit_renewal_writes_the_ci_contract(
    task: ForecastTask, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[dict[str, Any]] = []

    def fake(self: rn.RenewalModel, t: ForecastTask) -> rn.RenewalRun:
        calls.append(dict(self.sampler))
        return _fake_run(t, self.variant)

    monkeypatch.setattr(rn.RenewalModel, "run", fake)
    sampler = {"num_chains": 2, "num_samples": 50, "seed": 7}
    record = rn.fit_renewal(
        task, sampler, tmp_path, variant="hubrenewal", config_name="main"
    )
    assert calls[0]["num_chains"] == 2
    assert record["sampler"] == "nuts"
    assert record["model"] == "hubrenewal"
    assert record["config"] == "main"
    assert record["num_chains"] == 2
    assert {c["chain"] for c in record["chains"]} == {0, 1}
    assert {"tree_depth_share", "ebfmi", "mean_accept", "step_size"} <= set(
        record["chains"][0]
    )
    assert len(record["worst_rhat"]) <= 10
    assert {p["name"] for p in record["parameters"]} == {"sd", "z"}
    ppc = json.loads((tmp_path / "ppc.json").read_text())
    assert ppc["available"] is True
    assert ppc["config"] == "main"
    assert ppc["model"] == "hubrenewal"

    result = rn.forecast_renewal(task, tmp_path)
    assert isinstance(result, Forecast)
    assert set(result.samples["sample"]) == set(range(1, task.n_samples + 1))
    assert set(result.samples["target_end_date"]) == set(task.target_end_dates)
    assert result.modelled is not None


def test_renewal_model_forecast_uses_the_run(
    task: ForecastTask, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        rn.RenewalModel, "run", lambda self, t: _fake_run(t, self.variant)
    )
    samples = get_model("hubrenewal").forecast(task)
    assert set(samples["location"]) == set(task.locations)


@pytest.mark.slow
def test_tiny_fit_end_to_end(task: ForecastTask, tmp_path: Path) -> None:
    sampler = {"num_chains": 1, "num_warmup": 20, "num_samples": 20}
    record = rn.fit_renewal(
        task, sampler, tmp_path, variant="hubrenewal", config_name="main"
    )
    assert record["chains"][0]["draws"] == 20
    result = rn.forecast_renewal(task, tmp_path)
    assert np.isfinite(result.samples["value"]).all()
