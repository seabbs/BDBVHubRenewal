from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from bdbvhubrenewal.evaluation.backtest import BacktestConfig, make_tasks
from bdbvhubrenewal.forecasters import (
    NATIONAL,
    ForecastModel,
    ForecastTask,
    check_samples,
    get_model,
    list_models,
)
from bdbvhubrenewal.forecasters import renewal as rn
from bdbvhubrenewal.submission.fitting import Forecast, get_config

pytest.importorskip("pyrenew")


@pytest.fixture
def task(target: pd.DataFrame, delays: pd.DataFrame) -> ForecastTask:
    rows = target.loc[target["vintage"] == "2026-10-17"]
    return make_tasks(rows, delays, BacktestConfig(n_samples=20, seed=3))[0]


def test_variants_are_registered_as_fitted_models() -> None:
    specs = {s.name: s for s in list_models()}
    for name in ("renewal", "renewalsp"):
        assert specs[name].fitted
        model = get_model(name)
        assert isinstance(model, ForecastModel)
        assert model.name == name
    # The backtest options are registered too.
    for name, field, value in (
        ("renewalhold", "future_rt", "hold"),
        ("renewalpo", "importation", "per_origin"),
    ):
        assert specs[name].fitted
        if field is not None:
            config = rn.model_config(rn.VARIANTS[name])
            assert getattr(config, field) == value
    # The single patch fits the national sum, so cannot mask a province.
    assert get_model("renewal").supports_mask
    assert not get_model("renewalsp").supports_mask
    assert not specs["persistence"].fitted
    # Hub model abbreviations: letters, digits and underscores, under 16.
    for name in specs:
        assert len(name) < 16


def test_fit_configs_run_the_renewal_model() -> None:
    assert get_config("main").stand_in is None
    assert rn.VARIANTS["renewal"].base == "MAIN"
    assert rn.FIT_VARIANTS == {
        "main": "renewal",
        "renewalsp": "renewalsp",
        "bvd": "renewalbvd",
    }
    assert get_config("bvd").stand_in is None


def test_model_config_applies_overrides() -> None:
    single = rn.model_config(rn.VARIANTS["renewalsp"])
    assert single.national
    assert rn.model_config(rn.VARIANTS["renewal"]) == rn.engine().MAIN


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
    config = rn.model_config(rn.VARIANTS["renewal"])
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
    config = rn.model_config(rn.VARIANTS["renewal"])
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
    config = rn.model_config(rn.VARIANTS["renewalsp"])
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
    config = rn.model_config(rn.VARIANTS["renewal"])
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


def test_ppc_rows_add_national(task: ForecastTask) -> None:
    config = rn.model_config(rn.VARIANTS["renewal"])
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
    # The counts as reported by each later vintage: one case a province.
    draws["as_reported"] = {
        cut: {
            "province": np.ones_like(draws["province"]),
            "inactive": np.zeros_like(draws["inactive"]),
        }
        for cut in rn.report_cutoffs(task, data)
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
        task, sampler, tmp_path, variant="renewal", config_name="main"
    )
    assert calls[0]["num_chains"] == 2
    assert record["sampler"] == "nuts"
    assert record["model"] == "renewal"
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
    assert ppc["model"] == "renewal"

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
    samples = get_model("renewal").forecast(task)
    assert set(samples["location"]) == set(task.locations)


@pytest.mark.slow
def test_tiny_fit_end_to_end(task: ForecastTask, tmp_path: Path) -> None:
    sampler = {"num_chains": 1, "num_warmup": 20, "num_samples": 20}
    record = rn.fit_renewal(
        task, sampler, tmp_path, variant="renewal", config_name="main"
    )
    assert record["chains"][0]["draws"] == 20
    result = rn.forecast_renewal(task, tmp_path)
    assert np.isfinite(result.samples["value"]).all()


def test_init_strategy_follows_the_sampler_table() -> None:
    from numpyro.infer import NUTS

    median = rn.init_strategy({})
    jitter = rn.init_strategy({"init": "jitter"})
    assert median.func.__name__ == "init_to_median"
    assert jitter.func.__name__ == "init_to_uniform"
    assert jitter.keywords["radius"] == rn.JITTER_RADIUS
    with pytest.raises(ValueError, match="init"):
        rn.init_strategy({"init": "nope"})
    kernel = rn.nuts_kernel(
        lambda: None, {**rn.DEFAULT_SAMPLER, "init": "jitter"}
    )
    assert isinstance(kernel, NUTS)
    assert kernel._init_strategy is jitter or (
        kernel._init_strategy.func is jitter.func
    )


def test_unassigned_adds_the_national_data_gap(task: ForecastTask) -> None:
    config = rn.model_config(rn.VARIANTS["renewal"])
    data = rn.vintage_inputs(task, config)
    draws = _fake_draws(data, task.n_samples, np.random.default_rng(6))
    samples, _ = rn.samples_table(task, data, draws, report_all=True)
    assert rn.data_gap(task, samples) is None
    # Two national cases a week with no province, in the last data weeks.
    target = task.target.copy()
    nat = target["location"] == NATIONAL
    weeks = sorted(set(target.loc[nat, "target_end_date"]))[-4:]
    hit = nat & target["target_end_date"].isin(weeks)
    target.loc[hit, "total"] += 2
    gapped = replace(task, target=target)
    gap = rn.data_gap(gapped, samples)
    assert gap is not None
    assert list(gap.columns) == ["target_end_date", "sample", "value"]
    seen = gap.loc[gap["target_end_date"].isin(weeks)]
    assert (seen["value"] == 2).all()
    # Later weeks carry the recent share of the provinces forward.
    later = [w for w in task.target_end_dates if w > max(weeks)]
    ahead = gap.loc[gap["target_end_date"].isin(later)]
    assert len(ahead) == len(later) * task.n_samples
    prov = samples.loc[samples["target_end_date"].isin(later)]
    by = prov.groupby(["target_end_date", "sample"])["value"].sum()
    ratio = ahead.set_index(["target_end_date", "sample"])["value"] / by
    assert ratio.dropna().between(0, 1).all()
    assert ratio.dropna().std() < 1e-9


def test_bvd_variant_reads_the_release_for_its_vintage(
    task: ForecastTask, tmp_path: Path
) -> None:
    import shutil

    from bdbvhubrenewal.data import bvd

    fixtures = Path(__file__).resolve().parents[1] / "data" / "bvd"
    releases = bvd.parse_releases(
        json.loads((fixtures / "releases.json").read_text())
    )
    bvd.save_releases(
        releases.loc[releases["tag"].isin(["results-3151", "results-3086"])],
        tmp_path,
    )
    for tag in ("results-3151", "results-3086"):
        shutil.copytree(fixtures / tag, tmp_path / tag)
    assert rn.VARIANTS["renewalbvd"].register
    rows = rn.bvd_rows(task, tmp_path)
    assert rows is not None
    assert rows["as_of"].iloc[0] == "2026-10-15"  # results-3086 for 17 Oct
    config = rn.model_config(rn.VARIANTS["renewalbvd"])
    data = rn.vintage_inputs(task, config, bvd=rows)
    assert data.bvd is not None
    assert rn.vintage_inputs(task, config).bvd is None
    assert "renewalbvd" in {s.name for s in list_models()}


def test_single_patch_fits_each_location_on_its_own(
    task: ForecastTask, monkeypatch: pytest.MonkeyPatch
) -> None:
    seen: list[tuple[str, ...]] = []

    def fake(self: rn.RenewalModel, t: ForecastTask) -> rn.RenewalRun:
        seen.append(t.locations)
        return _fake_run(t, self.variant)

    monkeypatch.setattr(rn.RenewalModel, "run", fake)
    model = get_model("renewalsp")
    assert isinstance(model, rn.RenewalModel)
    fit = model.forecast_fit(task)
    # One fit per province with enough cases, and one for the nation.
    rows = task.target.loc[task.target["vintage"] == task.vintage]
    cases = rows.groupby("location")["total"].sum()
    enough = [
        p for p in task.locations if cases[p] >= rn.SINGLE_PATCH_MIN_CASES
    ]
    few = [p for p in task.locations if p not in enough]
    assert few, "the fixture should have a province with few cases"
    assert sorted(seen) == sorted([(p,) for p in enough] + [(NATIONAL,)])
    assert set(fit.runs) == {*enough, NATIONAL}
    assert fit.unassigned is None
    assert fit.national is not None
    assert list(fit.national.columns) == ["target_end_date", "sample", "value"]
    assert set(fit.samples["location"]) == set(task.locations)
    # The national rows are the national fit, not the provinces' sum.
    samples = model.forecast(task)
    assert NATIONAL in set(samples["location"])


def test_fit_records_the_posterior_draw_behind_each_trajectory(
    task: ForecastTask, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def fake(self: rn.RenewalModel, t: ForecastTask) -> rn.RenewalRun:
        run = _fake_run(t, self.variant)
        draws = {**run.draws, "draw": np.arange(t.n_samples)[::-1]}
        return replace(run, draws=draws)

    monkeypatch.setattr(rn.RenewalModel, "run", fake)
    rn.fit_renewal(
        task, {"num_chains": 2}, tmp_path, variant="renewal", config_name="main"
    )
    saved = pd.read_csv(tmp_path / rn.TRAJECTORY_DRAW_FILE)
    assert list(saved.columns) == ["location", "sample", "draw"]
    assert set(saved["location"]) == {"all"}
    assert saved["sample"].tolist() == list(range(1, task.n_samples + 1))
    assert saved["draw"].tolist() == list(range(task.n_samples))[::-1]


def test_single_patch_has_no_depletion() -> None:
    config = rn.model_config(rn.VARIANTS["renewalsp"])
    assert config.depletion is False


def test_chain_method_defaults_to_vectorized(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import numpyro.infer

    seen: dict[str, Any] = {}

    class FakeMCMC:
        def __init__(self, kernel: Any, **kw: Any) -> None:
            del kernel
            seen.update(kw)

        def run(self, *args: Any, **kw: Any) -> None:
            del args, kw

    class FakeModel:
        def validate_data(self, **kw: Any) -> None:
            del kw

        def sample(self) -> None:
            return None

    import jax

    cleared: list[bool] = []
    monkeypatch.setattr(jax, "clear_caches", lambda: cleared.append(True))
    monkeypatch.setattr(numpyro.infer, "MCMC", FakeMCMC)
    monkeypatch.setattr(rn.engine(), "model_args", lambda *_: {})
    settings = rn.sampler_settings()
    assert settings["chain_method"] == "vectorized"
    rn.run_nuts(FakeModel(), None, settings)
    assert seen["chain_method"] == "vectorized"
    # Each fit compiles its own model, so earlier executables are dropped.
    assert cleared == [True]
    rn.run_nuts(FakeModel(), None, {**settings, "chain_method": "sequential"})
    assert seen["chain_method"] == "sequential"


def test_inputs_carry_the_publication_lag_from_earlier_files(
    target: pd.DataFrame, delays: pd.DataFrame
) -> None:
    from bdbvhubrenewal.evaluation.backtest import BacktestConfig, make_tasks
    from bdbvhubrenewal.forecasters.renewal import vintage_inputs
    from bdbvhubrenewal.model import MAIN

    tasks = {
        t.vintage: t
        for t in make_tasks(
            target.loc[target["kind"] == "real"],
            delays.assign(kind="real"),
            BacktestConfig(seed=3),
        )
    }
    late = vintage_inputs(tasks["2026-10-17"], MAIN)
    assert late.publication_lag_pmf is not None
    assert late.publication_lag_pmf.sum() == pytest.approx(1.0)
    assert vintage_inputs(tasks["2026-10-09"], MAIN).publication_lag_pmf is None


def test_lag_source_warns_when_the_prior_stands_in(
    target: pd.DataFrame, delays: pd.DataFrame
) -> None:
    from bdbvhubrenewal.evaluation.backtest import BacktestConfig, make_tasks
    from bdbvhubrenewal.forecasters.renewal import (
        lag_warnings,
        vintage_inputs,
    )
    from bdbvhubrenewal.model import MAIN

    tasks = {
        t.vintage: t
        for t in make_tasks(
            target.loc[target["kind"] == "real"],
            delays.assign(kind="real"),
            BacktestConfig(seed=3),
        )
    }
    empirical = replace(MAIN, publication_lag="empirical")
    early = vintage_inputs(tasks["2026-10-09"], empirical)
    late = vintage_inputs(tasks["2026-10-17"], empirical)
    (warning,) = lag_warnings([early], empirical)
    assert warning.startswith("publication_lag:")
    assert lag_warnings([late], empirical) == []
    # The default fixed lag never warns.
    assert lag_warnings([early], MAIN) == []


def test_report_cutoffs_are_days_from_the_model_start(
    task: ForecastTask,
) -> None:
    task = replace(task, report_by=("2026-10-24", "2026-10-31"))
    data = rn.vintage_inputs(task, rn.model_config(rn.VARIANTS["renewal"]))
    cuts = rn.report_cutoffs(task, data)
    start = pd.Timestamp(data.start_date)
    assert cuts == {(pd.Timestamp(d) - start).days: d for d in task.report_by}


@pytest.mark.parametrize("name", ["renewal", "renewalsp"])
def test_forecast_gives_counts_as_reported_later(
    task: ForecastTask, monkeypatch: pytest.MonkeyPatch, name: str
) -> None:
    monkeypatch.setattr(
        rn.RenewalModel, "run", lambda self, t: _fake_run(t, self.variant)
    )
    task = replace(task, report_by=("2026-10-24",))
    samples = get_model(name).forecast(task)
    check_samples(samples, task)
    later = samples.loc[samples["as_of"] == "2026-10-24"]
    assert later["target_end_date"].max() <= "2026-10-24"
    assert not later.empty
    final = samples.loc[samples["as_of"].isna()]
    assert set(final["target_end_date"]) == set(task.target_end_dates)
    # Without later vintages there is no as_of column.
    plain = get_model(name).forecast(replace(task, report_by=()))
    assert "as_of" not in plain


def test_forecast_records_a_summary_of_each_fit(
    task: ForecastTask, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        rn.RenewalModel, "run", lambda self, t: _fake_run(t, self.variant)
    )
    model = get_model("renewal")
    assert isinstance(model, rn.RenewalModel)
    model.sampler = {**model.sampler, "num_chains": 2, "num_warmup": 300}
    model.forecast(task)
    (row,) = model.fit_diagnostics
    assert row["fit"] == "all"
    assert row["num_chains"] == 2
    assert row["num_warmup"] == 300
    assert row["seconds"] == 1.5
    assert row["max_rhat"] > 0.9
    assert row["depth_share"] == 0.0
    assert row["divergent_share"] == 0.0
