"""The Chronos-2 forecasters, with a stand-in for Chronos-2."""

from __future__ import annotations

import dataclasses
import importlib.util

import numpy as np
import pandas as pd
import pytest

from bdbvhubrenewal.evaluation.backtest import BacktestConfig, make_tasks
from bdbvhubrenewal.forecasters import (
    ForecastTask,
    FoundationModel,
    ModelUnavailableError,
    check_samples,
    foundation,
    get_model,
    list_models,
)
from bdbvhubrenewal.forecasters.foundation import (
    CACHE_ENV,
    FT_CONTEXT,
    FT_HORIZON,
    LAYOUTS,
    N_DELAYS,
    View,
    chronos_sampler,
    cov_examples,
    delay_triangle,
    fine_tuned_checkpoint,
    history_examples,
    quantile_draws,
    report_series,
)

MODELS = ("fm_complete", "fm_delay_series", "fm_delay_cov", "fm_delay_cov_ft")


def _constant(value: float):
    """A sampler whose every draw is `value`."""

    def sample(inputs, horizon, n, seed, _checkpoint):
        assert seed >= 0
        return [
            np.full((n, np.atleast_2d(x["target"]).shape[0], horizon), value)
            for x in inputs
        ]

    return sample


def _recording(store: list):
    """A sampler that keeps its inputs and checkpoint and draws zeros."""

    def sample(inputs, horizon, n, seed, checkpoint):
        store.append((inputs, horizon, checkpoint))
        return _constant(0.0)(inputs, horizon, n, seed, checkpoint)

    return sample


@pytest.fixture(scope="module")
def task(
    target: pd.DataFrame, delays: pd.DataFrame, daily: pd.DataFrame
) -> ForecastTask:
    rows = target.loc[target["vintage"] == "2026-10-24"]
    real = target.loc[target["kind"] == "real"]
    return make_tasks(
        rows, delays, BacktestConfig(seed=3), history=real, daily=daily
    )[0]


@pytest.fixture(autouse=True)
def _no_fine_tuning(monkeypatch: pytest.MonkeyPatch) -> None:
    """Stand in for the fine-tuned checkpoint of every task."""
    monkeypatch.setattr(
        foundation, "fine_tuned_checkpoint", lambda _task, _name: "ckpt"
    )


def _reported(task: ForecastTask, out: pd.DataFrame) -> np.ndarray:
    keys = pd.MultiIndex.from_frame(out[["location", "target_end_date"]])
    return (
        task.target.set_index(["location", "target_end_date"])["total"]
        .reindex(keys)
        .fillna(0.0)
        .to_numpy()
    )


def test_registered_as_fitted_models() -> None:
    specs = {s.name: s for s in list_models()}
    for name in MODELS:
        assert specs[name].fitted
        assert not specs[name].baseline
        model = get_model(name)
        assert isinstance(model, FoundationModel)
        assert model.layout in LAYOUTS
        # The fit store keys a fit on the repr, so it must not hold an
        # address.
        assert "0x" not in repr(model)


def test_triangle_classes_and_complete_week() -> None:
    delays = pd.DataFrame(
        {
            "location": ["A", "A", "A", "B", "B", "C"],
            "onset_week": ["2026-10-03"] * 3 + ["2026-10-10"] * 3,
            "delay": [-2, 6, 7, 20, 7 * N_DELAYS, 3],
        }
    )
    weeks = ("2026-10-03", "2026-10-10", "2026-10-17")
    tri = delay_triangle(delays, "2026-10-24", ("A", "B"), weeks)
    assert tri.counts.shape == (2, 3, N_DELAYS)
    # -2 and 6 days are class 0 and 7 days class 1; 7K days is dropped
    # and place C is not asked for.
    assert tri.counts[0, 0, :2].tolist() == [2.0, 1.0]
    assert tri.counts[1, 1, 2] == 1.0
    assert tri.counts.sum() == 4.0
    # Week ending 10-17 has its last delay-0 report day on 10-23.
    assert weeks[tri.last] == "2026-10-17"
    tri = delay_triangle(delays, "2026-10-22", ("A", "B"), weeks)
    assert weeks[tri.last] == "2026-10-10"


def test_report_series_puts_each_class_on_the_report_clock() -> None:
    delays = pd.DataFrame(
        {
            "location": ["A", "A", "A"],
            "onset_week": ["2026-10-03", "2026-10-03", "2026-10-10"],
            "delay": [0, 8, 1],
        }
    )
    weeks = ("2026-10-03", "2026-10-10", "2026-10-17", "2026-10-24")
    tri = delay_triangle(delays, "2026-10-31", ("A",), weeks, 2)
    series = report_series(tri)
    assert series.shape == (1, 2, tri.last + 1)
    # Class 0 at its onset week, class 1 one week later.
    assert series[0, 0].tolist() == [1.0, 1.0, 0.0, 0.0]
    assert series[0, 1].tolist() == [0.0, 1.0, 0.0, 0.0]


def test_quantile_draws_invert_the_quantile_function() -> None:
    levels = np.array([0.1, 0.5, 0.9])
    values = np.array([[[0.0, 1.0, 3.0]], [[2.0, 2.0, 2.0]]])
    u = np.array([[0.5], [0.3]])
    out = quantile_draws(levels, values, u)
    assert out.shape == (2, 1)
    assert out[0, 0] == pytest.approx(1.0)
    assert out[1, 0] == pytest.approx(2.0)
    # Outside the levels the draw is the outermost quantile.
    low = quantile_draws(levels, values[:1], np.array([[0.01]]))
    assert low[0, 0] == pytest.approx(0.0)


@pytest.mark.parametrize("name", MODELS)
def test_draws_are_counts_at_least_the_reported(
    name: str, task: ForecastTask
) -> None:
    reported = None
    for value in (-50.0, 0.0, 3.0):
        model = dataclasses.replace(get_model(name), sample=_constant(value))
        out = model.forecast(task)
        check_samples(out, task)
        values = out["value"].to_numpy()
        assert (values >= 0).all()
        assert np.array_equal(values, np.round(values))
        reported = _reported(task, out)
        assert (values >= reported).all()
    # Draws above the reported counts raise the forecast past them.
    assert (values > reported).any()


@pytest.mark.parametrize("name", MODELS)
def test_low_draws_give_the_reported_counts(
    name: str, task: ForecastTask
) -> None:
    model = dataclasses.replace(get_model(name), sample=_constant(-50.0))
    out = model.forecast(task)
    assert np.array_equal(out["value"].to_numpy(), _reported(task, out))


@pytest.mark.parametrize("name", MODELS)
def test_maps_final_counts_through_a_delay(
    name: str, task: ForecastTask
) -> None:
    model = dataclasses.replace(get_model(name), sample=_constant(0.0))
    model.forecast(task)
    summary = model.delay_summary
    assert summary is not None
    assert set(summary["model"]) == {name}
    assert set(summary["quantity"]) == {"reported_share"}


def test_delay_cov_masks_cells_not_yet_reported(task: ForecastTask) -> None:
    seen: list = []
    model = dataclasses.replace(
        get_model("fm_delay_cov"), sample=_recording(seen)
    )
    model.forecast(task)
    inputs, horizon, _ = seen[0]
    first = inputs[0]
    past = np.stack(list(first["past_covariates"].values()))
    future = np.stack(list(first["future_covariates"].values()))
    assert past.shape == (N_DELAYS, first["target"].shape[-1])
    assert future.shape == (N_DELAYS, horizon)
    # The context is complete; the future holds the reported cells of
    # each delay class, fewer for longer delays, then nothing.
    assert not np.isnan(past).any()
    known = (~np.isnan(future)).sum(axis=1)
    assert (np.diff(known) <= 0).all()
    assert known[0] == N_DELAYS - 1
    assert known[-1] == 0
    assert np.isnan(future[:, N_DELAYS - 1 :]).all()


def test_complete_forecasts_from_the_last_complete_week(
    task: ForecastTask,
) -> None:
    complete: list = []
    cov: list = []
    dataclasses.replace(
        get_model("fm_complete"), sample=_recording(complete)
    ).forecast(task)
    dataclasses.replace(
        get_model("fm_delay_cov"), sample=_recording(cov)
    ).forecast(task)
    (a, h_a, _), (b, h_b, _) = complete[0], cov[0]
    assert h_a == h_b
    assert len(a) == len(b) == len(task.locations)
    assert "past_covariates" not in a[0]
    assert np.array_equal(a[0]["target"], b[0]["target"])


@pytest.mark.parametrize("name", ["fm_delay_series", "fm_delay_cov"])
def test_no_delay_data_is_unavailable(name: str, task: ForecastTask) -> None:
    bare = dataclasses.replace(task, delays=None)
    model = dataclasses.replace(get_model(name), sample=_constant(0.0))
    with pytest.raises(ModelUnavailableError, match="delay line data"):
        model.forecast(bare)


@pytest.mark.skipif(
    importlib.util.find_spec("chronos") is not None,
    reason="Chronos-2 is installed",
)
def test_chronos_missing_is_unavailable() -> None:
    with pytest.raises(ModelUnavailableError, match="extra chronos"):
        chronos_sampler([{"target": np.zeros((1, 5))}], 3, 10, 0)


class _FakePipeline:
    """Quantiles that grow with the level and the week."""

    quantiles = (0.1, 0.5, 0.9)

    def predict_quantiles(self, groups, prediction_length, quantile_levels):
        import torch

        out = []
        for g in groups:
            n_var = np.atleast_2d(g["target"]).shape[0]
            week = torch.arange(prediction_length, dtype=torch.float32)
            level = torch.tensor(quantile_levels, dtype=torch.float32)
            q = week[:, None] + level[None, :]
            out.append(q.expand(n_var, -1, -1))
        return out, None


@pytest.mark.skipif(
    importlib.util.find_spec("torch") is None, reason="needs torch"
)
def test_trajectories_keep_their_level_across_weeks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        foundation, "_chronos", lambda _checkpoint: _FakePipeline()
    )
    groups = [{"target": np.zeros((2, 5))}, {"target": np.zeros((1, 5))}]
    draws = chronos_sampler(groups, 4, 50, 1)
    assert [d.shape for d in draws] == [(50, 2, 4), (50, 1, 4)]
    # Each week is the week index plus one level per trajectory.
    level = draws[0] - np.arange(4)
    assert np.allclose(level, level[:, :, :1])
    assert ((level > 0.1 - 1e-6) & (level < 0.9 + 1e-6)).all()
    assert not np.allclose(level[:, 0], level[:, 1])


def test_fine_tuned_model_samples_from_its_checkpoint(
    task: ForecastTask,
) -> None:
    for name, want in (("fm_delay_cov", None), ("fm_delay_cov_ft", "ckpt")):
        seen: list = []
        model = dataclasses.replace(get_model(name), sample=_recording(seen))
        model.forecast(task)
        assert seen[0][2] == want


def _view(reported: np.ndarray, first: str, last: int) -> View:
    weeks = pd.date_range(first, periods=reported.shape[1], freq="7D")
    counts = np.repeat(reported[:, :, None], N_DELAYS, axis=2)
    return View(list(weeks.strftime("%Y-%m-%d")), reported, counts, last)


def test_cov_examples_split_at_the_vintage() -> None:
    n_weeks = 60
    as_of = _view(np.full((2, n_weeks), 1.0), "2026-01-03", 30)
    as_of.reported[1] = 0.0
    final = _view(np.full((2, n_weeks + 1), 5.0), "2025-12-27", 50)
    final.reported[1] = 0.0
    out = cov_examples(as_of, final)
    # The second province has no cases in the window.
    assert len(out) == 1
    target = out[0]["target"]
    start = as_of.n_context
    assert target.shape == (FT_CONTEXT + FT_HORIZON,)
    # Reported to c, final after it.
    assert (target[:FT_CONTEXT][-start:] == 1.0).all()
    assert np.isnan(target[: FT_CONTEXT - start]).all()
    assert (target[FT_CONTEXT:] == 5.0).all()
    past = np.stack(list(out[0]["past_covariates"].values()))
    assert past.shape == (N_DELAYS, FT_CONTEXT + FT_HORIZON)
    # Class k is reported for k fewer weeks after c, as the vintage shows.
    known = (~np.isnan(past[:, FT_CONTEXT:])).sum(axis=1)
    assert known.tolist() == list(range(N_DELAYS - 1, -1, -1))
    assert set(out[0]["future_covariates"].values()) == {None}


def test_cov_examples_leave_final_weeks_not_complete_out() -> None:
    as_of = _view(np.ones((1, 60)), "2026-01-03", 30)
    final = _view(np.full((1, 60), 5.0), "2026-01-03", 30 + 3)
    target = cov_examples(as_of, final)[0]["target"]
    assert (target[FT_CONTEXT : FT_CONTEXT + 3] == 5.0).all()
    assert np.isnan(target[FT_CONTEXT + 3 :]).all()


def test_history_examples_ignore_later_releases(
    task: ForecastTask, target: pd.DataFrame, delays: pd.DataFrame
) -> None:
    out = history_examples(task)
    assert out
    later_target = target.loc[target["vintage"] == "2026-11-01"]
    later_delays = delays.loc[delays["vintage"] == "2026-11-01"]
    assert task.target_history is not None
    assert task.delay_history is not None
    leaked = dataclasses.replace(
        task,
        target_history=pd.concat([task.target_history, later_target]),
        delay_history=pd.concat([task.delay_history, later_delays]),
    )
    again = history_examples(leaked)
    assert len(again) == len(out)
    for a, b in zip(out, again, strict=True):
        assert np.array_equal(a["target"], b["target"], equal_nan=True)


def test_fine_tuned_checkpoint_is_reused(
    task: ForecastTask, tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.undo()
    monkeypatch.setenv(CACHE_ENV, str(tmp_path))
    saved = tmp_path / "fm_delay_cov_ft" / task.vintage / "finetuned-ckpt"
    saved.mkdir(parents=True)

    def no_model(_checkpoint):
        raise AssertionError("fine-tuned again")

    monkeypatch.setattr(foundation, "_chronos", no_model)
    assert fine_tuned_checkpoint(task, "fm_delay_cov_ft") == str(saved)
