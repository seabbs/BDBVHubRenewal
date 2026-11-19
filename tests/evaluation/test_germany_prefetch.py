"""German nowcasts predict Chronos-2 for every date in a few passes."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from bdbvhubrenewal.data.germany import read_triangle
from bdbvhubrenewal.evaluation.germany import germany_forecasts
from bdbvhubrenewal.forecasters import foundation

pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("torch") is None, reason="needs torch"
)

FIXTURE = (
    Path(__file__).resolve().parents[2]
    / "tests"
    / "fixtures"
    / "germany"
    / "germany_covid19_hosp.csv.gz"
)
DATES = ["2021-10-04", "2021-10-05", "2021-10-06", "2021-10-09"]
MODELS = ["fm_chronos_daily_ragged", "fm_chronos_weekly_ragged"]


class _Pipeline:
    """Quantiles set by each group's own values; records each pass."""

    quantiles = (0.1, 0.5, 0.9)

    def __init__(self) -> None:
        self.calls = 0

    def predict_quantiles(
        self, inputs, prediction_length, quantile_levels, **_kw
    ):
        import torch

        self.calls += 1
        out = []
        for g in inputs:
            own = np.atleast_2d(g["target"]).mean(axis=1)
            week = np.arange(prediction_length)
            level = np.asarray(quantile_levels)
            q = own[:, None, None] + week[None, :, None] + level[None, None]
            out.append(torch.tensor(q, dtype=torch.float32))
        return out, None


def test_dates_share_chronos_passes(monkeypatch: pytest.MonkeyPatch) -> None:
    triangle = read_triangle(FIXTURE)
    pipe = _Pipeline()
    monkeypatch.setattr(foundation, "_chronos", lambda *_: pipe)
    alone = pd.concat(
        [germany_forecasts(MODELS, triangle, [d], 30, 1) for d in DATES],
        ignore_index=True,
    )
    per_date = pipe.calls
    pipe.calls = 0
    together = germany_forecasts(MODELS, triangle, DATES, 30, 1)
    pd.testing.assert_frame_equal(
        together.reset_index(drop=True),
        alone.sort_values(
            [*together.columns[:5], "quantile"], kind="stable"
        ).reset_index(drop=True),
    )
    # A pass for each model, task schema (states, age groups) and horizon
    # (dates and offsets on other weekdays have other daily horizons), not
    # for each date and task.
    assert pipe.calls < per_date


def test_tasks_are_built_once_and_reused(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # Every checkpoint scores the same German tasks: built once into
    # `tasks_dir`, then read back, with the same forecasts.
    from bdbvhubrenewal.evaluation import germany as de

    triangle = read_triangle(FIXTURE)
    monkeypatch.setattr(foundation, "_chronos", lambda *_: _Pipeline())
    first = germany_forecasts(MODELS, triangle, DATES, 30, 1, tmp_path)
    # gzip-compressed: a date's 7 offsets would be 7 times the plain
    # pickle's disk.
    assert sorted(p.name for p in tmp_path.iterdir()) == [
        f"{d}.pkl.gz" for d in sorted(DATES)
    ]

    def no_build(*_a, **_k):
        raise AssertionError("tasks rebuilt")

    monkeypatch.setattr(de, "nowcast_tasks", no_build)
    again = germany_forecasts(MODELS, triangle, DATES, 30, 1, tmp_path)
    pd.testing.assert_frame_equal(first, again)


def test_the_task_store_is_keyed_on_its_inputs(tmp_path: Path) -> None:
    from bdbvhubrenewal.evaluation.checkpoint import german_tasks_dir

    data = tmp_path / "triangle.csv"
    data.write_text("a\n")
    one = german_tasks_dir(tmp_path, data, 1000, 1)
    assert one == german_tasks_dir(tmp_path, data, 1000, 1)
    assert one != german_tasks_dir(tmp_path, data, 1000, 2)
    assert one != german_tasks_dir(tmp_path, data, 100, 1)
    data.write_text("ab\n")
    assert one != german_tasks_dir(tmp_path, data, 1000, 1)


def test_the_task_store_key_covers_the_code_the_tasks_depend_on(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import dataclasses
    import inspect

    from bdbvhubrenewal.data import dates
    from bdbvhubrenewal.evaluation import checkpoint as ck
    from bdbvhubrenewal.forecasters import base

    data = tmp_path / "triangle.csv"
    data.write_text("a\n")
    one = ck.german_tasks_dir(tmp_path, data, 1000, 1)
    real = inspect.getsource

    def edited(obj):
        text = real(obj)
        return text + "# edit" if obj is dates else text

    monkeypatch.setattr(inspect, "getsource", edited)
    assert ck.german_tasks_dir(tmp_path, data, 1000, 1) != one
    monkeypatch.setattr(inspect, "getsource", real)

    @dataclasses.dataclass(frozen=True)
    class Wider(base.ForecastTask):
        extra: int = 0

    monkeypatch.setattr(base, "ForecastTask", Wider)
    assert ck.german_tasks_dir(tmp_path, data, 1000, 1) != one
