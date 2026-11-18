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
    # (dates on other weekdays have other daily horizons), not for each
    # date and task.
    assert pipe.calls <= per_date / 2
