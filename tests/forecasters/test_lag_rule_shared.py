"""One lag rule for training and prediction (forecaster.lag_count)."""

import importlib.util
import sys
from pathlib import Path

import pandas as pd

from nowcastfoundation import finetune, forecaster

ROOT = Path(__file__).parents[2]


def _counts(days: int = 60) -> pd.DataFrame:
    """Daily onsets of 10 cases at two places, half reported a day late."""
    rows = []
    first = pd.Timestamp("2026-08-01")
    for t in range(days):
        ref = first + pd.Timedelta(days=t)
        for place in ("A", "B"):
            for lag, n in ((0, 5), (1, 5)):
                rows.append(
                    {
                        "location": place,
                        "reference_date": ref.date().isoformat(),
                        "report_date": (ref + pd.Timedelta(days=lag))
                        .date()
                        .isoformat(),
                        "count": n,
                    }
                )
    return pd.DataFrame(rows)


def test_training_blocks_take_their_lags_from_the_shared_rule(monkeypatch):
    calls = []
    real = forecaster.lag_count

    def spy(*args, **kwargs):
        calls.append(real(*args, **kwargs))
        return calls[-1]

    monkeypatch.setattr(forecaster, "lag_count", spy)
    cutoffs = [
        (pd.Timestamp("2026-08-01") + pd.Timedelta(days=d)).date().isoformat()
        for d in range(1, 40)
    ]
    block = finetune.training_block(
        _counts(), "daily", ("CD", "A", "B"), cutoffs
    )
    assert calls, "training_block did not call forecaster.lag_count"
    places = len(block["places"])
    assert block["n_covariates"] == places * calls[-1]


def test_the_trainer_draws_no_lag_count() -> None:
    spec = importlib.util.spec_from_file_location(
        "finetune_chronos", ROOT / "scripts" / "finetune_chronos.py"
    )
    sys.path.insert(0, str(ROOT / "scripts"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    assert not hasattr(mod, "MAX_LAGS")
    assert not hasattr(mod, "LAG_CAP")
    source = (ROOT / "scripts" / "finetune_chronos.py").read_text()
    assert "rng.choice(MAX_LAGS" not in source


def test_forecasts_take_their_lags_from_the_same_rule(monkeypatch):
    from nowcastutils.base import get_model

    calls = []
    real = forecaster.lag_count

    def spy(*args, **kwargs):
        calls.append(real(*args, **kwargs))
        return calls[-1]

    monkeypatch.setattr(forecaster, "lag_count", spy)
    rows = finetune.releases(
        _counts(),
        "daily",
        [
            (pd.Timestamp("2026-08-01") + pd.Timedelta(days=d))
            .date()
            .isoformat()
            for d in range(1, 40)
        ],
    )
    model = get_model("fm_chronos_daily_ragged")
    tri = model._triangle(rows, ("A", "B"), rows["cutoff"].max())
    assert calls, "the forecaster did not call forecaster.lag_count"
    assert tri.partial.shape[2] == calls[-1]
