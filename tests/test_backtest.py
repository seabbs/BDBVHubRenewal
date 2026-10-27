from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
import yaml

from bdbv_forecasts.backtest import (
    BacktestConfig,
    HeldOutError,
    forecast_weeks,
    load_config,
    run_development,
    run_heldout,
    run_recovery,
    saturdays,
)
from bdbv_forecasts.models import get_model

ROOT = Path(__file__).resolve().parents[1]


def test_load_config() -> None:
    cfg = load_config(ROOT / "config" / "backtest.toml")
    assert cfg.n_samples == 100
    assert cfg.heldout_reference_dates == ("2026-10-24",)
    assert cfg.reconstructed_dates[0] == "2026-06-06"
    assert all(pd.Timestamp(d).dayofweek == 5 for d in cfg.simulated_dates)


def test_forecast_weeks() -> None:
    weeks = forecast_weeks("2026-10-31", BacktestConfig())
    assert weeks[0] == "2026-10-03"
    assert weeks[-1] == "2026-11-21"
    assert len(weeks) == 8


def test_saturdays() -> None:
    assert saturdays("2026-10-01", "2026-10-17") == (
        "2026-10-03",
        "2026-10-10",
        "2026-10-17",
    )


def test_development_never_scores_heldout(
    target: pd.DataFrame, delays: pd.DataFrame, config: BacktestConfig
) -> None:
    forecasts, scores = run_development(
        [get_model("persistence")], target, delays, config
    )
    assert "2026-10-24" not in set(forecasts["reference_date"])
    assert "2026-10-24" not in set(scores["reference_date"])
    # Reconstructed vintages with a reference date on or after the
    # held-out date are not built.
    rec = forecasts.loc[forecasts["kind"] == "reconstructed"]
    assert set(rec["vintage"]) == {"2026-08-29", "2026-10-10"}
    assert set(scores["kind"]) == {"real", "reconstructed"}
    # Only weeks at least two weeks before the truth's epi week are scored.
    assert scores["target_end_date"].max() <= "2026-10-10"
    assert (scores["vintage"] < "2026-10-24").all()
    assert (scores["n_samples"] == 100).all()


def test_heldout_needs_flag_and_choice_record(
    target: pd.DataFrame,
    delays: pd.DataFrame,
    config: BacktestConfig,
    tmp_path: Path,
) -> None:
    models = [get_model("persistence")]
    with pytest.raises(HeldOutError, match="evaluate_holdout"):
        run_heldout(models, target, delays, config)
    with pytest.raises(HeldOutError, match="model choice"):
        run_heldout(models, target, delays, config, evaluate_holdout=True)
    config.model_choice.write_text("proposed_model: persistence\n")
    scores = run_heldout(
        models,
        target,
        delays,
        config,
        evaluate_holdout=True,
        out_dir=tmp_path / "out",
    )
    assert set(scores["reference_date"]) == {"2026-10-24"}
    assert set(scores["kind"]) == {"real"}
    record = yaml.safe_load((tmp_path / "out" / "heldout-run.yml").read_text())
    assert len(record["model_choice_sha256"]) == 64


def test_recovery_on_simulated_vintages(config: BacktestConfig) -> None:
    _, scores = run_recovery([get_model("persistence")], config)
    assert set(scores["kind"]) == {"simulated"}
    assert len(scores) > 0
