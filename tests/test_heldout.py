from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
import yaml

from bdbv_forecasts.backtest import BacktestConfig
from bdbv_forecasts.heldout import (
    HeldOutError,
    ModelChoice,
    heldout_pairs,
    load_model_choices,
    model_for_vintage,
    run_heldout,
)

CHOICES = (
    ModelChoice(date="2026-10-10", model="persistence"),
    ModelChoice(date="2026-10-20", model="renewal"),
)


def test_load_model_choices(tmp_path: Path) -> None:
    path = tmp_path / "model-choice.yml"
    path.write_text(
        yaml.safe_dump(
            {
                "choices": [
                    {"date": "2026-10-20", "model": "renewal"},
                    {
                        "date": "2026-10-10",
                        "model": "persistence",
                        "summary": "Baseline only.",
                    },
                ]
            }
        )
    )
    choices = load_model_choices(path)
    assert [c.model for c in choices] == ["persistence", "renewal"]
    assert choices[0].summary == "Baseline only."
    assert load_model_choices(tmp_path / "missing.yml") == ()


def test_model_choices_need_unique_dates(tmp_path: Path) -> None:
    path = tmp_path / "model-choice.yml"
    path.write_text(
        "choices:\n"
        "  - {date: 2026-10-10, model: a}\n"
        "  - {date: 2026-10-10, model: b}\n"
    )
    with pytest.raises(ValueError, match="date"):
        load_model_choices(path)


def test_model_for_vintage() -> None:
    assert model_for_vintage(CHOICES, "2026-10-09") is None
    assert model_for_vintage(CHOICES, "2026-10-10") is None
    assert model_for_vintage(CHOICES, "2026-10-17") == "persistence"
    assert model_for_vintage(CHOICES, "2026-10-24") == "renewal"


def test_heldout_pairs() -> None:
    vintages = ["2026-08-18", "2026-10-09", "2026-10-17", "2026-10-24"]
    pairs = heldout_pairs(CHOICES[:1], vintages)
    assert list(pairs.itertuples(index=False, name=None)) == [
        ("2026-10-17", "persistence", "2026-10-24"),
        ("2026-10-24", "persistence", None),
    ]


def test_heldout_needs_flag_and_choice(
    target: pd.DataFrame,
    delays: pd.DataFrame,
    config: BacktestConfig,
    tmp_path: Path,
) -> None:
    with pytest.raises(HeldOutError, match="evaluate_holdout"):
        run_heldout(target, delays, config)
    with pytest.raises(HeldOutError, match="model choice"):
        run_heldout(target, delays, config, evaluate_holdout=True)


def test_heldout_scores_each_forecast_under_its_model(
    target: pd.DataFrame,
    delays: pd.DataFrame,
    config: BacktestConfig,
    tmp_path: Path,
) -> None:
    config.model_choice.write_text(
        "choices:\n  - {date: 2026-10-10, model: persistence}\n"
    )
    out = tmp_path / "out"
    scores = run_heldout(
        target, delays, config, evaluate_holdout=True, out_dir=out
    )
    # Only the 2026-10-17 forecast has later data.
    assert set(scores["vintage"]) == {"2026-10-17"}
    assert set(scores["truth_vintage"]) == {"2026-10-24"}
    assert set(scores["chosen_model"]) == {"persistence"}
    assert set(scores["model"]) == {"persistence"}
    assert set(scores["kind"]) == {"real"}
    record = yaml.safe_load((out / "heldout-run.yml").read_text())
    assert len(record["model_choice_sha256"]) == 64
    assert record["pending"] == [
        {"vintage": "2026-10-24", "model": "persistence"}
    ]
    forecasts = pd.read_csv(out / "heldout-forecasts.csv.gz")
    assert set(forecasts["vintage"]) == {"2026-10-17", "2026-10-24"}
