from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pandas as pd
import pytest
import yaml

from bdbvhubrenewal.evaluation.backtest import BacktestConfig
from bdbvhubrenewal.evaluation.heldout import (
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
    assert pairs["vintage"].tolist() == ["2026-10-17", "2026-10-24"]
    assert pairs["model"].tolist() == ["persistence", "persistence"]
    # Missing is None or NaN depending on the pandas string dtype.
    assert pairs["truth_vintage"].iloc[0] == "2026-10-24"
    assert pd.isna(pairs["truth_vintage"].iloc[1])


def test_heldout_needs_flag_and_choice(
    target: pd.DataFrame,
    delays: pd.DataFrame,
    config: BacktestConfig,
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
    # Both baselines are scored on every held-out vintage.
    assert set(scores["model"]) == {
        "persistence",
        "persistencenow",
        "persistdaily",
    }
    assert set(scores["kind"]) == {"real"}
    record = yaml.safe_load((out / "heldout-run.yml").read_text())
    assert len(record["model_choice_sha256"]) == 64
    assert record["pending"] == [
        {"vintage": "2026-10-24", "model": "persistence"}
    ]
    forecasts = pd.read_csv(out / "heldout-forecasts.csv.gz")
    assert set(forecasts["vintage"]) == {"2026-10-17", "2026-10-24"}


def test_heldout_scores_both_baselines_beside_a_fitted_choice(
    target: pd.DataFrame,
    delays: pd.DataFrame,
    config: BacktestConfig,
    tmp_path: Path,
) -> None:
    config.model_choice.write_text(
        "choices:\n  - {date: 2026-10-10, model: renewal}\n"
    )
    scores = run_heldout(
        target,
        delays,
        config,
        evaluate_holdout=True,
        out_dir=tmp_path,
        fit=False,
    )
    assert set(scores["model"]) == {
        "persistence",
        "persistencenow",
        "persistdaily",
    }
    assert set(scores["chosen_model"]) == {"renewal"}


def test_heldout_also_runs_the_other_variants(
    target: pd.DataFrame,
    delays: pd.DataFrame,
    config: BacktestConfig,
    tmp_path: Path,
) -> None:
    from bdbvhubrenewal.forecasters import Persistence, register_model
    from bdbvhubrenewal.forecasters.base import _REGISTRY

    register_model(
        "shadow",
        lambda: Persistence(name="shadow"),
        "A second cheap model.",
        replace=True,
    )
    config.model_choice.write_text(
        "choices:\n  - {date: 2026-10-10, model: persistence}\n"
    )
    scores = run_heldout(
        target,
        delays,
        config,
        evaluate_holdout=True,
        out_dir=tmp_path,
        models=("shadow",),
    )
    try:
        assert set(scores["model"]) == {
            "persistence",
            "persistencenow",
            "persistdaily",
            "shadow",
        }
        # Each score still records the model chosen at the time.
        assert set(scores["chosen_model"]) == {"persistence"}
    finally:
        _REGISTRY.pop("shadow")


def test_heldout_keeps_and_rescores_other_models_forecasts(
    target: pd.DataFrame,
    delays: pd.DataFrame,
    config: BacktestConfig,
    tmp_path: Path,
) -> None:
    config.model_choice.write_text(
        "choices:\n  - {date: 2026-10-10, model: persistence}\n"
    )
    run_heldout(target, delays, config, evaluate_holdout=True, out_dir=tmp_path)
    path = tmp_path / "heldout-forecasts.csv.gz"
    fc = pd.read_csv(path)
    # A fitted variant's forecasts, as a run on archie would leave them.
    pd.concat([fc, fc.assign(model="other")]).to_csv(path, index=False)
    scores = run_heldout(
        target, delays, config, evaluate_holdout=True, out_dir=tmp_path
    )
    assert set(scores["model"]) == {
        "persistence",
        "persistencenow",
        "persistdaily",
        "other",
    }
    assert set(pd.read_csv(path)["model"]) == {
        "persistence",
        "persistencenow",
        "persistdaily",
        "other",
    }


def test_heldout_without_fits_reads_committed_forecasts(
    target: pd.DataFrame,
    delays: pd.DataFrame,
    config: BacktestConfig,
    tmp_path: Path,
) -> None:
    from bdbvhubrenewal.forecasters import Persistence, register_model
    from bdbvhubrenewal.forecasters.base import _REGISTRY

    class Slow(Persistence):
        def forecast(self, task):  # type: ignore[no-untyped-def]
            del task
            raise AssertionError("a fitted model was refitted")

    register_model(
        "slowfit", lambda: Slow(name="slowfit"), "Fitted.", fitted=True
    )
    try:
        config.model_choice.write_text(
            "choices:\n  - {date: 2026-10-10, model: slowfit}\n"
        )
        # Committed forecasts of the fitted model, as archie leaves them.
        fc = run_heldout(
            target,
            delays,
            replace(config, model_choice=_persistence_choice(tmp_path)),
            evaluate_holdout=True,
            out_dir=tmp_path / "seed",
        )
        seeded = pd.read_csv(tmp_path / "seed" / "heldout-forecasts.csv.gz")
        out = tmp_path / "out"
        out.mkdir()
        seeded.assign(model="slowfit", chosen_model="slowfit").to_csv(
            out / "heldout-forecasts.csv.gz", index=False
        )
        scores = run_heldout(
            target,
            delays,
            config,
            evaluate_holdout=True,
            out_dir=out,
            fit=False,
        )
    finally:
        _REGISTRY.pop("slowfit")
    assert not fc.empty
    assert set(scores["model"]) == {
        "persistence",
        "persistencenow",
        "persistdaily",
        "slowfit",
    }
    assert set(scores["chosen_model"]) == {"slowfit"}


def _persistence_choice(tmp_path: Path) -> Path:
    path = tmp_path / "persistence-choice.yml"
    path.write_text("choices:\n  - {date: 2026-10-10, model: persistence}\n")
    return path


def test_heldout_with_no_vintage_after_the_choice(
    target: pd.DataFrame,
    delays: pd.DataFrame,
    config: BacktestConfig,
    tmp_path: Path,
) -> None:
    # A choice after the latest vintage holds nothing out yet.
    config.model_choice.write_text(
        "choices:\n  - {date: 2026-12-31, model: persistencenow}\n"
    )
    scores = run_heldout(
        target, delays, config, evaluate_holdout=True, out_dir=tmp_path
    )
    assert scores.empty
    record = yaml.safe_load((tmp_path / "heldout-run.yml").read_text())
    assert record["scored"] == []
    assert record["pending"] == []
