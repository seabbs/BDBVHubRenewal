from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
import yaml

from bdbvhubrenewal.data.zones import province_of
from bdbvhubrenewal.evaluation.backtest import BacktestConfig, merge_by_model
from bdbvhubrenewal.evaluation.heldout import (
    HeldOutError,
    ModelChoice,
    heldout_pairs,
    load_model_choices,
    model_for_vintage,
    run_heldout,
)
from bdbvhubrenewal.forecasters.renewal import DELAY_SUMMARY_FILE

CHOICES = (
    ModelChoice(date="2026-10-10", model="randomwalk"),
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
                        "model": "randomwalk",
                        "summary": "Baseline only.",
                    },
                ]
            }
        )
    )
    choices = load_model_choices(path)
    assert [c.model for c in choices] == ["randomwalk", "renewal"]
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
    assert model_for_vintage(CHOICES, "2026-10-17") == "randomwalk"
    assert model_for_vintage(CHOICES, "2026-10-24") == "renewal"


def test_heldout_pairs() -> None:
    vintages = ["2026-08-18", "2026-10-09", "2026-10-17", "2026-10-24"]
    pairs = heldout_pairs(CHOICES[:1], vintages)
    assert pairs["vintage"].tolist() == ["2026-10-17", "2026-10-24"]
    assert pairs["model"].tolist() == ["randomwalk", "randomwalk"]
    # Missing is None or NaN depending on the pandas string dtype.
    assert pairs["truth_vintage"].iloc[0] == "2026-10-24"
    assert pd.isna(pairs["truth_vintage"].iloc[1])


def test_heldout_needs_flag_and_choice(
    target: pd.DataFrame,
    delays: pd.DataFrame,
    daily: pd.DataFrame,
    config: BacktestConfig,
) -> None:
    with pytest.raises(HeldOutError, match="evaluate_holdout"):
        run_heldout(target, delays, config, daily=daily)
    with pytest.raises(HeldOutError, match="model choice"):
        run_heldout(target, delays, config, evaluate_holdout=True, daily=daily)


@pytest.fixture(scope="module")
def rw_run(
    target: pd.DataFrame,
    delays: pd.DataFrame,
    daily: pd.DataFrame,
    tmp_path_factory: pytest.TempPathFactory,
) -> tuple[BacktestConfig, pd.DataFrame, Path]:
    """One held-out run with randomwalk chosen and a second cheap model."""
    from bdbvhubrenewal.forecasters import RandomWalk, register_model
    from bdbvhubrenewal.forecasters.base import _REGISTRY

    tmp = tmp_path_factory.mktemp("heldout")
    config = BacktestConfig(
        n_samples=100,
        seed=1,
        development_until="2026-10-10",
        model_choice=_randomwalk_choice(tmp),
        simulated_dates=("2026-08-01", "2026-09-05"),
    )
    register_model(
        "shadow",
        lambda: RandomWalk(name="shadow"),
        "A second cheap model.",
        replace=True,
    )
    try:
        scores = run_heldout(
            target,
            delays,
            config,
            evaluate_holdout=True,
            out_dir=tmp / "out",
            models=("shadow",),
            daily=daily,
        )
    finally:
        _REGISTRY.pop("shadow")
    return config, scores, tmp / "out"


def test_heldout_scores_each_forecast_under_its_model(
    rw_run: tuple[BacktestConfig, pd.DataFrame, Path],
) -> None:
    _, scores, out = rw_run
    # The 2026-10-17 and 2026-10-24 forecasts have later data.
    assert set(scores["origin"]) == {"2026-10-17", "2026-10-24"}
    assert set(scores["truth_vintage"]) == {"2026-10-24", "2026-11-01"}
    # Each score records the model chosen at the time.
    assert set(scores["chosen_model"]) == {"randomwalk"}
    # Both baselines and the other variants asked for are scored on every
    # held-out vintage.
    assert set(scores["model"]) == {
        "randomwalk",
        "randomwalknowcast",
        "shadow",
    }
    # The nowcast is scored on the reported target through its own delay.
    reported = scores.loc[scores["target"] == "reported", "model"]
    assert "randomwalknowcast" in set(reported)
    record = yaml.safe_load((out / "heldout-run.yml").read_text())
    assert len(record["model_choice_sha256"]) == 64
    assert record["pending"] == [
        {"vintage": "2026-11-01", "model": "randomwalk"}
    ]
    forecasts = pd.read_csv(out / "heldout-forecasts.csv.gz")
    assert set(forecasts["vintage"]) == {
        "2026-10-17",
        "2026-10-24",
        "2026-11-01",
    }


def test_heldout_keeps_and_rescores_other_models_forecasts(
    target: pd.DataFrame,
    delays: pd.DataFrame,
    daily: pd.DataFrame,
    rw_run: tuple[BacktestConfig, pd.DataFrame, Path],
    tmp_path: Path,
) -> None:
    config, _, out = rw_run
    path = tmp_path / "heldout-forecasts.csv.gz"
    fc = pd.read_csv(out / "heldout-forecasts.csv.gz")
    fc = fc.loc[fc["model"] != "shadow"]
    # A fitted variant's forecasts, as `fit-store load` leaves them.
    other = fc.loc[fc["model"] == "randomwalk"].assign(model="other")
    pd.concat([fc, other]).to_csv(path, index=False)
    scores = run_heldout(
        target,
        delays,
        config,
        evaluate_holdout=True,
        out_dir=tmp_path,
        daily=daily,
    )
    assert set(scores["model"]) == {
        "randomwalk",
        "randomwalknowcast",
        "other",
    }
    assert set(pd.read_csv(path)["model"]) == {
        "randomwalk",
        "randomwalknowcast",
        "other",
    }


def test_heldout_without_fits_reads_committed_forecasts(
    target: pd.DataFrame,
    delays: pd.DataFrame,
    daily: pd.DataFrame,
    config: BacktestConfig,
    rw_run: tuple[BacktestConfig, pd.DataFrame, Path],
    tmp_path: Path,
) -> None:
    from bdbvhubrenewal.forecasters import RandomWalk, register_model
    from bdbvhubrenewal.forecasters.base import _REGISTRY

    class Slow(RandomWalk):
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
        # Stored forecasts of the fitted model, as `fit-store load` writes them.
        seeded = pd.read_csv(rw_run[2] / "heldout-forecasts.csv.gz")
        seeded = seeded.loc[seeded["model"] == "randomwalk"]
        out = tmp_path / "out"
        out.mkdir()
        seeded.assign(model="slowfit", chosen_model="slowfit").to_csv(
            out / "heldout-forecasts.csv.gz", index=False
        )
        # Its stored delay, which reports every case on its onset day: a
        # row per province and the shared one, as the renewal fits write.
        places = set(seeded["location"]) - {"CD"}
        provinces = [*sorted({province_of(p) for p in places}), "shared"]
        pd.DataFrame(
            [
                {
                    "model": "slowfit",
                    "vintage": vintage,
                    "location": loc,
                    "x": x,
                    "quantity": "reported_share",
                    "q50": 1.0,
                }
                for vintage in ("2026-10-17", "2026-10-24")
                for loc in provinces
                for x in (0, 1)
            ]
        ).to_csv(out / DELAY_SUMMARY_FILE, index=False)
        scores = run_heldout(
            target,
            delays,
            config,
            evaluate_holdout=True,
            out_dir=out,
            daily=daily,
        )
    finally:
        _REGISTRY.pop("slowfit")
    assert set(scores["model"]) == {
        "randomwalk",
        "randomwalknowcast",
        "slowfit",
    }
    assert set(scores["chosen_model"]) == {"slowfit"}
    reported = scores.loc[scores["target"] == "reported", "model"]
    assert "slowfit" in set(reported)


def _randomwalk_choice(tmp_path: Path) -> Path:
    path = tmp_path / "randomwalk-choice.yml"
    path.write_text("choices:\n  - {date: 2026-10-10, model: randomwalk}\n")
    return path


def test_heldout_with_no_vintage_after_the_choice(
    target: pd.DataFrame,
    delays: pd.DataFrame,
    daily: pd.DataFrame,
    config: BacktestConfig,
    tmp_path: Path,
) -> None:
    # A choice after the latest vintage holds nothing out yet.
    config.model_choice.write_text(
        "choices:\n  - {date: 2026-12-31, model: randomwalknowcast}\n"
    )
    scores = run_heldout(
        target,
        delays,
        config,
        evaluate_holdout=True,
        out_dir=tmp_path,
        daily=daily,
    )
    assert scores.empty
    record = yaml.safe_load((tmp_path / "heldout-run.yml").read_text())
    assert record["scored"] == []
    assert record["pending"] == []


def test_merge_by_model_reads_an_empty_gzipped_file(tmp_path: Path) -> None:
    path = tmp_path / "heldout-forecasts.csv.gz"
    pd.DataFrame().to_csv(path, index=False)
    new = pd.DataFrame({"model": ["randomwalk"], "value": [1.0]})
    pd.testing.assert_frame_equal(merge_by_model(path, new), new)


def test_merge_by_model_replaces_only_the_models_run(tmp_path: Path) -> None:
    path = tmp_path / "forecasts.csv.gz"
    old = pd.DataFrame({"model": ["randomwalk", "other"], "value": [1.0, 2.0]})
    old.to_csv(path, index=False)
    new = pd.DataFrame({"model": ["randomwalk"], "value": [3.0]})
    merged = merge_by_model(path, new)
    assert sorted(zip(merged["model"], merged["value"], strict=True)) == [
        ("other", 2.0),
        ("randomwalk", 3.0),
    ]
    assert merge_by_model(tmp_path / "missing.csv", new) is new
    # A rerun of every model in the file replaces it.
    assert merge_by_model(path, old) is old
    # A run of no model, as `score_tasks` returns it, keeps every row.
    empty = pd.DataFrame(columns=["truth_vintage"])
    pd.testing.assert_frame_equal(merge_by_model(path, empty), old)
