from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from bdbv_hub_renewal.evaluation.backtest import BacktestConfig
from bdbv_hub_renewal.evaluation.hubcompare import (
    ATTRIBUTION,
    display_names,
    load_hub_submissions,
    load_team_metadata,
    match_vintage,
    our_forecasts,
    score_comparison,
)
from bdbv_hub_renewal.reporting.plots import plot_comparison

WEEKS = ["2026-09-19", "2026-09-26", "2026-10-03", "2026-10-10"]


def _write_hub(root: Path) -> None:
    out = root / "model-output"
    rng = np.random.default_rng(1)
    rows = [
        {
            "reference_date": "2026-10-17",
            "target": "wk inc confirmed cases",
            "target_end_date": w,
            "location": loc,
            "output_type": "sample",
            "output_type_id": s,
            "value": float(rng.poisson(400 if loc == "CD" else 100)),
        }
        for loc in ("CD", "CD-IT")
        for w in WEEKS
        for s in range(1, 101)
    ]
    (out / "teamA-weekly").mkdir(parents=True)
    pd.DataFrame(rows).to_csv(
        out / "teamA-weekly" / "2026-10-17-teamA-weekly.csv", index=False
    )
    (out / "epiforecasts-renewal").mkdir(parents=True)
    pd.DataFrame(
        {
            "reference_date": "2026-10-12",
            "target": "cumulative cases",
            "location": "CD",
            "output_type": "quantile",
            "output_type_id": [0.025, 0.5, 0.975],
            "value": [12605, 18575, 32062],
        }
    ).to_csv(
        out / "epiforecasts-renewal" / "2026-10-12-epiforecasts-renewal.csv",
        index=False,
    )
    meta = root / "model-metadata"
    meta.mkdir()
    (meta / "teamA-weekly.yml").write_text(
        'team_name: "Team A"\nmodel_name: "Weekly"\nlicense: "CC0-1.0"\n'
    )
    (meta / "epiforecasts-renewal.yml").write_text(
        'team_name: "epiforecasts"\nmodel_name: "Renewal"\n'
        'license: "CC-BY-4.0"\n'
    )


def test_load_hub_submissions_splits_rounds(tmp_path: Path) -> None:
    _write_hub(tmp_path)
    weekly, other = load_hub_submissions(tmp_path)
    assert set(weekly["model"]) == {"teamA-weekly"}
    assert set(weekly["output_type"]) == {"sample"}
    assert weekly["sample"].max() == 100
    assert list(other["model"]) == ["epiforecasts-renewal"]
    assert other["target"].iloc[0] == "cumulative cases"


def test_load_team_metadata(tmp_path: Path) -> None:
    _write_hub(tmp_path)
    meta = load_team_metadata(tmp_path)
    assert dict(zip(meta["model"], meta["license"], strict=True)) == {
        "epiforecasts-renewal": "CC-BY-4.0",
        "teamA-weekly": "CC0-1.0",
    }


def test_attribution_names_the_hub() -> None:
    assert "BDBV Modeling Hub" in ATTRIBUTION
    assert "INSP" in ATTRIBUTION


def test_match_vintage(target: pd.DataFrame) -> None:
    real = target.loc[target["kind"] == "real"]
    vintages = sorted(set(real["vintage"]))
    # A real vintage serves 2026-10-24 and 2026-10-31.
    assert match_vintage("2026-10-24", vintages) == ("2026-10-17", "real")
    assert match_vintage("2026-10-31", vintages) == ("2026-10-24", "real")
    # None serves 2026-10-03, so it is reconstructed from the Saturday before.
    assert match_vintage("2026-10-03", vintages) == (
        "2026-09-26",
        "reconstructed",
    )
    # Nothing can be built after the latest vintage's round.
    assert match_vintage("2026-11-07", vintages) is None


def test_our_forecasts_use_matching_vintages(
    target: pd.DataFrame, delays: pd.DataFrame
) -> None:
    config = BacktestConfig(n_samples=100, seed=1)
    fc = our_forecasts(
        ["2026-10-03", "2026-10-24"], target, delays, config, ["persistence"]
    )
    by_ref = fc.groupby("reference_date")[["vintage", "source"]].first()
    assert by_ref.loc["2026-10-03"].tolist() == ["2026-09-26", "reconstructed"]
    assert by_ref.loc["2026-10-24"].tolist() == ["2026-10-17", "real"]
    assert set(fc["model"]) == {"epiforecasts-persistence"}


def test_score_comparison_on_common_weeks(
    tmp_path: Path, target: pd.DataFrame, delays: pd.DataFrame
) -> None:
    _write_hub(tmp_path)
    weekly, _ = load_hub_submissions(tmp_path)
    config = BacktestConfig(n_samples=100, seed=1)
    ours = our_forecasts(
        ["2026-10-17"], target, delays, config, ["persistence"]
    )
    scores, pending = score_comparison(weekly, ours, target)
    assert pending.empty
    assert set(scores["model"]) == {"teamA-weekly", "epiforecasts-persistence"}
    # Both models are scored on the same weeks and locations only.
    keys = scores.groupby("model")[["location", "target_end_date"]].apply(
        lambda d: frozenset(map(tuple, d.to_numpy()))
    )
    assert keys.nunique() == 1
    assert set(scores["location"]) == {"CD", "CD-IT"}
    assert set(scores["truth_vintage"]) == {"2026-10-24"}
    assert set(scores["source"]) == {"hub", "real"}


def test_score_comparison_waits_for_later_truth(
    tmp_path: Path, target: pd.DataFrame, delays: pd.DataFrame
) -> None:
    _write_hub(tmp_path)
    weekly, _ = load_hub_submissions(tmp_path)
    weekly["reference_date"] = "2026-10-31"
    config = BacktestConfig(n_samples=100, seed=1)
    ours = our_forecasts(
        ["2026-10-31"], target, delays, config, ["persistence"]
    )
    scores, pending = score_comparison(weekly, ours, target)
    assert scores.empty
    assert list(pending["reference_date"]) == ["2026-10-31"]


@pytest.mark.parametrize(
    ("mode", "expected"),
    [
        (
            "named",
            {
                "teamB-x": "teamB-x",
                "teamA-y": "teamA-y",
                "epiforecasts-persistence": "epiforecasts-persistence",
            },
        ),
        (
            "anonymised",
            {
                "teamA-y": "Team A",
                "teamB-x": "Team B",
                "epiforecasts-persistence": "epiforecasts-persistence",
            },
        ),
        (
            "epiforecasts",
            {"epiforecasts-persistence": "epiforecasts-persistence"},
        ),
    ],
)
def test_display_names(mode: str, expected: dict[str, str]) -> None:
    models = ["teamB-x", "teamA-y", "epiforecasts-persistence"]
    assert display_names(models, mode) == expected


def test_display_names_rejects_unknown_mode() -> None:
    with pytest.raises(ValueError, match="display"):
        display_names(["a-b"], "public")


def test_plot_comparison(
    tmp_path: Path, target: pd.DataFrame, delays: pd.DataFrame
) -> None:
    _write_hub(tmp_path)
    weekly, _ = load_hub_submissions(tmp_path)
    config = BacktestConfig(n_samples=100, seed=1)
    ours = our_forecasts(
        ["2026-10-17"], target, delays, config, ["persistence"]
    )
    forecasts = pd.concat([weekly.assign(source="hub"), ours])
    fig = plot_comparison(
        forecasts,
        target,
        "CD",
        names={"teamA-weekly": "Team A"},
    )
    assert len(fig.axes) == 1


def test_cli_compare_hub(tmp_path: Path) -> None:
    import os

    from bdbv_hub_renewal.cli import main

    _write_hub(tmp_path / "hub")
    root = Path(__file__).resolve().parents[2]
    cfg = tmp_path / "cmp.toml"
    cfg.write_text(
        '[comparison]\nmodels = ["teamA-weekly", "epiforecasts-renewal"]\n'
    )
    old = Path.cwd()
    os.chdir(root)
    try:
        code = main(
            [
                "compare-hub",
                "--config",
                str(cfg),
                "--hub-dir",
                str(tmp_path / "hub"),
                "--out",
                str(tmp_path / "out"),
                "--models",
                "persistence",
            ]
        )
    finally:
        os.chdir(old)
    assert code == 0
    scores = pd.read_csv(tmp_path / "out" / "scores.csv")
    assert set(scores["model"]) == {"teamA-weekly", "epiforecasts-persistence"}
    teams = pd.read_csv(tmp_path / "out" / "teams.csv")
    assert set(teams["license"]) == {"CC0-1.0", "CC-BY-4.0"}
