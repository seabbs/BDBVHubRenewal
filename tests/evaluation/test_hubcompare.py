from __future__ import annotations

import gzip
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from bdbvhubrenewal.evaluation.backtest import BacktestConfig
from bdbvhubrenewal.evaluation.hubcompare import (
    ATTRIBUTION,
    OPEN_LICENCES,
    forecast_quantiles,
    load_comparison_config,
    load_hub_submissions,
    load_team_metadata,
    match_vintage,
    our_forecasts,
    resolve_baseline,
    score_comparison,
    stored_forecasts,
    submission_dir,
    summarise_comparison,
)
from bdbvhubrenewal.reporting.plots import plot_hub_round

ROOT = Path(__file__).resolve().parents[2]
WEEKS = ["2026-09-19", "2026-09-26", "2026-10-03", "2026-10-10"]


def _hub_rows(
    ref: str = "2026-10-17", weeks: list[str] = WEEKS
) -> pd.DataFrame:
    rng = np.random.default_rng(1)
    return pd.DataFrame(
        [
            {
                "reference_date": ref,
                "target": "wk inc confirmed cases",
                "target_end_date": w,
                "location": loc,
                "output_type": "sample",
                "output_type_id": s,
                "value": float(rng.poisson(400 if loc == "CD" else 100)),
            }
            for loc in ("CD", "CD-IT")
            for w in weeks
            for s in range(1, 101)
        ]
    )


def _write_hub(root: Path) -> None:
    out = root / "model-output"
    (out / "teamA-weekly").mkdir(parents=True)
    _hub_rows().to_csv(
        out / "teamA-weekly" / "2026-10-17-teamA-weekly.csv", index=False
    )
    # A gzipped file, as committed, for a second team.
    (out / "teamB-late").mkdir(parents=True)
    with gzip.open(
        out / "teamB-late" / "2026-10-17-teamB-late.csv.gz", "wt"
    ) as fh:
        _hub_rows(weeks=WEEKS[2:]).to_csv(fh, index=False)
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
    (meta / "teamB-late.yml").write_text(
        'team_name: "Team B"\nmodel_name: "Late"\nlicense: "CC-BY-4.0"\n'
    )


def _ours(target: pd.DataFrame, delays: pd.DataFrame) -> pd.DataFrame:
    config = BacktestConfig(n_samples=100, seed=1)
    return our_forecasts(
        ["2026-10-17"], target, delays, config, ["persistence"], min_horizon=-4
    )


def test_load_hub_submissions_keeps_weekly_samples(tmp_path: Path) -> None:
    _write_hub(tmp_path)
    weekly = load_hub_submissions(tmp_path)
    assert set(weekly["model"]) == {"teamA-weekly", "teamB-late"}
    assert weekly["sample"].max() == 100
    assert set(weekly["source"]) == {"hub"}
    # Reads several roots, as the committed and cached files are apart.
    assert len(load_hub_submissions(tmp_path, tmp_path / "none")) == len(weekly)


def test_load_team_metadata(tmp_path: Path) -> None:
    _write_hub(tmp_path)
    meta = load_team_metadata(tmp_path)
    assert dict(zip(meta["model"], meta["license"], strict=True)) == {
        "teamA-weekly": "CC0-1.0",
        "teamB-late": "CC-BY-4.0",
    }


def test_submission_dir_follows_the_licence(tmp_path: Path) -> None:
    assert set(OPEN_LICENCES) == {"CC0-1.0", "CC-BY-4.0"}
    keep, cache = tmp_path / "keep", tmp_path / "cache"
    assert submission_dir("CC-BY-4.0", keep, cache) == keep
    assert submission_dir("CC0-1.0", keep, cache) == keep
    assert submission_dir("Proprietary", keep, cache) == cache
    assert submission_dir("", keep, cache) == cache


def test_attribution_names_the_hub() -> None:
    assert "BDBV Modeling Hub" in ATTRIBUTION


def test_load_comparison_config() -> None:
    cfg = load_comparison_config(ROOT / "config" / "hub-comparison.toml")
    assert cfg.baseline == "persistencenow"
    assert "CD" in cfg.locations


def test_resolve_baseline() -> None:
    # Falls back to the registry baseline until the preferred one exists.
    assert resolve_baseline("persistence") == "persistence"
    assert resolve_baseline("not-registered") == "persistence"


def test_match_vintage(target: pd.DataFrame) -> None:
    real = target.loc[target["kind"] == "real"]
    vintages = sorted(set(real["vintage"]))
    assert match_vintage("2026-10-31", vintages) == ("2026-10-24", "real")
    assert match_vintage("2026-10-03", vintages) == (
        "2026-09-26",
        "reconstructed",
    )
    assert match_vintage("2026-11-07", vintages) is None


def test_our_forecasts_cover_the_hub_weeks(
    target: pd.DataFrame, delays: pd.DataFrame
) -> None:
    config = BacktestConfig(n_samples=100, seed=1)
    fc = our_forecasts(
        ["2026-10-31"], target, delays, config, ["persistence"], min_horizon=-5
    )
    assert set(fc["model"]) == {"persistence"}
    assert set(fc["source"]) == {"real"}
    assert fc["horizon"].min() == -5
    assert fc["horizon"].max() == 3


def test_stored_forecasts_pick_matching_rounds(tmp_path: Path) -> None:
    stored = pd.DataFrame(
        {
            "model": ["renewal", "renewal", "persistence", "renewal"],
            "kind": ["real", "real", "real", "reconstructed"],
            "vintage": ["2026-10-24", "2026-10-17", "2026-10-24", "2026-10-24"],
            "reference_date": [
                "2026-10-31",
                "2026-10-24",
                "2026-10-31",
                "2026-10-31",
            ],
            "location": "CD",
            "target_end_date": "2026-10-31",
            "horizon": 0,
            "sample": 1,
            "value": 1.0,
        }
    )
    path = tmp_path / "forecasts.csv.gz"
    stored.to_csv(path, index=False)
    out = stored_forecasts(
        [path, tmp_path / "missing.csv.gz"],
        ["2026-10-31"],
        exclude=["persistence"],
    )
    # Real vintages win over reconstructed ones for the same round.
    assert list(out["model"]) == ["renewal"]
    assert list(out["source"]) == ["real"]


def test_score_comparison_flags_complete_weeks(
    tmp_path: Path, target: pd.DataFrame, delays: pd.DataFrame
) -> None:
    _write_hub(tmp_path)
    forecasts = pd.concat(
        [load_hub_submissions(tmp_path), _ours(target, delays)]
    )
    scores, pending = score_comparison(forecasts, target, 3)
    assert pending.empty
    assert set(scores["model"]) == {"teamA-weekly", "teamB-late", "persistence"}
    # Each model is scored on the weeks it covered.
    late = scores.loc[scores["model"] == "teamB-late"]
    assert set(late["target_end_date"]) == set(WEEKS[2:])
    assert set(scores["truth_vintage"]) == {"2026-10-24"}
    # Truth 2026-10-24, K = 3: weeks ending by 2026-10-03 are complete.
    done = scores.loc[scores["complete"], "target_end_date"]
    assert done.max() == "2026-10-03"


def test_score_comparison_waits_for_later_data(target: pd.DataFrame) -> None:
    hub = _hub_rows(ref="2026-10-31").assign(model="teamA-weekly")
    hub = hub.rename(columns={"output_type_id": "sample"}).assign(source="hub")
    scores, pending = score_comparison(hub, target, 3)
    assert scores.empty
    assert list(pending["reference_date"]) == ["2026-10-31"]
    assert pending["vintage"].iloc[0] == "2026-10-24"


def test_summarise_comparison(
    tmp_path: Path, target: pd.DataFrame, delays: pd.DataFrame
) -> None:
    _write_hub(tmp_path)
    forecasts = pd.concat(
        [load_hub_submissions(tmp_path), _ours(target, delays)]
    )
    scores, _ = score_comparison(forecasts, target, 3)
    table = summarise_comparison(scores, "persistence")
    rows = table.set_index("model")
    assert rows.loc["persistence", "relative_wis"] == pytest.approx(1)
    # Complete weeks only: teamB-late forecast 2026-10-03 and 2026-10-10,
    # and only 2026-10-03 is complete, in two locations.
    assert rows.loc["teamB-late", "n"] == 2
    assert {"coverage_50", "coverage_90", "bias", "wis"} <= set(table.columns)


def test_forecast_quantiles(tmp_path: Path) -> None:
    _write_hub(tmp_path)
    q = forecast_quantiles(load_hub_submissions(tmp_path))
    assert {"q05", "q25", "q50", "q75", "q95"} <= set(q.columns)
    assert (q["q05"] <= q["q50"]).all()
    assert (q["q50"] <= q["q95"]).all()
    assert len(q) == q[["model", "location", "target_end_date"]].shape[0]


def test_plot_hub_round(
    tmp_path: Path, target: pd.DataFrame, delays: pd.DataFrame
) -> None:
    _write_hub(tmp_path)
    forecasts = pd.concat(
        [load_hub_submissions(tmp_path), _ours(target, delays)]
    )
    q = forecast_quantiles(forecasts)
    fig = plot_hub_round(q, target, "2026-10-17", locations=("CD", "CD-IT"))
    assert len([ax for ax in fig.axes if ax.get_visible()]) == 2
    labels = [t.get_text() for t in fig.axes[0].get_legend().get_texts()]
    assert "Still being reported" in labels


def test_comparison_label_names_our_hub_models() -> None:
    from bdbvhubrenewal.reporting.plots import comparison_label

    ids = {"persistencenow": "seabbs-baseline"}
    assert (
        comparison_label("persistencenow", "real", ids)
        == "persistencenow (seabbs-baseline)"
    )
    assert comparison_label("persistence", "real", ids) == "persistence (ours)"
    assert comparison_label("OHTJHU-BM", "hub", ids) == "OHTJHU-BM"


def test_own_hub_ids() -> None:
    from bdbvhubrenewal.evaluation.hubcompare import own_hub_ids

    ids = own_hub_ids(ROOT / "config" / "hub.toml")
    assert ids["persistencenow"] == "seabbs-baseline"
