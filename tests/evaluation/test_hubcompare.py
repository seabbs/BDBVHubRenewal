from __future__ import annotations

import gzip
import tomllib
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from bdbvhubrenewal.data import load_daily_zones
from bdbvhubrenewal.data.zones import hub_zones
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
from bdbvhubrenewal.reporting.plots import comparison_colours, plot_hub_round

ROOT = Path(__file__).resolve().parents[2]

pytestmark = pytest.mark.usefixtures("flat_model")
WEEKS = ["2026-09-19", "2026-09-26", "2026-10-03", "2026-10-10"]
DAILY_ZONES = load_daily_zones(ROOT / "data")
ZONES = hub_zones(ROOT / "data" / "hub-config" / "tasks.json")
NO_ZONES = pd.DataFrame(
    {
        **{c: pd.Series(dtype=str) for c in ("kind", "vintage", "location")},
        "target_end_date": pd.Series(dtype=str),
        "total": pd.Series(dtype="int64"),
    }
)


def _hub_rows(
    ref: str = "2026-10-24", weeks: list[str] = WEEKS
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
        out / "teamA-weekly" / "2026-10-24-teamA-weekly.csv", index=False
    )
    # A gzipped file, as committed, for a second team.
    (out / "teamB-late").mkdir(parents=True)
    with gzip.open(
        out / "teamB-late" / "2026-10-24-teamB-late.csv.gz", "wt"
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


_CONFIG = BacktestConfig(n_samples=100, seed=1)


def _ours(
    target: pd.DataFrame, delays: pd.DataFrame, daily: pd.DataFrame
) -> pd.DataFrame:
    config = _CONFIG
    return our_forecasts(
        ["2026-10-24"],
        target,
        delays,
        config,
        ["flat"],
        daily,
        DAILY_ZONES,
        ZONES,
        min_horizon=-4,
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
    assert cfg.baseline == "randomwalk"
    assert cfg.submitted == "data/our-submissions"
    assert "CD" in cfg.locations


def test_resolve_baseline() -> None:
    # Falls back to the registry baseline until the preferred one exists.
    assert resolve_baseline("randomwalk") == "randomwalk"
    assert resolve_baseline("not-registered") == "randomwalk"


def test_match_vintage(target: pd.DataFrame) -> None:
    real = target.loc[target["kind"] == "real"]
    vintages = sorted(set(real["vintage"]))
    assert match_vintage("2026-10-31", vintages) == "2026-10-24"
    assert match_vintage("2026-10-03", vintages) is None
    assert match_vintage("2026-11-07", vintages) == "2026-11-01"
    assert match_vintage("2026-11-14", vintages) is None


def test_our_forecasts_cover_the_hub_weeks(
    target: pd.DataFrame, delays: pd.DataFrame, daily: pd.DataFrame
) -> None:
    config = BacktestConfig(n_samples=100, seed=1)
    fc = our_forecasts(
        ["2026-10-31"],
        target,
        delays,
        config,
        ["flat"],
        daily,
        DAILY_ZONES,
        ZONES,
        min_horizon=-5,
    )
    assert set(fc["model"]) == {"flat"}
    assert set(fc["source"]) == {"real"}
    # The model runs at zone level too.
    assert (fc["location"].str.count("-") == 2).any()
    keys = ["model", "location", "target_end_date", "sample"]
    assert not fc.duplicated(keys).any()
    assert fc["horizon"].min() == -5
    assert fc["horizon"].max() == 3


def test_stored_forecasts_pick_matching_rounds(tmp_path: Path) -> None:
    stored = pd.DataFrame(
        {
            "model": ["renewal", "renewal", "randomwalk", "renewal", "renewal"],
            "kind": ["real", "real", "real", "simulated", "real"],
            "vintage": [
                "2026-10-24",
                "2026-10-17",
                "2026-10-24",
                "2026-10-24",
                "2026-10-24",
            ],
            "reference_date": [
                "2026-10-31",
                "2026-10-24",
                "2026-10-31",
                "2026-10-31",
                "2026-10-31",
            ],
            "as_of": [None, None, None, None, "2026-11-01"],
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
        exclude=["randomwalk"],
    )
    # Only final counts of real vintages are taken.
    assert list(out["model"]) == ["renewal"]
    assert list(out["source"]) == ["real"]


def test_score_comparison_scores_each_later_release(
    tmp_path: Path,
    target: pd.DataFrame,
    delays: pd.DataFrame,
    daily: pd.DataFrame,
) -> None:
    _write_hub(tmp_path)
    forecasts = pd.concat(
        [load_hub_submissions(tmp_path), _ours(target, delays, daily)]
    )
    scores, energy, pending = score_comparison(
        forecasts, target, delays, _CONFIG, daily, NO_ZONES
    )
    assert pending.empty
    assert set(scores["model"]) == {"teamA-weekly", "teamB-late", "flat"}
    # The 2026-10-24 round is made from the 2026-10-17 release and scored
    # on what each later release reported.
    assert set(scores["origin"]) == {"2026-10-17"}
    assert set(scores["reference_date"]) == {"2026-10-24"}
    reported = scores.loc[scores["target"] == "reported"]
    assert set(reported["truth_vintage"]) == {"2026-10-24", "2026-11-01"}
    # Each model is scored on the weeks it covered.
    late = reported.loc[reported["model"] == "teamB-late"]
    assert set(late["target_end_date"]) == set(WEEKS[2:])
    assert set(scores.loc[scores["model"] == "flat", "source"]) == {"real"}
    # The energy score compares every model of a round on shared cells.
    assert set(energy["model"]) == {"teamA-weekly", "teamB-late", "flat"}
    unit = ["reference_date", "target", "truth_vintage", "scale"]
    assert (energy.groupby(unit)["n_cells"].nunique() == 1).all()


def test_zone_truth_fills_missing_zone_weeks_with_zero(
    target: pd.DataFrame,
) -> None:
    from bdbvhubrenewal.evaluation.hubcompare import zone_truth

    zones = pd.DataFrame(
        {
            "vintage": "2026-10-24",
            "location": ["CD-IT-01"],
            "target_end_date": ["2026-10-10"],
            "total": [3],
        }
    )
    out = zone_truth(zones, ("CD-IT-01", "CD-IT-02"), target)
    week = out.loc[out["vintage"] == "2026-10-24"].set_index(
        ["location", "target_end_date"]
    )["total"]
    assert week[("CD-IT-01", "2026-10-10")] == 3
    assert week[("CD-IT-02", "2026-10-10")] == 0
    assert set(out["kind"]) == {"real"}
    assert set(out["vintage"]) == set(target["vintage"])


def test_score_comparison_scores_hub_zones(
    target: pd.DataFrame, delays: pd.DataFrame, daily: pd.DataFrame
) -> None:
    from bdbvhubrenewal.evaluation.hubcompare import zone_truth

    hub = _hub_rows().assign(model="teamA-weekly", source="hub")
    zone = hub.loc[hub["location"] == "CD-IT"].assign(location="CD-IT-01")
    hub = pd.concat([hub, zone]).rename(columns={"output_type_id": "sample"})
    real = target.loc[target["kind"] == "real"]
    counts = real.loc[real["location"] == "CD-IT"].assign(location="CD-IT-01")
    zones = zone_truth(counts, ("CD-IT-01",), target)
    scores, _, _ = score_comparison(hub, target, delays, _CONFIG, daily, zones)
    assert "CD-IT-01" in set(scores["location"])
    assert {"reported", "complete"} >= set(
        scores.loc[scores["location"] == "CD-IT-01", "target"]
    )


def test_score_comparison_waits_for_later_data(
    target: pd.DataFrame, delays: pd.DataFrame, daily: pd.DataFrame
) -> None:
    hub = _hub_rows(ref="2026-11-07").assign(model="teamA-weekly")
    hub = hub.rename(columns={"output_type_id": "sample"}).assign(source="hub")
    scores, _, pending = score_comparison(
        hub, target, delays, _CONFIG, daily, NO_ZONES
    )
    assert scores.empty
    assert list(pending["reference_date"]) == ["2026-11-07"]
    assert pending["vintage"].iloc[0] == "2026-11-01"


def test_summarise_comparison(
    tmp_path: Path,
    target: pd.DataFrame,
    delays: pd.DataFrame,
    daily: pd.DataFrame,
) -> None:
    _write_hub(tmp_path)
    forecasts = pd.concat(
        [load_hub_submissions(tmp_path), _ours(target, delays, daily)]
    )
    scores, _, _ = score_comparison(
        forecasts, target, delays, _CONFIG, daily, NO_ZONES
    )
    table = summarise_comparison(scores, "flat")
    rows = table.loc[table["target"] == "reported"].set_index("model")
    assert rows.loc["flat", "relative_wis"] == pytest.approx(1)
    # teamB-late forecast 2026-10-03 and 2026-10-10 in two locations,
    # each scored against two later releases.
    assert rows.loc["teamB-late", "n"] == 8
    assert {"coverage_50", "coverage_90", "bias", "wis"} <= set(table.columns)


def test_trajectory_samples_keep_open_teams_and_first_draws() -> None:
    from bdbvhubrenewal.evaluation.hubcompare import trajectory_samples

    rows = [
        {
            "model": m,
            "source": src,
            "reference_date": "2026-10-24",
            "location": "CD",
            "target_end_date": "2026-10-10",
            "sample": i,
            "value": 1.0,
        }
        for m, src in (("ours", "real"), ("open", "hub"), ("closed", "hub"))
        for i in (3, 1, 2)
    ]
    teams = pd.DataFrame(
        {"model": ["open", "closed"], "license": ["CC-BY-4.0", "other"]}
    )
    out = trajectory_samples(pd.DataFrame(rows), teams, 2)
    assert set(out["model"]) == {"ours", "open"}
    assert set(out["sample"]) == {1, 2}


def test_forecast_quantiles(tmp_path: Path) -> None:
    _write_hub(tmp_path)
    q = forecast_quantiles(load_hub_submissions(tmp_path))
    assert {"q05", "q25", "q50", "q75", "q95"} <= set(q.columns)
    assert (q["q05"] <= q["q50"]).all()
    assert (q["q50"] <= q["q95"]).all()
    assert len(q) == q[["model", "location", "target_end_date"]].shape[0]


def test_plot_hub_round(
    tmp_path: Path,
    target: pd.DataFrame,
    delays: pd.DataFrame,
    daily: pd.DataFrame,
) -> None:
    _write_hub(tmp_path)
    forecasts = pd.concat(
        [load_hub_submissions(tmp_path), _ours(target, delays, daily)]
    )
    q = forecast_quantiles(forecasts)
    colours = comparison_colours(
        {"flat (main build)": "randomwalk"}, [], ["teamA-weekly", "teamB-late"]
    )
    fig = plot_hub_round(
        q, target, "2026-10-24", colours, locations=("CD", "CD-IT")
    )
    # Ours above the hub teams, one column per location.
    assert len(fig.axes) == 4
    labels = [t.get_text() for t in fig.legends[0].get_texts()]
    assert "Still being reported" in labels
    assert {"teamA-weekly", "teamB-late"} <= set(labels)
    hub_ax = fig.axes[2]
    assert hub_ax.get_title().endswith("Hub teams")
    lines = {ln.get_label(): ln for ln in hub_ax.get_lines()}
    # A hub team keeps its style in a round it alone forecast.
    later = q.loc[q["model"] == "teamB-late"].assign(
        reference_date="2026-10-31"
    )
    both = pd.concat([q, later])
    again = plot_hub_round(
        both, target, "2026-10-31", colours, locations=("CD", "CD-IT")
    )
    again_lines = {ln.get_label(): ln for ln in again.axes[2].get_lines()}
    assert (
        again_lines["teamB-late"].get_color() == lines["teamB-late"].get_color()
    )


def test_with_submissions_keeps_main_builds_beside_submissions() -> None:
    from bdbvhubrenewal.evaluation.hubcompare import with_submissions

    def rows(model: str, source: str, ref: str, value: float) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "model": model,
                "source": source,
                "reference_date": ref,
                "location": "CD",
                "target_end_date": "2026-11-07",
                "sample": [1, 2],
                "value": value,
            }
        )

    ids = {"randomwalknowcast": "seabbs-rwnowcast"}
    forecasts = pd.concat(
        [
            rows("randomwalknowcast", "real", "2026-10-31", 1.0),
            rows("randomwalknowcast", "real", "2026-11-07", 2.0),
            rows("seabbs-rwnowcast", "hub", "2026-11-07", 3.0),
            rows("teamA", "hub", "2026-11-07", 4.0),
        ]
    )
    submitted = rows("seabbs-rwnowcast", "hub", "2026-11-07", 5.0)
    out = with_submissions(forecasts, submitted, ids)
    got = out.groupby(["model", "source", "reference_date"])["value"].first()
    # The hub's copy of our file goes, and the main build stays.
    assert got.to_dict() == {
        ("randomwalknowcast", "real", "2026-10-31"): 1.0,
        ("randomwalknowcast", "real", "2026-11-07"): 2.0,
        ("seabbs-rwnowcast", "submitted", "2026-11-07"): 5.0,
        ("teamA", "hub", "2026-11-07"): 4.0,
    }


def test_comparison_label_names_our_hub_models() -> None:
    from bdbvhubrenewal.reporting.plots import comparison_label

    assert (
        comparison_label("seabbs-rwnowcast", "submitted")
        == "seabbs-rwnowcast (submitted)"
    )
    assert comparison_label("randomwalk", "real") == "randomwalk (main build)"
    assert comparison_label("OHTJHU-BM", "hub") == "OHTJHU-BM"


def test_own_hub_ids() -> None:
    from bdbvhubrenewal.evaluation.hubcompare import own_hub_ids

    ids = own_hub_ids(ROOT / "config" / "hub.toml")
    assert ids["randomwalknowcast"] == "seabbs-rwnowcast"
    # The renewal track's id carries the model the chain submits first.
    track = tomllib.loads((ROOT / "config" / "hub.toml").read_text())
    assert ids[track["submission"]["model"]] == "seabbs-renewal"
    # Only models set to submit have a hub id.
    off = [m for m, t in track.get("models", {}).items() if not t["submit"]]
    assert not set(off) & set(ids)


def test_comparison_colours_are_fixed_per_model() -> None:
    from bdbvhubrenewal.reporting.plots import (
        HUB_GREYS,
        MODEL_COLOURS,
        MODEL_STYLE,
        comparison_colours,
        model_style,
    )

    teams = ["teamA", "teamB"]
    got = comparison_colours(
        {"randomwalk (main build)": "randomwalk"},
        ["seabbs-x (submitted)"],
        teams,
    )
    first = (MODEL_COLOURS["light"][0], MODEL_COLOURS["dark"][0], "")
    assert got["randomwalk (main build)"] == first
    # A submitted series has a colour no model uses.
    assert got["seabbs-x (submitted)"][:2] not in {
        model_style(m, [])[:2] for m in MODEL_STYLE
    }
    assert {got[t][0] for t in teams} <= set(HUB_GREYS["light"])
    assert {got[t][1] for t in teams} <= set(HUB_GREYS["dark"])
    # A team keeps its grey whichever of our models are shown.
    assert comparison_colours({}, [], teams)["teamB"] == got["teamB"]
