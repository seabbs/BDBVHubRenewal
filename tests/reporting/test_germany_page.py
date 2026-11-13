"""The German nowcast benchmark page."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from bdbvhubrenewal.evaluation.germany import (
    SCORE_COLUMNS,
    load_germany_config,
)
from bdbvhubrenewal.reporting.germany import germany_page

ROOT = Path(__file__).resolve().parents[2]
CONFIG = load_germany_config(ROOT / "config" / "germany.toml")


def _rows(models: tuple[tuple[str, float], ...]) -> pd.DataFrame:
    rows = []
    for model, wis in models:
        for place, pool in (
            ("DE", "national"),
            ("DE-BY", "states"),
            ("80+", "age"),
        ):
            for h in (0, -7):
                end = pd.Timestamp("2021-12-01") + pd.Timedelta(days=h)
                rows.append(
                    {
                        "model": model,
                        "target": "40-day",
                        "origin": "2021-12-01",
                        "truth_vintage": str(end + pd.Timedelta(days=40))[:10],
                        "location": place,
                        "target_end_date": str(end)[:10],
                        "horizon": h,
                        "truth": 100.0,
                        **dict.fromkeys(SCORE_COLUMNS, 0.5),
                        "wis": wis,
                        "pool": pool,
                    }
                )
    return pd.DataFrame(rows)


LEVELS = (0.025, 0.1, 0.25, 0.5, 0.75, 0.9, 0.975)


def _forecasts(models: tuple[str, ...]) -> pd.DataFrame:
    rows = []
    for model in models:
        for origin in ("2021-11-22", "2021-11-23", "2022-02-14"):
            for place in ("DE", "DE-BY", "80+"):
                for h in (0, -7):
                    end = pd.Timestamp(origin) + pd.Timedelta(days=h)
                    for i, q in enumerate(LEVELS):
                        rows.append(
                            {
                                "model": model,
                                "origin": origin,
                                "location": place,
                                "target_end_date": str(end)[:10],
                                "horizon": h,
                                "quantile": q,
                                "value": 10.0 * (i + 1),
                                "truth": 42.0,
                                "truth_vintage": "2022-03-01",
                            }
                        )
    return pd.DataFrame(rows)


def _scores(path: Path) -> None:
    path.mkdir(parents=True)
    ours = ("randomwalk", "randomwalknowcast")
    _forecasts(ours).to_csv(path / "forecasts.csv.gz", index=False)
    teams = ("KIT", "KIT-frozen_baseline")
    _forecasts(teams).to_csv(path / "teams-forecasts.csv.gz", index=False)
    pd.DataFrame(
        {
            "origin": "2021-11-22",
            "location": "DE",
            "target_end_date": ["2021-11-21", "2021-11-22"],
            "total": [30, 20],
        }
    ).to_csv(path / "asof.csv.gz", index=False)
    ours = _rows((("randomwalk", 10.0), ("randomwalknowcast", 5.0)))
    held = ours.assign(origin="2022-03-01")
    pd.concat([ours, held]).to_csv(path / "scores.csv.gz", index=False)
    _rows((("KIT", 4.0),)).to_csv(path / "teams.csv.gz", index=False)


def test_page_tables_and_summaries(tmp_path: Path) -> None:
    res = tmp_path / "germany"
    _scores(res)
    fig, gen = tmp_path / "figures", tmp_path / "generated"
    germany_page(fig, gen, res, CONFIG)
    table = (gen / "germany-scores.md").read_text()
    assert CONFIG.split_date in table
    assert "randomwalknowcast" in table
    assert "| KIT " in table
    assert "rel_wis" in table
    assert "| wis " not in table
    assert "0.8" in table
    data = json.loads((fig / "scoresummary-germany-latest.json").read_text())
    assert [p["id"] for p in data["pools"]] == ["national", "states", "age"]
    names = {p["id"]: p["name"] for p in data["places"]}
    assert names["DE-BY"] == "Bavaria"
    assert names["80+"] == "Age 80+"
    assert data["models"][data["baseline"]]["name"] == "randomwalknowcast"
    assert data["origins"] == ["2021-12-01"]
    assert [m["name"] for m in data["models"]][-1] == "KIT"
    keys = [m["key"] for m in data["metrics"]]
    assert keys == ["wis", "ae_median", "coverage_50", "coverage_95"]
    assert CONFIG.commit[:7] in (gen / "germany-sources.md").read_text()


def test_hidden_models_start_switched_off(tmp_path: Path) -> None:
    res = tmp_path / "germany"
    _scores(res)
    fig, gen = tmp_path / "figures", tmp_path / "generated"
    germany_page(fig, gen, res, CONFIG)
    assert CONFIG.hidden == ("randomwalk", "KIT-frozen_baseline")
    data = json.loads((fig / "scoresummary-germany-latest.json").read_text())
    on = {m["name"]: m["on"] for m in data["models"]}
    assert not on["randomwalk"]
    assert on["randomwalknowcast"]
    assert on["KIT"]


def test_nowcast_explorer_shows_development_plot_dates(tmp_path: Path) -> None:
    res = tmp_path / "germany"
    _scores(res)
    fig, gen = tmp_path / "figures", tmp_path / "generated"
    germany_page(fig, gen, res, CONFIG)
    assert "explorer-germany.json" in (gen / "germany-nowcasts.md").read_text()
    data = json.loads((fig / "explorer-germany.json").read_text())
    # Every 7th development date from the first; later dates are held out.
    assert data["rounds"] == ["2021-11-22"]
    on = {m["name"]: m["on"] for m in data["models"]}
    assert on == {
        "randomwalk": False,
        "randomwalknowcast": True,
        "KIT": True,
        "KIT-frozen_baseline": False,
    }
    assert data["levels"] == ["National", "States", "Age groups"]
    assert data["outer"] == "95%"
    # q05 and q95 carry the 2.5% and 97.5% quantiles.
    assert {(r[4], r[8]) for r in data["q"]} == {(10.0, 70.0)}
    assert data["data"]["DE"][0][1] == 42
    assert data["asof"] == [
        [0, "DE", "2021-11-21", 30],
        [0, "DE", "2021-11-22", 20],
    ]


def test_page_says_when_not_run(tmp_path: Path) -> None:
    fig, gen = tmp_path / "figures", tmp_path / "generated"
    germany_page(fig, gen, tmp_path / "missing", CONFIG)
    assert "Not run yet" in (gen / "germany-scores.md").read_text()
