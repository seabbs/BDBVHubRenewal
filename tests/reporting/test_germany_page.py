"""The German nowcast benchmark page."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from bdbvhubrenewal.evaluation.germany import load_germany_config
from bdbvhubrenewal.evaluation.scoring import SCORE_COLUMNS
from bdbvhubrenewal.reporting.germany import germany_page

ROOT = Path(__file__).resolve().parents[2]
CONFIG = load_germany_config(ROOT / "config" / "germany.toml")


def _scores(path: Path) -> None:
    rows = []
    for model, wis in (("randomwalk", 10.0), ("randomwalknowcast", 5.0)):
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
                        "n_samples": 100,
                        **dict.fromkeys(SCORE_COLUMNS, 0.5),
                        "wis": wis,
                        "log_wis": wis / 10,
                        "pool": pool,
                    }
                )
    path.mkdir(parents=True)
    pd.DataFrame(rows).to_csv(path / "scores.csv.gz", index=False)


def test_page_tables_and_summaries(tmp_path: Path) -> None:
    res = tmp_path / "germany"
    _scores(res)
    fig, gen = tmp_path / "figures", tmp_path / "generated"
    germany_page(fig, gen, res, CONFIG)
    table = (gen / "germany-scores.md").read_text()
    assert "randomwalknowcast" in table
    data = json.loads((fig / "scoresummary-germany-latest.json").read_text())
    assert [p["id"] for p in data["pools"]] == ["national", "states", "age"]
    names = {p["id"]: p["name"] for p in data["places"]}
    assert names["DE-BY"] == "Bavaria"
    assert names["80+"] == "Age 80+"
    assert data["models"][data["baseline"]]["name"] == "randomwalk"
    assert CONFIG.commit[:7] in (gen / "germany-sources.md").read_text()


def test_page_says_when_not_run(tmp_path: Path) -> None:
    fig, gen = tmp_path / "figures", tmp_path / "generated"
    germany_page(fig, gen, tmp_path / "missing", CONFIG)
    assert "Not run yet" in (gen / "germany-scores.md").read_text()
