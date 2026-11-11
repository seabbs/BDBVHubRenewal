"""The German nowcast benchmark page."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from bdbvhubrenewal.reporting.germany import germany_page


def _scores(path: Path) -> None:
    rows = []
    for model, source, wis in (
        ("randomwalk", "ours", 10.0),
        ("randomwalknowcast", "ours", 5.0),
        ("KIT", "hub", 8.0),
    ):
        for loc, age, level in (
            ("DE", "00+", "national"),
            ("DE-BY", "00+", "states"),
            ("DE", "80+", "age"),
        ):
            for h in (0, -7):
                rows.append(
                    {
                        "model": model,
                        "level": level,
                        "location": loc,
                        "age_group": age,
                        "forecast_date": "2021-12-01",
                        "target_end_date": str(
                            pd.Timestamp("2021-12-01") + pd.Timedelta(days=h)
                        )[:10],
                        "horizon": h,
                        "truth": 100.0,
                        "wis": wis,
                        "coverage_50": 1.0,
                        "coverage_95": 1.0,
                        "source": source,
                    }
                )
    path.mkdir(parents=True)
    pd.DataFrame(rows).to_csv(path / "scores.csv.gz", index=False)
    (path / "sources.json").write_text(
        json.dumps(
            {
                "files": [
                    {
                        "repo": "KITmetricslab/hospitalization-nowcast-hub",
                        "commit": "abc",
                        "path": "data-truth/x.csv",
                    }
                ],
                "accessed": "2026-11-11",
            }
        )
    )


def test_page_tables_and_summaries(tmp_path: Path) -> None:
    res = tmp_path / "germany"
    _scores(res)
    fig, gen = tmp_path / "figures", tmp_path / "generated"
    germany_page(fig, gen, res)
    table = (gen / "germany-scores.md").read_text()
    assert "randomwalknowcast" in table
    assert "KIT" in table
    data = json.loads((fig / "scoresummary-germany-latest.json").read_text())
    assert [p["id"] for p in data["pools"]] == ["national", "states", "age"]
    names = {p["id"]: p["name"] for p in data["places"]}
    assert names["DE-BY"] == "Bavaria"
    assert names["80+"] == "Age 80+"
    assert data["models"][data["baseline"]]["name"] == "randomwalk"
    assert "abc" in (gen / "germany-sources.md").read_text()


def test_page_says_when_not_run(tmp_path: Path) -> None:
    fig, gen = tmp_path / "figures", tmp_path / "generated"
    germany_page(fig, gen, tmp_path / "missing")
    assert "Not run yet" in (gen / "germany-scores.md").read_text()
