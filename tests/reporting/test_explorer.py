from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from bdbvhubrenewal.reporting import explorer

COLOURS = {
    "a (ours)": ("#111111", "#eeeeee", "6 3"),
    "hub": ("#222222", "#dddddd", ""),
}


def _inputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rows = []
    for model, ours in (("a (ours)", True), ("hub", False)):
        for loc in ("CD", "CD-IT", "CD-IT-01"):
            rows.append(
                {
                    "model": model,
                    "ours": ours,
                    "reference_date": "2026-10-31",
                    "location": loc,
                    "target_end_date": "2026-10-31",
                    "q05": 1.0,
                    "q25": 2.0,
                    "q50": 3.0,
                    "q75": 4.0,
                    "q95": 5.0,
                }
            )
    quantiles = pd.DataFrame(rows)
    target = pd.DataFrame(
        {
            "kind": "real",
            "vintage": "2026-10-24",
            "location": ["CD", "CD-IT"],
            "target_end_date": "2026-10-17",
            "total": [7, 4],
        }
    )
    return quantiles, target


def test_explorer_data_levels_and_rows() -> None:
    quantiles, target = _inputs()
    data = explorer_data = explorer.explorer_data(
        quantiles, target, {"CD": "DRC"}, 3, COLOURS
    )
    levels = {loc["id"]: loc["level"] for loc in data["locations"]}  # type: ignore[attr-defined]
    assert levels == {"CD": "National", "CD-IT": "Province", "CD-IT-01": "Zone"}
    assert data["complete_through"] == "2026-10-03"
    assert len(data["q"]) == 6  # type: ignore[arg-type]
    assert data["data"]["CD"] == [["2026-10-17", 7]]  # type: ignore[index]
    assert explorer_data is data


def test_write_explorer_makes_the_folder(tmp_path: Path) -> None:
    quantiles, target = _inputs()
    data = explorer.explorer_data(quantiles, target, {}, 3, COLOURS)
    path = tmp_path / "sub" / "x.json"
    explorer.write_explorer(path, data)
    assert json.loads(path.read_text())["rounds"] == ["2026-10-31"]


def test_explorer_html_names_the_data() -> None:
    html = explorer.explorer_html("../figures/x.json", "Forecasts")
    assert 'data-src="../figures/x.json"' in html
    assert 'class="explorer"' in html


def test_explorer_hides_models_until_switched_on() -> None:
    quantiles, target = _inputs()
    data = explorer.explorer_data(
        quantiles, target, {}, 3, COLOURS, hidden=["a (ours)"]
    )
    assert data["models"] == [
        {
            "name": "a (ours)",
            "ours": True,
            "on": False,
            "light": "#111111",
            "dark": "#eeeeee",
            "dash": "6 3",
            "trajectories": False,
        },
        {
            "name": "hub",
            "ours": False,
            "on": True,
            "light": "#222222",
            "dark": "#dddddd",
            "dash": "",
            "trajectories": False,
        },
    ]


def test_explorer_html_can_start_with_all_rounds() -> None:
    assert 'data-round="all"' in explorer.explorer_html(
        "x.json", "Forecasts", all_rounds=True
    )
    assert "data-round" not in explorer.explorer_html("x.json", "Forecasts")


def test_explorer_html_names_the_summary_it_tabulates() -> None:
    html = explorer.explorer_html("x.json", "Forecasts", summary="s.json")
    assert 'data-summary="s.json"' in html


def test_trajectories_keep_the_same_draws_in_every_place() -> None:
    samples = pd.DataFrame(
        [
            {
                "model": "a (ours)",
                "reference_date": "2026-10-31",
                "location": loc,
                "target_end_date": week,
                "sample": k,
                "value": float(k),
            }
            for loc in ("CD", "CD-IT")
            for week in ("2026-10-24", "2026-10-31")
            for k in (7, 3, 9)
        ]
    )
    t = explorer.trajectories(samples, n=2)
    assert set(t["draw"]) == {0, 1}
    # Draw 0 is sample 3 everywhere, so it is joint across places.
    first = t.loc[t["draw"] == 0]
    assert [list(v) for v in first["values"]] == [[3.0, 3.0], [3.0, 3.0]]
    assert set(first["start"]) == {"2026-10-24"}


def test_explorer_marks_models_with_trajectories() -> None:
    quantiles, target = _inputs()
    samples = pd.DataFrame(
        {
            "model": "a (ours)",
            "reference_date": "2026-10-31",
            "location": "CD",
            "target_end_date": ["2026-10-24", "2026-10-31"],
            "sample": 1,
            "value": [5.0, 6.0],
        }
    )
    data = explorer.explorer_data(
        quantiles, target, {}, 3, COLOURS, samples=samples
    )
    assert [m["trajectories"] for m in data["models"]] == [True, False]  # type: ignore[attr-defined]
    assert data["traj"] == [[0, 0, "CD", 0, "2026-10-24", 5.0, 6.0]]
