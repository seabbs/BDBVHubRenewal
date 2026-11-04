from __future__ import annotations

import json
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from bdbvhubrenewal.evaluation.nowcast import relative_wis
from bdbvhubrenewal.reporting.scoresummary import summary_rows

SUMMARISE = Path(__file__).parents[2] / "docs" / "javascripts" / "summarise.js"
MODELS = ["randomwalk", "a", "b"]
COLOURS = dict.fromkeys(MODELS, ("#000000", "#ffffff"))


def _scores() -> pd.DataFrame:
    rng = np.random.default_rng(3)
    rows = []
    for model in MODELS:
        for origin, release in (
            ("2026-10-17", "2026-10-24"),
            ("2026-10-17", "2026-11-01"),
            ("2026-10-24", "2026-11-01"),
        ):
            for loc in ("CD", "CD-IT", "CD-NK", "CD-IT-01"):
                for week in ("2026-10-10", "2026-10-17"):
                    # randomwalk has no forecast for CD-NK in one week, so
                    # those rows of the others are not matched.
                    if model == "randomwalk" and (loc, week) == (
                        "CD-NK",
                        "2026-10-17",
                    ):
                        continue
                    rows.append(
                        {
                            "model": model,
                            "target": "reported",
                            "origin": origin,
                            "truth_vintage": release,
                            "location": loc,
                            "target_end_date": week,
                            "wis": rng.gamma(2, 5),
                            "log_wis": rng.gamma(2, 0.1),
                            "coverage_50": float(rng.random() < 0.5),
                            "coverage_90": float(rng.random() < 0.9),
                            "bias": rng.uniform(-1, 1),
                        }
                    )
    return pd.DataFrame(rows)


def _node(data: dict[str, object], keep: str) -> list[dict[str, object]]:
    script = (
        f"const s = require({json.dumps(str(SUMMARISE))});"
        "const d = JSON.parse(require('fs').readFileSync(0, 'utf8'));"
        f"const keep = {keep};"
        "process.stdout.write(JSON.stringify(s(d, keep)));"
    )
    out = subprocess.run(
        ["node", "-e", script],
        input=json.dumps(data),
        capture_output=True,
        text=True,
        check=True,
    )
    return json.loads(out.stdout)


@pytest.mark.parametrize("pool", ["national", "provinces", "zones"])
def test_browser_summary_matches_relative_wis(pool: str) -> None:
    scores = _scores()
    depth = scores["location"].str.count("-")
    scores["pool"] = depth.map({0: "national", 1: "provinces", 2: "zones"})
    data = summary_rows(scores, MODELS, COLOURS, {})
    places = [i for i, p in enumerate(data["places"]) if p["pool"] == pool]
    got = _node(data, f"function (r) {{ return {places}.includes(r[3]); }}")
    want = relative_wis(scores, by=("pool",)).set_index(["pool", "model"])
    for row in got:
        exp = want.loc[(pool, MODELS[row["model"]])]
        assert row["n"] == exp["n"]
        assert row["rel"]["wis"] == pytest.approx(exp["rel_wis"], abs=1e-3)
        assert row["rel"]["log_wis"] == pytest.approx(
            exp["rel_log_wis"], abs=1e-3
        )
        for col in ("wis", "coverage_50", "coverage_90", "bias"):
            assert row["mean"][col] == pytest.approx(exp[col], abs=1e-3)
    assert len(got) == len(MODELS)


def test_summary_rows_index_every_key() -> None:
    names = {"CD-IT": "Ituri", "CD-NK": "Nord Kivu"}
    data = summary_rows(_scores(), MODELS, COLOURS, names)
    assert data["baseline"] == 0
    assert [p["id"] for p in data["pools"]] == [
        "national",
        "provinces",
        "zones",
    ]
    assert data["places"][1] == {
        "id": "CD-IT",
        "name": "Ituri",
        "pool": "provinces",
    }
    assert len(data["rows"][0]) == 5 + len(data["metrics"])
    json.dumps(data)
