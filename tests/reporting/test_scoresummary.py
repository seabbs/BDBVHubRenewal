from __future__ import annotations

import json
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from bdbvhubrenewal.evaluation.nowcast import relative_energy, relative_wis
from bdbvhubrenewal.evaluation.scoring import relative_skill
from bdbvhubrenewal.reporting.scoresummary import summary_html, summary_rows

SUMMARISE = Path(__file__).parents[2] / "docs" / "javascripts" / "summarise.js"
MODELS = ["randomwalk", "a", "b"]
COLOURS = dict.fromkeys(MODELS, ("#000000", "#ffffff", ""))


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
                            "horizon": -1 if week == "2026-10-10" else 0,
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


def test_summary_html_names_the_view_and_fallback() -> None:
    html = summary_html(
        {"latest": "x.json"}, "Scores", view="place", static="a.svg"
    )
    assert 'data-view="place"' in html
    assert "a-dark.svg#only-dark" in html


ENERGY_JS = SUMMARISE.with_name("energy.js")


def _energy() -> pd.DataFrame:
    rng = np.random.default_rng(4)
    rows = [
        {
            "model": model,
            "origin": origin,
            "truth_vintage": release,
            "scale": scale,
            "energy": rng.gamma(2, 5),
            "log_energy": rng.gamma(2, 0.1),
        }
        for model in MODELS
        for origin, release in (
            ("2026-10-17", "2026-10-24"),
            ("2026-10-17", "2026-11-01"),
            ("2026-10-24", "2026-11-01"),
        )
        for scale in ("national", "provinces", "zones")
        # randomwalk misses one release, so it is not matched.
        if not (model == "randomwalk" and release == "2026-10-24")
    ]
    return pd.DataFrame(rows)


@pytest.mark.parametrize("pool", ["national", "provinces", "zones"])
def test_browser_energy_matches_relative_energy(pool: str) -> None:
    energy = _energy()
    data = summary_rows(_scores(), MODELS, COLOURS, {}, energy=energy)
    ids = [p["id"] for p in data["pools"]]
    script = (
        f"const s = require({json.dumps(str(ENERGY_JS))});"
        "const d = JSON.parse(require('fs').readFileSync(0, 'utf8'));"
        f"const keep = function (r) {{ return r[3] === {ids.index(pool)}; }};"
        "process.stdout.write(JSON.stringify(s(d, keep)));"
    )
    got = json.loads(
        subprocess.run(
            ["node", "-e", script],
            input=json.dumps(data),
            capture_output=True,
            text=True,
            check=True,
        ).stdout
    )
    want = relative_energy(energy, by=("scale",)).set_index(["scale", "model"])
    assert len(got) == len(MODELS)
    for m, row in got.items():
        exp = want.loc[(pool, MODELS[int(m)])]
        assert row["n"] == exp["n_energy"] == 2
        for col in ("energy", "log_energy"):
            assert row["rel"][col] == pytest.approx(exp[f"rel_{col}"], abs=1e-3)


def test_a_zero_baseline_gives_no_relative_score() -> None:
    scores = _scores()
    base = (scores["model"] == "randomwalk") & (scores["location"] == "CD")
    scores.loc[base, ["wis", "log_wis"]] = 0.0
    data = summary_rows(scores, MODELS, COLOURS, {})
    got = _node(data, "function (r) { return r[3] === 0; }")
    assert data["places"][0]["id"] == "CD"
    assert all(r["rel"]["wis"] is None for r in got)
    rel = relative_wis(scores.loc[scores["location"] == "CD"])
    assert rel["rel_wis"].isna().all()


def test_browser_summary_by_horizon_matches_relative_wis() -> None:
    scores = _scores()
    data = summary_rows(scores, MODELS, COLOURS, {})
    got = _node(data, "function (r, i) { return d.horizons[i] === 0; }")
    want = relative_wis(scores, by=("horizon",)).set_index(["horizon", "model"])
    for row in got:
        exp = want.loc[(0, MODELS[row["model"]])]
        assert row["n"] == exp["n"]
        assert row["rel"]["wis"] == pytest.approx(exp["rel_wis"], abs=1e-3)


def test_browser_relative_scores_are_mean_over_mean_like_relative_skill() -> (
    None
):
    # randomwalk has no forecast for one place and week, so the browser and
    # relative_skill both leave it out and divide mean by mean. The mean of
    # per-forecast ratios would differ.
    scores = _scores()
    data = summary_rows(scores, MODELS, COLOURS, {})
    got = _node(data, "function () { return true; }")
    want = relative_skill(
        scores,
        metrics=("wis", "log_wis"),
        match_on=("origin", "truth_vintage", "location", "target_end_date"),
    ).set_index("model")
    pair = scores.merge(
        scores.loc[scores["model"] == "randomwalk"],
        on=["origin", "truth_vintage", "location", "target_end_date"],
        suffixes=("", "_base"),
    )
    ratios = (pair["wis"] / pair["wis_base"]).groupby(pair["model"]).mean()
    for row in got:
        model = MODELS[row["model"]]
        exp = want.loc[model]
        assert row["n"] == exp["n"]
        assert row["rel"]["wis"] == pytest.approx(exp["relative_wis"], abs=1e-6)
        assert row["rel"]["log_wis"] == pytest.approx(
            exp["relative_log_wis"], abs=1e-6
        )
        if model != "randomwalk":
            assert row["rel"]["wis"] != pytest.approx(ratios[model], abs=1e-3)


def test_summary_html_offers_a_map_only_when_given_one() -> None:
    srcs = {"latest": "x.json"}
    assert 'data-map="m.geojson"' in summary_html(
        srcs, "Scores", view="place", map_src="m.geojson"
    )
    assert "data-map" not in summary_html(srcs, "Scores", view="place")


def test_summary_html_lists_each_target_in_order() -> None:
    html = summary_html({"complete": "c.json", "latest": "l.json"}, "Scores")
    assert (
        """data-targets='{"complete": "c.json", "latest": "l.json"}'""" in html
    )


def _node_with(
    js: Path, data: dict[str, object], keep: str, baseline: int
) -> object:
    script = (
        f"const s = require({json.dumps(str(js))});"
        "const d = JSON.parse(require('fs').readFileSync(0, 'utf8'));"
        f"const keep = {keep};"
        f"process.stdout.write(JSON.stringify(s(d, keep, {baseline})));"
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
def test_browser_summary_takes_any_model_as_the_baseline(pool: str) -> None:
    # "a" has no forecast for one place and week, so the matched set, n
    # and the means move with the baseline, not only the relative scores.
    scores = _scores()
    drop = (scores["model"] == "a") & (scores["location"] == "CD-IT")
    drop &= scores["target_end_date"] == "2026-10-10"
    scores = scores.loc[~drop]
    depth = scores["location"].str.count("-")
    scores["pool"] = depth.map({0: "national", 1: "provinces", 2: "zones"})
    data = summary_rows(scores, MODELS, COLOURS, {})
    places = [i for i, p in enumerate(data["places"]) if p["pool"] == pool]
    keep = f"function (r) {{ return {places}.includes(r[3]); }}"
    got = _node_with(SUMMARISE, data, keep, MODELS.index("a"))
    assert isinstance(got, list)
    want = relative_wis(scores, by=("pool",), baseline="a").set_index(
        ["pool", "model"]
    )
    assert len(got) == len(MODELS)
    for row in got:
        exp = want.loc[(pool, MODELS[row["model"]])]
        assert row["n"] == exp["n"]
        assert row["rel"]["wis"] == pytest.approx(exp["rel_wis"], abs=1e-3)
        assert row["rel"]["log_wis"] == pytest.approx(
            exp["rel_log_wis"], abs=1e-3
        )
        for col in ("wis", "coverage_50", "coverage_90", "bias"):
            assert row["mean"][col] == pytest.approx(exp[col], abs=1e-3)


@pytest.mark.parametrize("pool", ["national", "provinces", "zones"])
def test_browser_energy_takes_any_model_as_the_baseline(pool: str) -> None:
    energy = _energy()
    data = summary_rows(_scores(), MODELS, COLOURS, {}, energy=energy)
    ids = [p["id"] for p in data["pools"]]
    keep = f"function (r) {{ return r[3] === {ids.index(pool)}; }}"
    got = _node_with(ENERGY_JS, data, keep, MODELS.index("b"))
    assert isinstance(got, dict)
    want = relative_energy(energy, by=("scale",), baseline="b").set_index(
        ["scale", "model"]
    )
    assert len(got) == len(MODELS)
    for m, row in got.items():
        exp = want.loc[(pool, MODELS[int(m)])]
        assert row["n"] == exp["n_energy"]
        for col in ("energy", "log_energy"):
            assert row["rel"][col] == pytest.approx(exp[f"rel_{col}"], abs=1e-3)


def test_a_zero_energy_baseline_gives_no_relative_score() -> None:
    energy = _energy()
    energy.loc[energy["model"] == "a", ["energy", "log_energy"]] = 0.0
    data = summary_rows(_scores(), MODELS, COLOURS, {}, energy=energy)
    got = _node_with(ENERGY_JS, data, "function () { return true; }", 1)
    assert isinstance(got, dict)
    assert all(r["rel"]["energy"] is None for r in got.values())
    rel = relative_energy(energy, by=(), baseline="a")
    assert rel["rel_energy"].isna().all()
