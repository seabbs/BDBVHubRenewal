from __future__ import annotations

import pandas as pd

from bdbvhubrenewal.evaluation.nowcast import BASELINE
from bdbvhubrenewal.reporting import explorer
from bdbvhubrenewal.reporting.legend import starts_on
from bdbvhubrenewal.reporting.scorechart import score_chart_data
from bdbvhubrenewal.reporting.scoresummary import summary_rows
from bdbvhubrenewal.reporting.zone_nowcast import place_chart_data

MODELS = [BASELINE, "renewal"]
COLOURS = dict.fromkeys(MODELS, ("#000000", "#ffffff"))


def _on(data: dict[str, object]) -> dict[str, bool]:
    return {m["name"]: m["on"] for m in data["models"]}  # type: ignore[attr-defined]


def test_as_reported_starts_off_even_when_asked_on() -> None:
    assert not starts_on(BASELINE, MODELS)
    assert starts_on("renewal", MODELS)
    assert not starts_on("renewal", [])


def test_every_interactive_starts_with_as_reported_off() -> None:
    week = {
        "origin": "2026-10-17",
        "truth_vintage": "2026-10-24",
        "location": "CD",
        "target_end_date": "2026-10-17",
        "horizon": 0,
    }
    scores = pd.DataFrame([{"model": m, **week, "wis": 1.0} for m in MODELS])
    summary = summary_rows(
        scores,
        MODELS,
        COLOURS,
        {},
        baseline=BASELINE,
        metrics=[("wis", "WIS", None, True)],
    )
    chart = score_chart_data(
        scores.assign(n=1),
        "horizon",
        {"wis": ("WIS", None)},
        MODELS,
        colours=COLOURS,
    )
    quantiles = pd.DataFrame(
        [
            {
                "model": m,
                "ours": True,
                "reference_date": "2026-10-17",
                "location": "CD",
                "target_end_date": "2026-10-17",
                **dict.fromkeys(("q05", "q25", "q50", "q75", "q95"), 1.0),
            }
            for m in MODELS
        ]
    )
    target = pd.DataFrame(
        {
            "kind": ["real"],
            "vintage": ["2026-10-24"],
            "location": ["CD"],
            "target_end_date": ["2026-10-17"],
            "total": [1],
        }
    )
    explore = explorer.explorer_data(quantiles, target, {}, 3, COLOURS)
    samples = pd.DataFrame(
        [
            {
                "model": m,
                "vintage": "2026-10-24",
                "reference_date": "2026-10-24",
                "location": "CD",
                "target_end_date": "2026-10-17",
                "sample": k,
                "value": float(k),
            }
            for m in MODELS
            for k in range(3)
        ]
    )
    counts = target.drop(columns=["kind", "vintage"])
    place = place_chart_data(samples, counts, {}, COLOURS)
    for data in (summary, chart, explore, place):
        assert _on(data) == {BASELINE: False, "renewal": True}
