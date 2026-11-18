"""Nowcast panels for any dataset: quantile or sample forecasts."""

import csv
import gzip

import numpy as np
import pandas as pd
import pytest
from flowlib.dashboard import nowcasts, runs

from bdbvhubrenewal.evaluation.nowcast import reporting_delay, rolling_truth
from bdbvhubrenewal.evaluation.scoring import crps_sample

SEVEN = ("0.025", "0.1", "0.25", "0.5", "0.75", "0.9", "0.975")
PAGE = (0.025, 0.25, 0.5, 0.75, 0.975)


def _quantiles(
    path,
    model,
    truth=10.0,
    base=8.0,
    target=None,
    score=None,
    location="DE",
    week="2022-02-01",
):
    path.parent.mkdir(parents=True, exist_ok=True)
    cols = [
        "model",
        "location",
        "target_end_date",
        "quantile",
        "value",
        "truth",
        "truth_vintage",
    ]
    cols += ["target"] if target else []
    cols += ["score"] if score is not None else []
    with gzip.open(path, "wt") as f:
        w = csv.DictWriter(f, cols)
        w.writeheader()
        for i, q in enumerate(SEVEN):
            row = {
                "model": model,
                "location": location,
                "target_end_date": week,
                "quantile": q,
                "value": base + i,
                "truth": truth,
                "truth_vintage": "2022-03-01",
            }
            if target:
                row["target"] = target
            if score is not None:
                row["score"] = score
            w.writerow(row)


def _setup():
    rs = runs.load_runs([], "s=slots-,hue=217", [])
    refs = runs.load_references(
        [
            {
                "key": "zero",
                "label": "zero-shot ragged",
                "colour": "#0a0",
                "model": "^(?!randomwalknowcast$)(?!.*_complete$)(?!KIT)",
                "checkpoint": "zero-shot",
                "role": "start",
            },
            {
                "key": "rw",
                "label": "randomwalknowcast",
                "colour": "#666",
                "model": "^randomwalknowcast$",
                "role": "baseline",
            },
        ],
        [],
    )
    return rs, refs


def test_quantile_series_by_target_with_old_files_as_7day(tmp_path):
    _quantiles(
        tmp_path / "reference" / "2022-02-09.csv.gz", "fm_chronos_daily_ragged"
    )
    _quantiles(
        tmp_path / "slots-aaaaaa" / "2022-02-09.csv.gz",
        "fm_chronos_daily_ragged_ft",
        target="daily",
    )
    rs, refs = _setup()
    built = nowcasts.build(tmp_path, rs, refs, levels=PAGE)
    assert built["targets"] == ["7day", "daily"]
    one = built["series"]["DE"]["aaaaaa"]["daily"]["daily"]["2022-02-09"]
    assert one["q"] == [[8.0, 10.0, 11.0, 12.0, 14.0]]
    assert "7day" in built["series"]["DE"]["zero"]


def test_stats_and_score_from_the_seven_levels(tmp_path):
    # Truth 10.5: inside 50% (10..12)? no; inside 95% (8..14) yes.
    _quantiles(
        tmp_path / "reference" / "2022-02-09.csv.gz",
        "fm_chronos_daily_ragged",
        truth=9.5,
    )
    rs, refs = _setup()
    built = nowcasts.build(tmp_path, rs, refs, levels=PAGE)
    cov50, cov95, bias = built["stats"]["DE"]["zero"]["7day"]["daily"][
        "2022-02-09"
    ]
    assert (cov50, cov95) == (0.0, 1.0)
    # CDF at 9.5 between 0.1 (9) and 0.25 (10): 0.175; bias 1 - 0.35.
    assert bias == pytest.approx(0.65)
    assert built["score"]["DE"]["zero"]["7day"]["daily"]["2022-02-09"] == (
        pytest.approx(nowcasts.wis([8, 9, 10, 11, 12, 13, 14], 9.5), abs=0.01)
    )


def test_a_file_score_wins_over_wis(tmp_path):
    _quantiles(
        tmp_path / "reference" / "2022-02-09.csv.gz",
        "fm_chronos_daily_ragged",
        score=3.25,
    )
    rs, refs = _setup()
    built = nowcasts.build(tmp_path, rs, refs, levels=PAGE)
    assert built["score"]["DE"]["zero"]["7day"]["daily"]["2022-02-09"] == 3.25


def test_other_reference_models_are_shown_under_their_name(tmp_path):
    _quantiles(tmp_path / "reference" / "2022-02-09.csv.gz", "KIT-nowcast")
    rs, refs = _setup()
    built = nowcasts.build(tmp_path, rs, refs, levels=PAGE)
    assert "model:KIT-nowcast" in built["sets"]
    meta = {m["id"]: m for m in nowcasts.meta(built, rs, refs, {})}
    assert meta["model:KIT-nowcast"]["label"] == "KIT-nowcast"
    assert meta["model:KIT-nowcast"]["run"] == "ref"


def test_excluded_and_unclaimed_checkpoints_are_left_out(tmp_path):
    _quantiles(
        tmp_path / "reference" / "2022-02-09.csv.gz", "fm_chronos_daily_ragged"
    )
    _quantiles(
        tmp_path / "slots-aaaaaa" / "2022-02-09.csv.gz",
        "fm_chronos_daily_ragged_ft",
    )
    _quantiles(
        tmp_path / "ffffff" / "2022-02-09.csv.gz", "fm_chronos_daily_ragged_ft"
    )
    rs, refs = _setup()
    built = nowcasts.build(tmp_path, rs, refs, levels=PAGE, excluded={"aaaaaa"})
    assert set(built["sets"]) == {"zero"}


def test_checkpoints_of_a_run_are_shaded_by_size(tmp_path):
    for h in ("aaaaaa", "bbbbbb"):
        _quantiles(
            tmp_path / f"slots-{h}" / "2022-02-09.csv.gz",
            "fm_chronos_daily_ragged_ft",
        )
    rs, refs = _setup()
    built = nowcasts.build(tmp_path, rs, refs, levels=PAGE)
    meta = nowcasts.meta(built, rs, refs, {"aaaaaa": 250, "bbbbbb": 500})
    cols = [m["color"] for m in meta if m["run"] == "s"]
    assert cols == ["hsl(217,80%,66%)", "hsl(217,80%,38%)"]


def test_one_location_has_truth_and_latest_by_target(tmp_path):
    _quantiles(
        tmp_path / "reference" / "2022-02-09.csv.gz", "fm_chronos_daily_ragged"
    )
    rs, refs = _setup()
    built = nowcasts.build(tmp_path, rs, refs, levels=PAGE)
    got = nowcasts.location(built, "DE")
    assert set(got) == {"series", "score", "stats", "truth", "latest"}
    assert got["truth"]["7day"] == {"t": ["2022-02-01"], "y": [10.0]}


def _releases(path):
    weeks = pd.date_range("2026-08-01", periods=6, freq="7D")
    rows = []
    for v in pd.date_range("2026-08-15", periods=5, freq="7D"):
        for w in weeks[weeks <= v]:
            age = (v - w).days
            total = 100 if age >= 14 else 60
            for loc in ("CD", "CD-KN"):
                rows.append(
                    {
                        "kind": "real",
                        "vintage": v.date().isoformat(),
                        "location": loc,
                        "target_end_date": w.date().isoformat(),
                        "total": total,
                        "cutoff": v.date().isoformat(),
                    }
                )
    pd.DataFrame(rows).to_csv(path, index=False)
    return pd.DataFrame(rows)


def test_samples_are_scored_against_the_scorer_s_truth(tmp_path):
    target = _releases(tmp_path / "target.csv")
    raw = tmp_path / "raw" / "reference"
    raw.mkdir(parents=True)
    rng = np.random.default_rng(1)
    draws = rng.poisson(90, 200).astype(float)
    with gzip.open(raw / "2026-08-29.csv.gz", "wt") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "location",
                "target_end_date",
                "sample",
                "value",
                "model",
                "origin",
                "target",
            ]
        )
        for i, v in enumerate(draws):
            w.writerow(
                [
                    "CD",
                    "2026-08-01",
                    i,
                    v,
                    "fm_chronos_daily_ragged",
                    "2026-08-29",
                    "7day",
                ]
            )
    ds = {
        "key": "bdbv",
        "truth_level": 0.98,
        "truth_skip": [],
        "national": "CD",
        "targets": ["7day"],
    }
    out = tmp_path / "summary"
    delay = nowcasts.summarise_samples(
        tmp_path / "raw", out, tmp_path / "target.csv", ds
    )
    real = target.assign(location=target["location"])
    assert delay == reporting_delay(real, 0.98, (), "CD")
    truth = rolling_truth(real, delay)
    y = truth.loc[
        (truth.location == "CD") & (truth.target_end_date == "2026-08-01"),
        "truth",
    ].item()
    with gzip.open(out / "reference" / "2026-08-29.csv.gz", "rt") as f:
        rows = list(csv.DictReader(f))
    assert {r["quantile"] for r in rows} == set(SEVEN)
    assert float(rows[0]["truth"]) == y
    assert float(rows[0]["score"]) == pytest.approx(
        crps_sample(y, draws), abs=1e-3
    )
    assert float(rows[0]["latest"]) == 100


def test_wis_parts_on_both_scales_sum_to_each_scale_s_wis(tmp_path):
    from bdbvhubrenewal.evaluation.scoring import wis_quantile

    _quantiles(
        tmp_path / "reference" / "2022-02-09.csv.gz",
        "fm_chronos_daily_ragged",
        truth=15.5,
    )
    rs, refs = _setup()
    built = nowcasts.build(tmp_path, rs, refs, levels=PAGE)
    parts = built["parts"]["DE"]["zero"]["7day"]["daily"]["2022-02-09"]
    q = [8.0, 9, 10, 11, 12, 13, 14]
    nat = wis_quantile([15.5], [q], nowcasts.LEVELS)
    log = wis_quantile(np.log1p([15.5]), [np.log1p(q)], nowcasts.LEVELS)
    want = (
        [nat[p][0] for p in nowcasts.WIS_PARTS]
        + [log["wis"][0]]
        + [log[p][0] for p in nowcasts.WIS_PARTS]
    )
    assert parts == pytest.approx(want, abs=1e-3)
    # The truth is above every quantile: underprediction only.
    assert parts[1] == 0
    assert parts[0] + parts[2] == pytest.approx(
        built["score"]["DE"]["zero"]["7day"]["daily"]["2022-02-09"], abs=0.01
    )
    assert nowcasts.location(built, "DE")["parts"] == built["parts"]["DE"]


def test_a_file_score_has_no_parts(tmp_path):
    _quantiles(
        tmp_path / "reference" / "2022-02-09.csv.gz",
        "fm_chronos_daily_ragged",
        score=3.25,
    )
    rs, refs = _setup()
    built = nowcasts.build(tmp_path, rs, refs, levels=PAGE)
    assert "zero" not in built["parts"].get("DE", {})
