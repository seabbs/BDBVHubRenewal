"""Nowcast panels for any dataset: quantile or sample forecasts."""

import csv
import gzip
import os

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
    outer=None,
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
    values = [(q, base + i) for i, q in enumerate(SEVEN)]
    values += list(zip(("0.05", "0.95"), outer or (), strict=False))
    with gzip.open(path, "wt") as f:
        w = csv.DictWriter(f, cols)
        w.writeheader()
        for q, value in values:
            row = {
                "model": model,
                "location": location,
                "target_end_date": week,
                "quantile": q,
                "value": value,
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
    cov50, cov95, bias, band = built["stats"]["DE"]["zero"]["7day"]["daily"][
        "2022-02-09"
    ]
    # No 5% and 95% quantiles (the German hub's): the outer band is 95%.
    assert (cov50, cov95, band) == (0.0, 1.0, 95)
    assert (
        built["series"]["DE"]["zero"]["7day"]["daily"]["2022-02-09"]["band"]
        == 95
    )
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
    assert set(got) == {"series", "score", "parts", "stats", "truth", "latest"}
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
    # The seven levels and the 90% interval's.
    assert {r["quantile"] for r in rows} == {*SEVEN, "0.05", "0.95"}
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


def test_with_5_and_95_quantiles_bands_and_coverage_are_90(tmp_path):
    # 90% interval 9.6 to 12.4: truth 9.5 is outside it, inside 95%.
    _quantiles(
        tmp_path / "reference" / "2022-02-09.csv.gz",
        "fm_chronos_daily_ragged",
        truth=9.5,
        outer=(9.6, 12.4),
    )
    rs, refs = _setup()
    built = nowcasts.build(tmp_path, rs, refs, levels=nowcasts.LEVELS)
    got = built["stats"]["DE"]["zero"]["7day"]["daily"]["2022-02-09"]
    assert got[1] == 0.0
    assert got[3] == 90
    series = built["series"]["DE"]["zero"]["7day"]["daily"]["2022-02-09"]
    assert series["band"] == 90
    # The outermost band drawn is the 90% interval.
    assert series["q"][0][0] == 9.6
    assert series["q"][0][-1] == 12.4
    # The WIS stays on the seven levels.
    assert built["score"]["DE"]["zero"]["7day"]["daily"]["2022-02-09"] == (
        pytest.approx(nowcasts.wis([8, 9, 10, 11, 12, 13, 14], 9.5), abs=0.01)
    )


def test_summaries_from_before_the_90_interval_are_written_again(tmp_path):
    raw = tmp_path / "raw" / "reference" / "2026-08-29.csv.gz"
    raw.parent.mkdir(parents=True)
    pd.DataFrame(
        {
            "sample": range(50),
            "value": range(50),
            "model": "m",
            "origin": "2026-08-29",
            "location": "CD",
            "target_end_date": "2026-08-01",
            "target": "7day",
        }
    ).to_csv(raw, index=False)
    target = tmp_path / "target.csv"
    pd.DataFrame(
        {
            "location": ["CD"],
            "target_end_date": ["2026-08-01"],
            "vintage": ["2026-08-29"],
            "total": [10],
            "kind": ["real"],
        }
    ).to_csv(target, index=False)
    out = tmp_path / "summary"
    old = out / "reference" / "2026-08-29.csv.gz"
    old.parent.mkdir(parents=True)
    old.write_bytes(gzip.compress(b"model,quantile\n"))
    future = raw.stat().st_mtime + 100
    os.utime(old, (future, future))
    ds = {"key": "bdbv", "national": "CD", "targets": ["7day"]}
    nowcasts.summarise_samples(tmp_path / "raw", out, target, ds)
    with gzip.open(old, "rt") as f:
        levels = {r["quantile"] for r in csv.DictReader(f)}
    assert {"0.05", "0.95"} <= levels
    assert (out / ".levels").exists()
