"""Checkpoint evaluation: each published checkpoint's scores."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from bdbvhubrenewal.evaluation import checkpoint as ck
from bdbvhubrenewal.evaluation.scoring import coverage_sample, crps_sample


def test_each_label_runs_its_models():
    assert ck.label_models("reference") == (
        "randomwalknowcast",
        "fm_chronos_daily_ragged",
        "fm_chronos_weekly_ragged",
        "fm_chronos_daily_complete",
        "fm_chronos_weekly_complete",
    )
    assert ck.label_models("abc123") == (
        "fm_chronos_daily_ragged_ft",
        "fm_chronos_weekly_ragged_ft",
    )


@pytest.mark.parametrize(
    ("settings", "run"),
    [
        ({"series_bias": True, "sources": ["daily+weekly"]}, "one-model"),
        ({"series_bias": True, "sources": ["daily"]}, "attn-bias"),
        ({"series_slots": {"init": "zero"}}, "slots-zeroinit"),
        ({"series_slots": {"init": "x"}}, "slots"),
        ({}, "no-slots"),
    ],
)
def test_run_of_a_checkpoint(settings, run):
    assert ck.run_of(settings) == run


def test_outbreaks_and_grid_from_settings_or_release_text():
    assert ck.outbreaks_and_grid({"n_outbreaks": 500, "grid": "r5"}, {}) == (
        500,
        "r5",
    )
    release = {
        "name": "x",
        "body": "trained on 2,000 simulated outbreaks, round 4",
    }
    assert ck.outbreaks_and_grid({}, release) == (2000, "r4")
    assert ck.outbreaks_and_grid({}, {}) == (0, "unknown")


def test_development_origins_stop_before_the_scored_rounds():
    vintages = ["2026-10-01", "2026-10-08", "2026-11-21", "2026-11-28"]
    got = ck.development_origins(vintages, skip=["2026-10-08"])
    assert got == ["2026-10-01"]


def _draws(rng, cells, m=200):
    rows = []
    for model, loc, week in cells:
        rows.append(
            pd.DataFrame(
                {
                    "model": model,
                    "origin": "2026-10-01",
                    "location": loc,
                    "target_end_date": week,
                    "sample": np.arange(m),
                    "value": rng.poisson(10, m).astype(float),
                }
            )
        )
    return pd.concat(rows, ignore_index=True)


def test_cell_scores_equal_scoring_each_cell():
    rng = np.random.default_rng(0)
    cells = [
        (m, loc, w)
        for m in ("a", "b")
        for loc in ("CD", "CD-IT", "CD-IT-10")
        for w in ("2026-10-03", "2026-10-10")
    ]
    draws = _draws(rng, cells)
    truth = draws[["location", "target_end_date"]].drop_duplicates()
    truth = truth.assign(truth=rng.poisson(10, len(truth)).astype(float))
    truth["truth_vintage"] = "2026-10-20"
    got = ck.cell_scores(draws, truth)
    assert len(got) == len(cells)
    for _, row in got.iterrows():
        x = draws.loc[
            (draws["model"] == row["model"])
            & (draws["location"] == row["location"])
            & (draws["target_end_date"] == row["target_end_date"]),
            "value",
        ].to_numpy()
        y = row["truth"]
        assert row["crps"] == pytest.approx(crps_sample(y, x))
        assert row["log_crps"] == pytest.approx(
            crps_sample(np.log1p(y), np.log1p(x))
        )
        assert row["coverage_95"] == float(coverage_sample(y, x, 0.95))
    assert set(got["level"]) == {"national", "provinces", "zones"}


def test_zone_truth_is_each_zone_week_in_the_first_release_d_days_on():
    daily_zones = pd.DataFrame(
        {
            "vintage": ["2026-10-10", "2026-10-17", "2026-10-17"],
            "location": ["CD-IT-10"] * 3,
            "onset_date": ["2026-10-01", "2026-10-01", "2026-10-02"],
            "observed": [1, 1, 1],
            "imputed": [0, 0, 0],
        }
    )
    cutoffs = {"2026-10-10": "2026-10-09", "2026-10-17": "2026-10-16"}
    got = ck.zone_truth(daily_zones, cutoffs, ["CD-IT-10", "CD-IT-11"], 7)
    week = got.set_index(["location", "target_end_date"])
    # The week ending 3 Oct is due on 10 Oct: the 17 Oct release, 2 cases.
    assert week.loc[("CD-IT-10", "2026-10-03"), "truth"] == 2
    assert week.loc[("CD-IT-10", "2026-10-03"), "truth_vintage"] == "2026-10-17"
    # A forecast zone with no onset in a release counts 0 there.
    assert week.loc[("CD-IT-11", "2026-10-03"), "truth"] == 0


def _scores(models, checkpoints, levels, base=1.0):
    rows = []
    for model, chk in zip(models, checkpoints, strict=True):
        for level in levels:
            for origin in ("2026-10-01", "2026-10-08"):
                rows.append(
                    {
                        "model": model,
                        "checkpoint": chk,
                        "level": level,
                        "origin": origin,
                        "truth_vintage": "v",
                        "location": "CD",
                        "target_end_date": "w",
                        "crps": base
                        * (2 if model == "randomwalknowcast" else 1),
                        "log_crps": 1.0,
                        "coverage_50": 1.0,
                        "coverage_90": 1.0,
                        "coverage_95": 1.0,
                    }
                )
    return pd.DataFrame(rows)


def test_relative_divides_by_the_baseline_by_level():
    rows = _scores(
        ["randomwalknowcast", "fm_chronos_daily_ragged"],
        ["zero-shot", "abc"],
        ["national", "zones"],
    )
    got = ck.relative(
        ck.matched(rows, ck.BDBV_CELL), "ebola", ("crps", "log_crps")
    )
    mine = got.loc[got["checkpoint"] == "abc"]
    assert list(mine["rel_crps"]) == [0.5, 0.5]
    assert set(got["rule"]) == {ck.RULE}
    assert list(got.columns) == ck.COLUMNS


def test_merge_keeps_other_checkpoints_rule_rows():
    old = pd.DataFrame(
        [{"study": "ebola", "rule": ck.RULE, "checkpoint": "old", "level": "x"}]
    )
    new = pd.DataFrame(
        [{"study": "ebola", "rule": ck.RULE, "checkpoint": "abc", "level": "y"}]
    )
    got = ck.merge_evidence(old, new)
    assert sorted(got["checkpoint"]) == ["abc", "old"]


def test_merge_keeps_other_rules_rows_and_replaces_the_rules():
    old = pd.DataFrame(
        [
            {
                "study": "ebola",
                "rule": ck.RULE,
                "checkpoint": "abc",
                "level": "x",
            },
            {
                "study": "ebola",
                "rule": "other",
                "checkpoint": "abc",
                "level": "x",
            },
        ]
    )
    new = pd.DataFrame(
        [{"study": "ebola", "rule": ck.RULE, "checkpoint": "abc", "level": "y"}]
    )
    got = ck.merge_evidence(old, new)
    assert sorted(got["level"]) == ["x", "y"]
    assert set(got.loc[got["rule"] == ck.RULE, "level"]) == {"y"}


def test_a_unit_is_done_when_every_origin_is_cached(tmp_path):
    root = tmp_path / "bdbv" / "abc"
    root.mkdir(parents=True)
    (root / "2026-10-01.csv.gz").write_text("x")
    assert not ck.unit_done(root, ["2026-10-01", "2026-10-08"])
    (root / "2026-10-08.csv.gz").write_text("x")
    assert ck.unit_done(root, ["2026-10-01", "2026-10-08"])


def test_held_out_rows_from_settings():
    settings = {
        "held_out_loss": 1.5,
        "held_out_loss_start": 2.0,
        "held_out_loss_by_level": {"leaf": 1.2},
        "n_validation": 30,
    }
    got = ck.held_out_rows("abc", settings)
    assert list(got["level"]) == ["all", "leaf"]
    assert got["loss"].tolist() == [1.5, 1.2]
    assert list(got.columns) == ck.COLUMNS


def test_the_reported_model_is_the_live_runs_final_checkpoint():
    rows = pd.DataFrame(
        {
            "run": ["one-model", "one-model", "no-slots"],
            "grid": ["r5", "r5", "r5"],
            "outbreaks": [8400, 500, 8400],
        }
    )
    assert ck.chosen(rows).tolist() == [True, False, False]


def test_the_cli_names_its_units():
    with pytest.raises(SystemExit):
        ck.main(["nope", "cache"])


def test_release_unit_writes_settings_and_release(tmp_path, monkeypatch):
    def fake(cmd, **_kw):
        class Out:
            stdout = json.dumps({"name": "n", "body": "b"})

        if "download" in cmd:
            out = cmd[cmd.index("-O") + 1]
            Path(out).write_text(json.dumps({"grid": "r5"}))
        return Out()

    monkeypatch.setattr(ck.subprocess, "run", fake)
    ck.release_unit(tmp_path, "abc")
    assert json.loads((tmp_path / "heldout" / "abc.json").read_text()) == {
        "grid": "r5"
    }
    assert (tmp_path / "heldout" / "abc-release.json").exists()


def test_cached_draws_keep_the_7day_target(tmp_path):
    rows = pd.DataFrame(
        {
            "origin": ["2026-10-01"] * 2,
            "location": ["CD"] * 2,
            "target_end_date": ["2026-10-03"] * 2,
            "value": [1.0, 2.0],
            "target": ["7day", "daily"],
        }
    )
    rows.to_csv(tmp_path / "2026-10-01.csv.gz", index=False)
    got = ck._cached(tmp_path)
    assert got["value"].tolist() == [1.0]
    assert "target" not in got


def test_tables_carry_wis_and_its_components():
    for c in ("wis", "dispersion", "overprediction", "underprediction"):
        assert c in ck.COLUMNS
    rng = np.random.default_rng(1)
    draws = _draws(rng, [("a", "CD", "2026-10-03")])
    truth = pd.DataFrame(
        {"location": ["CD"], "target_end_date": ["2026-10-03"],
         "truth": [10.0], "truth_vintage": ["v"]}
    )  # fmt: skip
    got = ck.cell_scores(draws, truth).iloc[0]
    parts = got["dispersion"] + got["overprediction"] + got["underprediction"]
    assert parts == pytest.approx(got["wis"])


def test_relative_rows_mean_the_wis_components():
    rows = _scores(
        ["randomwalknowcast", "fm_chronos_daily_ragged"],
        ["zero-shot", "abc"],
        ["national"],
    ).assign(wis=3.0, dispersion=1.0, overprediction=0.5, underprediction=1.5)
    got = ck.relative(rows, "ebola", ("crps", "log_crps"))
    row = got.loc[got["checkpoint"] == "abc"].iloc[0]
    assert row["wis"] == 3.0
    assert (
        row["dispersion"] + row["overprediction"] + row["underprediction"]
        == 3.0
    )
