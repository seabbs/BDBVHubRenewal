"""Tests of scripts/checkpoint_scores.py's pure parts."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from bdbvhubrenewal.evaluation.marginals import COMMON_STREAM
from bdbvhubrenewal.evaluation.scoring import crps_sample

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "checkpoint_scores.py"


@pytest.fixture(scope="module")
def cs():
    spec = importlib.util.spec_from_file_location("checkpoint_scores", SCRIPT)
    assert spec is not None
    assert spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules["checkpoint_scores"] = mod
    spec.loader.exec_module(mod)
    return mod


def _release(tag: str, published: str) -> dict:
    return {"tagName": tag, "publishedAt": published}


RELEASES = [
    _release("chronos-ft-e8a1f5ae338b", "2026-11-15T16:21:13Z"),
    _release("chronos-ft-bbbbbbbbbbbb", "2026-11-17T09:00:00Z"),
    _release("round-2026-11-14", "2026-11-12T14:28:29Z"),
    _release("chronos-ft-aaaaaaaaaaaa", "2026-11-16T09:00:00Z"),
    _release("realdata-ft-cccccccccccc", "2026-11-17T01:00:00Z"),
    _release("chronos-ft-2c2b22a03107", "2026-11-13T14:59:14Z"),
]


def test_only_the_new_run_s_checkpoints_are_scored_oldest_first(cs) -> None:
    # The paper uses only the continuous run from 2026-11-16 09:00 UTC:
    # r3 and earlier checkpoints are never scored.
    assert cs.PUBLISHED_FROM == "2026-11-16T09:00:00Z"
    queue = cs.checkpoint_queue(RELEASES, scored=set())
    assert queue == ["aaaaaaaaaaaa", "bbbbbbbbbbbb"]
    assert not hasattr(cs, "BACKLOG")


def test_scored_and_foreign_tags_are_left_out(cs) -> None:
    queue = cs.checkpoint_queue(RELEASES, scored={"aaaaaaaaaaaa"})
    assert queue == ["bbbbbbbbbbbb"]
    assert "cccccccccccc" not in queue


def test_a_checkpoint_is_scored_once_every_study_has_rule_rows(cs) -> None:
    rows = pd.DataFrame(
        {
            "study": ["germany", "ebola", "held_out", "germany", "ebola"],
            "checkpoint": ["a", "a", "a", "b", "b"],
            "rule": [cs.RULE] * 5,
        }
    )
    old = pd.DataFrame(
        {
            "study": ["germany", "ebola", "held_out"],
            "checkpoint": ["c"] * 3,
            "rule": [cs.OLD_RULE] * 3,
        }
    )
    assert cs.scored_checkpoints(pd.concat([rows, old])) == {"a"}
    # Old evidence has no rule column: nothing is scored under the rule.
    assert cs.scored_checkpoints(old.drop(columns="rule")) == set()


R3 = {
    "grid": "r3",
    "n_outbreaks": 4000,
    "n_validation": 2098,
    "held_out_loss_start": 0.29,
    "held_out_loss": 0.18,
    "held_out_loss_by_level": {
        "leaf": 0.16,
        "national": 0.12,
        "intermediate": 0.08,
    },
}


def test_held_out_rows_read_the_trainer_losses(cs) -> None:
    rows = cs.held_out_rows("e8a1f5ae338b", R3)
    by = rows.set_index("level")
    assert list(by.index) == ["all", "leaf", "national", "intermediate"]
    assert by.loc["all", "loss"] == 0.18
    assert by.loc["all", "loss_zero_shot"] == 0.29
    assert by.loc["leaf", "loss"] == 0.16
    assert (rows["study"] == "held_out").all()
    assert (rows["n"] == 2098).all()
    assert (rows["n_origins"] == 0).all()
    with_real = cs.held_out_rows("x", {**R3, "real_validation_loss": 0.5})
    assert with_real.set_index("level").loc["real_validation", "loss"] == 0.5


def test_held_out_rows_mark_missing_losses(cs) -> None:
    rows = cs.held_out_rows("2c2b22a03107", {"n_validation": 786})
    assert rows["level"].tolist() == ["all"]
    assert rows["loss"].isna().all()


def test_outbreaks_and_grid_fall_back_to_the_release_text(cs) -> None:
    assert cs.outbreaks_and_grid(R3, {"name": "", "body": ""}) == (4000, "r3")
    release = {
        "name": "Chronos-2 fine-tuned checkpoint 2c2b22a03107",
        "body": "fine-tuned on 2,000 simulated epinowcast outbreaks "
        "(round 1). The models check",
    }
    assert cs.outbreaks_and_grid({}, release) == (2000, "r1")
    assert cs.outbreaks_and_grid({}, {"name": "", "body": ""}) == (
        0,
        "unknown",
    )


def test_chronos_draws_share_one_stream_and_others_keep_theirs(cs) -> None:
    import dataclasses

    from bdbvhubrenewal.forecasters import get_model

    zero = get_model("fm_chronos_daily_ragged")
    # The fine-tuned model differs from the zero-shot one by its name and
    # checkpoint; its own stream would be keyed on its name.
    ft = dataclasses.replace(zero, name="fm_chronos_daily_ragged_ft")
    assert (zero.stream or zero.name) != (ft.stream or ft.name)
    assert cs.common(zero).stream == cs.common(ft).stream == COMMON_STREAM
    rw = cs.common(get_model("randomwalknowcast"))
    assert getattr(rw, "stream", None) != COMMON_STREAM


def _samples(model: str, draws: np.ndarray, location: str) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "model": model,
            "origin": "2026-10-17",
            "location": location,
            "target_end_date": "2026-10-10",
            "sample": np.arange(1, len(draws) + 1),
            "value": draws,
        }
    )


def test_cell_scores_are_sample_crps_and_log_crps(cs) -> None:
    rng = np.random.default_rng(1)
    draws = rng.poisson(20, size=1000).astype(float)
    truth = pd.DataFrame(
        {
            "location": ["CD"],
            "target_end_date": ["2026-10-10"],
            "truth": [25.0],
            "truth_vintage": ["2026-11-15"],
        }
    )
    sc = cs.cell_scores(_samples("m", draws, "CD"), truth)
    assert len(sc) == 1
    assert sc["crps"].iloc[0] == pytest.approx(crps_sample(25.0, draws))
    assert sc["log_crps"].iloc[0] == pytest.approx(
        crps_sample(np.log1p(25.0), np.log1p(draws))
    )
    assert sc["truth_vintage"].iloc[0] == "2026-11-15"
    # A week with no truth yet is not scored.
    assert cs.cell_scores(_samples("m", draws, "CD-KN"), truth).empty


def _cells(model: str, checkpoint: str, crps: float) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "model": model,
            "checkpoint": checkpoint,
            "origin": ["2026-10-17", "2026-10-17"],
            "truth_vintage": ["2026-11-15"] * 2,
            "location": ["CD", "CD-KN"],
            "target_end_date": ["2026-10-10"] * 2,
            "level": ["national", "provinces"],
            "crps": [crps, crps],
            "log_crps": [crps / 10, crps / 10],
            "coverage_50": [1.0, 0.0],
            "coverage_90": [1.0, 1.0],
            "coverage_95": [1.0, 1.0],
        }
    )


def test_ebola_rows_are_crps_relative_to_randomwalknowcast(cs) -> None:
    cells = pd.concat(
        [
            _cells("randomwalknowcast", "zero-shot", 2.0),
            _cells("fm_chronos_daily_ragged", "zero-shot", 4.0),
            _cells("fm_chronos_daily_ragged", "e8a1f5ae338b", 1.0),
        ]
    )
    rows = cs.relative(cells, "ebola", ("crps", "log_crps"))
    assert list(rows.columns) == cs.COLUMNS
    by = rows.set_index(["level", "model", "checkpoint"])
    base = by.loc[("national", "randomwalknowcast", "zero-shot")]
    assert base["rel_crps"] == 1
    assert by.loc[
        ("national", "fm_chronos_daily_ragged", "e8a1f5ae338b"), "rel_crps"
    ] == pytest.approx(0.5)
    assert rows["rel_wis"].isna().all()
    assert (rows["rule"] == cs.RULE).all()


def test_merging_keeps_old_rows_and_replaces_rule_rows(cs) -> None:
    old = pd.DataFrame(
        {
            "study": ["germany"],
            "level": ["national"],
            "model": ["fm_chronos_daily_ragged"],
            "checkpoint": ["6b7aaa8d4f55"],
            "outbreaks": [1000],
            "n": [400],
            "rel_wis": [2.1],
        }
    )
    first = cs.held_out_rows("e8a1f5ae338b", R3)
    merged = cs.merge_evidence(old, first)
    assert list(merged.columns) == cs.COLUMNS
    assert merged.loc[merged["study"] == "germany", "rule"].tolist() == [
        cs.OLD_RULE
    ]
    again = cs.merge_evidence(merged, cs.held_out_rows("e8a1f5ae338b", R3))
    assert len(again) == len(merged)


def test_the_paper_reads_the_new_rows(cs, monkeypatch) -> None:
    monkeypatch.syspath_prepend(str(ROOT / "paper"))
    paper = pytest.importorskip("paperpipe.evidence")
    rows = pd.concat(
        [
            cs.relative(
                pd.concat(
                    [
                        _cells("randomwalknowcast", "zero-shot", 2.0),
                        _cells("fm_chronos_daily_ragged", "zero-shot", 4.0),
                        _cells("fm_chronos_daily_ragged", "e8a1f5ae338b", 1),
                    ]
                ),
                "ebola",
                ("crps", "log_crps"),
            ),
            cs.held_out_rows("e8a1f5ae338b", R3),
        ]
    )
    rows["outbreaks"] = rows["checkpoint"].map({"e8a1f5ae338b": 4000})
    rows["outbreaks"] = rows["outbreaks"].fillna(0).astype(int)
    paper.checkpoint_numbers(rows)
    paper.checkpoint_figure(rows)


def test_jax_compile_caches_are_freed_after_each_german_date(
    cs, tmp_path, monkeypatch
) -> None:
    # A long unit compiles JAX functions of new shapes at each date; kept,
    # they filled the unit's 6 GB scope ("LLVM compilation error: Cannot
    # allocate memory") after 29 of 80 dates.
    from bdbvhubrenewal.data import germany as data_de
    from bdbvhubrenewal.evaluation import germany as de

    events: list[str] = []
    config = type("C", (), {"triangle": "t.csv", "n_samples": 10,
                            "seed": 1, "truth_delay": 40})()  # fmt: skip
    monkeypatch.setattr(de, "load_germany_config", lambda _: config)
    monkeypatch.setattr(data_de, "hub_triangle", lambda _: None)
    monkeypatch.setattr(
        de, "nowcast_dates", lambda *_, **__: ["2021-11-22", "2021-11-23"]
    )

    def forecasts(_models, _triangle, dates, _n, _seed):
        events.append(f"forecast {dates[0]}")
        return pd.DataFrame({"value": [1.0]})

    monkeypatch.setattr(de, "germany_forecasts", forecasts)
    monkeypatch.setattr(de, "with_truth", lambda fc, *_: fc)
    monkeypatch.setattr(cs, "free_caches", lambda: events.append("free"))
    monkeypatch.delenv("BDBV_CHRONOS_FT", raising=False)
    cs.germany_unit(tmp_path, cs.REFERENCE)
    assert events == [
        "forecast 2021-11-22",
        "free",
        "forecast 2021-11-23",
        "free",
    ]


def test_freeing_caches_clears_jax_s_compiled_functions(cs) -> None:
    import jax
    import jax.numpy as jnp

    f = jax.jit(lambda x: x + 1)
    f(jnp.ones(3))
    assert f._cache_size() == 1
    cs.free_caches()
    assert f._cache_size() == 0


def test_slots_checkpoints_are_scored_first_under_their_paper_label(
    cs,
) -> None:
    # The paper's evidence comes from the final slots run: its releases are
    # chronos-slots-<hash>, pinned and named slots-<hash>, ahead of the
    # no-slots run's.
    releases = [
        *RELEASES,
        _release("chronos-slots-dddddddddddd", "2026-11-17T12:00:00Z"),
        _release("chronos-slots-eeeeeeeeeeee", "2026-11-17T10:00:00Z"),
    ]
    queue = cs.checkpoint_queue(releases, scored={"slots-eeeeeeeeeeee"})
    assert queue == ["slots-dddddddddddd", "aaaaaaaaaaaa", "bbbbbbbbbbbb"]


def test_each_checkpoint_label_maps_to_its_release(cs) -> None:
    assert cs.release_of("slots-dddddddddddd") == "chronos-slots-dddddddddddd"
    assert cs.release_of("aaaaaaaaaaaa") == "chronos-ft-aaaaaaaaaaaa"


def test_the_complete_data_models_are_a_zero_shot_reference(cs) -> None:
    # Sam 16:55: Chronos-2 given only complete data points, zero-shot.
    assert cs.COMPLETE == (
        "fm_chronos_daily_complete",
        "fm_chronos_weekly_complete",
    )
    assert set(cs.COMPLETE) <= set(cs.label_models(cs.REFERENCE))
    assert cs.label_models(cs.COMPLETE_LABEL) == cs.COMPLETE
    rows = pd.DataFrame({"model": ["fm_chronos_daily_complete"]})
    assert cs._tag(rows, cs.COMPLETE_LABEL)["checkpoint"].item() == "zero-shot"


def test_complete_forecasts_merge_into_the_reference_files(cs, tmp_path):
    for study in ("germany", "bdbv"):
        ref = tmp_path / study / cs.REFERENCE
        new = tmp_path / study / cs.COMPLETE_LABEL
        ref.mkdir(parents=True)
        new.mkdir(parents=True)
        pd.DataFrame({"model": ["randomwalknowcast"], "value": [1.0]}).to_csv(
            ref / "d1.csv.gz", index=False
        )
        pd.DataFrame(
            {"model": ["fm_chronos_daily_complete"], "value": [2.0]}
        ).to_csv(new / "d1.csv.gz", index=False)
    cs.merge_complete(tmp_path)
    cs.merge_complete(tmp_path)  # a second merge adds nothing
    for study in ("germany", "bdbv"):
        out = pd.read_csv(tmp_path / study / cs.REFERENCE / "d1.csv.gz")
        assert out["model"].tolist() == [
            "randomwalknowcast",
            "fm_chronos_daily_complete",
        ]
    assert (tmp_path / cs.COMPLETE_STAMP).exists()


def test_a_slots_cache_has_one_folder_named_by_its_label(cs, tmp_path):
    # One folder per checkpoint, slots-<hash>, which the dashboard mirrors.
    root = cs.unit_root(tmp_path, "germany", "slots-abcdef123456")
    assert root == tmp_path / "germany" / "slots-abcdef123456"
    assert [p.name for p in (tmp_path / "germany").iterdir()] == [root.name]


def test_rows_are_built_only_from_finished_units(cs, tmp_path):
    (tmp_path / "heldout").mkdir()
    ready = cs.ready_labels(tmp_path, "germany", ["reference", "abc"])
    assert ready == []


def test_interim_rows_are_kept_when_rows_are_rebuilt(cs) -> None:
    # A slots checkpoint with BDBV rows only is not scored, but a rebuild
    # must carry it, or its rows would be dropped.
    rows = pd.DataFrame(
        {
            "study": ["ebola", "held_out", "germany", "ebola", "ebola"],
            "checkpoint": ["slots-a", "slots-a", "zero-shot", "zero-shot", "x"],
            "rule": [cs.RULE] * 4 + [cs.OLD_RULE],
        }
    )
    assert cs.checkpoints_with_rows(rows) == {"slots-a"}


def test_each_checkpoint_s_rows_name_its_run(cs) -> None:
    zero = {"series_slots": {"n": 128, "init": "zero"}}
    rand = {"series_slots": {"n": 128, "init": "normal"}}
    assert cs.run_of(zero) == "slots-zeroinit"
    assert cs.run_of(rand) == "slots"
    assert cs.run_of({}) == "no-slots"
    assert "run" in cs.COLUMNS


def test_an_origin_no_model_can_forecast_is_cached_empty(cs) -> None:
    # The complete-data models need two releases at most 42 days apart,
    # which the first origin lacks: the origin is cached with no rows, so
    # the unit finishes, rather than failing.
    out = cs.origin_draws([], "2026-10-09")
    assert out.empty
    assert {"location", "target_end_date", "sample", "value", "model"} <= set(
        out.columns
    )
    one = pd.DataFrame(
        {
            "location": ["CD"],
            "target_end_date": ["2026-10-03"],
            "sample": [1],
            "value": [2.0],
            "model": ["m"],
        }
    )
    got = cs.origin_draws([one], "2026-10-09")
    assert got["origin"].tolist() == ["2026-10-09"]
    assert got["value"].dtype == "float32"


# The daily target ---------------------------------------------------------


def test_evidence_rows_carry_a_target_and_old_rows_are_7day(cs) -> None:
    assert cs.TARGETS == ("7day", "daily")
    assert "target" in cs.COLUMNS
    old = pd.DataFrame(
        {"study": ["germany"], "model": ["m"], "checkpoint": ["a"]}
    )
    merged = cs.merge_evidence(old, cs.held_out_rows("e8a1f5ae338b", R3))
    assert set(merged["target"]) == {"7day"}


def test_a_cache_file_with_no_target_column_is_7day(cs, tmp_path) -> None:
    path = tmp_path / "2021-11-22.csv.gz"
    pd.DataFrame({"model": ["m"], "value": [1.0]}).to_csv(path, index=False)
    assert cs.cached_targets(path) == {"7day"}
    assert cs.read_cache(path)["target"].tolist() == ["7day"]
    assert cs.cached_targets(tmp_path / "missing.csv.gz") == set()
    both = pd.DataFrame({"model": ["m", "m"], "target": ["7day", "daily"]})
    both.to_csv(path, index=False)
    assert cs.cached_targets(path) == {"7day", "daily"}


def test_the_daily_target_runs_our_daily_models_only(cs) -> None:
    assert cs.label_models(cs.REFERENCE, "daily") == (
        "randomwalknowcast",
        "fm_chronos_daily_ragged",
    )
    assert cs.label_models("slots-abc", "daily") == (
        "fm_chronos_daily_ragged_ft",
    )
    # The weekly models have no daily target, nor do the complete ones.
    assert cs.label_models(cs.COMPLETE_LABEL, "daily") == ()
    for label in (cs.REFERENCE, "slots-abc"):
        assert not any("weekly" in m for m in cs.label_models(label, "daily"))
    # The 7-day target is as before.
    assert cs.label_models("slots-abc") == cs.FINETUNED
    assert cs.label_models("slots-abc", "7day") == cs.FINETUNED


def test_the_reference_and_the_new_run_take_the_daily_target(
    cs, tmp_path
) -> None:
    held = tmp_path / "heldout"
    held.mkdir()
    runs = {
        "slots-new": {
            "series_bias": {"init": "zero"},
            "sources": ["daily", "weekly", "daily+weekly"],
        },
        "slots-bias": {"series_bias": {"init": "zero"}},
        "slots-old": {"series_slots": {"n": 128, "init": "normal"}},
        "slots-zero": {"series_slots": {"n": 128, "init": "zero"}},
        "old": {},
    }
    for label, settings in runs.items():
        (held / f"{label}.json").write_text(json.dumps(settings))
    assert cs.targets_for(tmp_path, cs.REFERENCE) == ("7day", "daily")
    assert cs.targets_for(tmp_path, "slots-new") == ("7day", "daily")
    # Old runs take it only when asked (`germany CACHE LABEL daily`).
    assert cs.targets_for(tmp_path, "slots-zero") == ("7day",)
    assert cs.targets_for(tmp_path, "slots-old") == ("7day",)
    assert cs.targets_for(tmp_path, "slots-bias") == ("7day",)
    assert cs.targets_for(tmp_path, "old") == ("7day",)
    assert cs.targets_for(tmp_path, cs.COMPLETE_LABEL) == ("7day",)
    # A checkpoint whose settings are not fetched yet is 7-day only.
    assert cs.targets_for(tmp_path, "slots-unknown") == ("7day",)


def _unit_stubs(cs, monkeypatch, events: list[str]) -> None:
    from bdbvhubrenewal.data import germany as data_de
    from bdbvhubrenewal.evaluation import germany as de

    config = type("C", (), {"triangle": "t.csv", "n_samples": 10,
                            "seed": 1, "truth_delay": 40})()  # fmt: skip
    monkeypatch.setattr(de, "load_germany_config", lambda _: config)
    monkeypatch.setattr(data_de, "hub_triangle", lambda _: None)
    monkeypatch.setattr(
        de, "nowcast_dates", lambda *_, **__: ["2021-11-22", "2021-11-23"]
    )

    def weekly(models, _triangle, dates, _n, _seed):
        events.append(f"7day {dates[0]} {','.join(models)}")
        return pd.DataFrame({"model": list(models), "value": 7.0})

    def daily(models, _triangle, dates, _n, _seed):
        events.append(f"daily {dates[0]} {','.join(models)}")
        return pd.DataFrame({"model": list(models), "value": 1.0})

    def truth(fc, _triangle, _delay, window=7):
        return fc.assign(truth=float(window))

    monkeypatch.setattr(de, "germany_forecasts", weekly)
    monkeypatch.setattr(de, "germany_daily_forecasts", daily, raising=False)
    monkeypatch.setattr(de, "with_truth", truth)
    monkeypatch.setattr(cs, "free_caches", lambda: None)
    monkeypatch.delenv("BDBV_CHRONOS_FT", raising=False)


def test_a_unit_adds_only_the_missing_target(cs, tmp_path, monkeypatch):
    events: list[str] = []
    _unit_stubs(cs, monkeypatch, events)
    root = tmp_path / "germany" / cs.REFERENCE
    root.mkdir(parents=True)
    # Cached before the daily target: no target column, all 7-day.
    old = pd.DataFrame({"model": ["randomwalknowcast"], "value": [3.0]})
    for d in ("2021-11-22", "2021-11-23"):
        old.to_csv(root / f"{d}.csv.gz", index=False)
    assert cs.unit_done(tmp_path, "germany", cs.REFERENCE)
    assert not cs.unit_done(tmp_path, "germany", cs.REFERENCE, "daily")
    cs.germany_unit(tmp_path, cs.REFERENCE, ("7day", "daily"))
    models = "randomwalknowcast,fm_chronos_daily_ragged"
    assert events == [
        f"daily 2021-11-22 {models}",
        f"daily 2021-11-23 {models}",
    ]
    got = cs.read_cache(root / "2021-11-22.csv.gz")
    seven = got.loc[got["target"] == "7day"]
    assert seven["value"].tolist() == [3.0]  # the cached forecasts are kept
    day = got.loc[got["target"] == "daily"]
    assert set(day["model"]) == set(models.split(","))
    assert (day["truth"] == 1.0).all()  # the one-day window's truth
    assert cs.unit_done(tmp_path, "germany", cs.REFERENCE, "daily")
    # A second run has nothing to add.
    cs.germany_unit(tmp_path, cs.REFERENCE, ("7day", "daily"))
    assert len(events) == 2


def test_a_new_unit_writes_both_targets_in_one_file(cs, tmp_path, monkeypatch):
    events: list[str] = []
    _unit_stubs(cs, monkeypatch, events)
    cs.germany_unit(tmp_path, "slots-abc", ("7day", "daily"))
    assert [e.split()[0] for e in events] == ["7day", "daily"] * 2
    got = cs.read_cache(
        tmp_path / "germany" / "slots-abc" / "2021-11-22.csv.gz"
    )
    assert got.groupby("target")["truth"].first().to_dict() == {
        "7day": 7.0,
        "daily": 1.0,
    }
    # 7-day only, as the poller runs it, is unchanged: no daily rows.
    events.clear()
    cs.germany_unit(tmp_path, "old", ("7day",))
    assert all(e.startswith("7day") for e in events)


def _target_cells(model: str, target: str, crps: float) -> pd.DataFrame:
    out = _cells(model, "zero-shot", crps).assign(target=target)
    if target == "daily":
        out["target_end_date"] = "2026-10-08"  # a day, not a Saturday
    return out


def test_relative_scores_divide_by_the_baseline_of_their_target(cs) -> None:
    cells = pd.concat(
        [
            _target_cells("randomwalknowcast", "7day", 2.0),
            _target_cells("fm_chronos_daily_ragged", "7day", 4.0),
            _target_cells("randomwalknowcast", "daily", 1.0),
            _target_cells("fm_chronos_daily_ragged", "daily", 4.0),
        ]
    )
    rows = cs.relative(cells, "ebola", ("crps", "log_crps"))
    assert list(rows.columns) == cs.COLUMNS
    by = rows.set_index(["target", "level", "model"])["rel_crps"]
    assert by[("7day", "national", "fm_chronos_daily_ragged")] == 2.0
    assert by[("daily", "national", "fm_chronos_daily_ragged")] == 4.0
    assert by[("daily", "national", "randomwalknowcast")] == 1.0
    # Cells with no target are the 7-day target's.
    plain = cs.relative(
        pd.concat(
            [
                _cells("randomwalknowcast", "zero-shot", 2.0),
                _cells("fm_chronos_daily_ragged", "zero-shot", 4.0),
            ]
        ),
        "ebola",
        ("crps", "log_crps"),
    )
    assert set(plain["target"]) == {"7day"}


def test_cells_are_matched_within_each_target(cs) -> None:
    # The weekly model has 7-day cells only; the daily target's cells are
    # matched over the models it has.
    cells = pd.concat(
        [
            _target_cells("randomwalknowcast", "7day", 2.0),
            _target_cells("fm_chronos_weekly_ragged", "7day", 3.0),
            _target_cells("randomwalknowcast", "daily", 1.0),
            _target_cells("fm_chronos_daily_ragged", "daily", 4.0),
        ]
    )
    out = cs.matched(cells, cs.BDBV_CELL)
    assert len(out) == len(cells)
    assert set(out.loc[out["target"] == "daily", "model"]) == {
        "randomwalknowcast",
        "fm_chronos_daily_ragged",
    }


def _quantile_file(
    models: list[str], target: str, horizon: int
) -> pd.DataFrame:
    from bdbvhubrenewal.evaluation.germany import QUANTILE_LEVELS

    rows = []
    for i, m in enumerate(models):
        for q in QUANTILE_LEVELS:
            rows.append(
                {
                    "model": m,
                    "origin": "2021-11-22",
                    "location": "DE",
                    "target_end_date": "2021-11-20",
                    "horizon": horizon,
                    "quantile": q,
                    "value": 100.0 + 10 * i + 50 * q,
                    "truth": 120.0,
                    "truth_vintage": "2021-12-30",
                    "target": target,
                }
            )
    return pd.DataFrame(rows)


def test_hub_teams_take_no_part_in_the_daily_target(cs, tmp_path) -> None:
    root = tmp_path / "germany" / cs.REFERENCE
    root.mkdir(parents=True)
    ours = ["randomwalknowcast", "fm_chronos_daily_ragged"]
    pd.concat(
        [
            _quantile_file([*ours, "MeanEnsemble"], "7day", -2),
            _quantile_file([*ours, "MeanEnsemble"], "daily", -2),
        ]
    ).to_csv(root / "2021-11-22.csv.gz", index=False)
    rows = cs.germany_rows(tmp_path, [cs.REFERENCE], [cs.REFERENCE])
    seven = rows.loc[rows["target"] == "7day"]
    daily = rows.loc[rows["target"] == "daily"]
    assert set(seven["model"]) == {*ours, "MeanEnsemble"}
    assert set(daily["model"]) == set(ours)
    base = daily.set_index("model").loc["randomwalknowcast", "rel_wis"]
    assert base == 1.0
    # With no daily labels there are no daily rows.
    only = cs.germany_rows(tmp_path, [cs.REFERENCE])
    assert set(only["target"]) == {"7day"}


def test_bdbv_daily_truth_is_each_day_in_the_first_release_d_days_on(
    cs,
) -> None:
    # Releases list only days with a case: a day a release covers with no
    # row is 0 in that release.
    daily = pd.DataFrame(
        {
            "vintage": ["r1", "r1", "r2", "r3"],
            "location": ["CD-KN"] * 4,
            "onset_date": [
                "2026-09-01",
                "2026-09-02",
                "2026-09-02",
                "2026-09-01",
            ],
            "observed": [1, 2, 3, 9],
            "imputed": [0, 0, 1, 0],
            "cutoff": ["2026-09-05", "2026-09-05", "2026-09-12", "2026-09-30"],
        }
    )
    truth = cs.bdbv_daily_truth(daily, 10)
    by = truth.set_index(["location", "target_end_date"])
    # 09-01 + 10 days is 09-11: release r2 (cut-off 09-12) has no row for
    # it, so its truth is 0, not r3's 9.
    assert by.loc[("CD-KN", "2026-09-01"), "truth"] == 0
    assert by.loc[("CD-KN", "2026-09-01"), "truth_vintage"] == "r2"
    assert by.loc[("CD-KN", "2026-09-02"), "truth"] == 4
    # The country is the sum of the provinces.
    assert by.loc[("CD", "2026-09-02"), "truth"] == 4
    # A day no release reaches D days on has no truth.
    assert ("CD-KN", "2026-09-25") not in by.index


def test_the_one_model_run_is_told_from_the_superseded_ones(cs) -> None:
    bias = {"series_bias": {"init": "zero"}}
    one = {**bias, "sources": ["daily", "weekly", "3day", "daily+weekly"]}
    assert cs.run_of(one) == "one-model"
    assert cs.run_of({**bias, "sources": ["weekly", "daily"]}) == "attn-bias"
    assert cs.run_of({"series_slots": {"init": "zero"}}) == "slots-zeroinit"
    assert cs.run_of({"series_slots": {"init": "normal"}}) == "slots"
    assert cs.run_of({}) == "no-slots"
    assert {"one-model"} == cs.CURRENT_RUNS


def test_daily_baseline_rows_name_their_provisional_rule(cs) -> None:
    # randomwalknowcast nowcasts weeks: its daily forecasts split each
    # week's draws over the days (`split_week_draws`), a provisional
    # choice the rows name so the paper can footnote or drop them.
    cells = pd.concat(
        [
            _target_cells("randomwalknowcast", "7day", 2.0),
            _target_cells("randomwalknowcast", "daily", 1.0),
            _target_cells("fm_chronos_daily_ragged", "daily", 4.0),
        ]
    )
    rows = cs.relative(cells, "ebola", ("crps", "log_crps"))
    assert "daily_rule" in cs.COLUMNS
    national = rows.loc[rows["level"] == "national"]
    rule = national.set_index(["target", "model"])["daily_rule"]
    assert rule[("daily", "randomwalknowcast")] == cs.DAILY_RULE
    assert cs.DAILY_RULE == "week-split"
    assert pd.isna(rule[("7day", "randomwalknowcast")])
    assert pd.isna(rule[("daily", "fm_chronos_daily_ragged")])


def test_a_daily_unit_is_done_for_the_dates_it_wrote(
    cs, tmp_path, monkeypatch
) -> None:
    # New BDBV origins arrive after a daily unit finished: it is not done
    # until they have the daily target too, though their files exist.
    monkeypatch.setattr(cs, "bdbv_origins", lambda: ["o1", "o2"])
    root = tmp_path / "bdbv" / "reference"
    root.mkdir(parents=True)
    for o in ("o1", "o2"):
        (root / f"{o}.csv.gz").write_text("")
    (root / cs.DAILY_DONE).write_text("o1\n")
    assert cs.unit_done(tmp_path, "bdbv", "reference")
    assert not cs.unit_done(tmp_path, "bdbv", "reference", "daily")
    (root / cs.DAILY_DONE).write_text("o1\no2\n")
    assert cs.unit_done(tmp_path, "bdbv", "reference", "daily")


def test_a_daily_unit_needs_the_7_day_forecasts_first(
    cs, tmp_path, monkeypatch
) -> None:
    events: list[str] = []
    _unit_stubs(cs, monkeypatch, events)
    with pytest.raises(SystemExit, match="7-day"):
        cs.germany_unit(tmp_path, cs.REFERENCE, ("daily",))
    assert events == []


def test_bdbv_origins_stop_before_the_first_scored_round(cs) -> None:
    # The paper keeps scored rounds (from 2026-11-21) out of fine-tuned
    # results, and drops any ebola row whose last origin is on or after it.
    assert cs.BDBV_SCORED_FROM == "2026-11-21"
    vintages = [
        "2026-10-09",
        "2026-11-15",
        "2026-11-21",
        "2026-11-28",
        "2026-12-05",
    ]
    got = cs.development_origins(vintages, skip=("2026-10-09",))
    assert got == ["2026-11-15"]


def test_only_the_final_checkpoint_of_the_live_run_is_marked_chosen(cs) -> None:
    # Sam: the reported model is grid r5's n8400 (#311), chosen by the
    # learning-rate decay to zero, never on German or BDBV scores.
    assert cs.FINAL_OUTBREAKS == 8400
    rows = pd.DataFrame(
        {
            "run": [
                "one-model",
                "one-model",
                "one-model",
                "no-slots",
                "zero-shot",
            ],
            "grid": ["r5", "r5", "r4", "r5", None],
            "outbreaks": [8400, 4000, 8400, 8400, 0],
        }
    )
    assert cs.chosen(rows).tolist() == [True, False, False, False, False]


def test_the_german_daily_target_is_refused_without_its_code(
    cs, tmp_path, monkeypatch
) -> None:
    from bdbvhubrenewal.evaluation import germany as de

    monkeypatch.delattr(de, "germany_daily_forecasts", raising=False)
    with pytest.raises(SystemExit, match="germany_daily_forecasts"):
        cs.germany_unit(tmp_path, cs.REFERENCE, ("7day", "daily"))


def test_the_bdbv_daily_target_is_refused_without_its_code(
    cs, tmp_path, monkeypatch
) -> None:
    from bdbvhubrenewal.evaluation import marginals

    monkeypatch.delattr(marginals, "daily_draws", raising=False)
    with pytest.raises(SystemExit, match="daily_draws"):
        cs.bdbv_unit(tmp_path, cs.REFERENCE, ("7day", "daily"))


def test_no_german_forecast_rows_without_its_script(cs, tmp_path) -> None:
    # scripts/germany_forecast.py is not on main; the table goes on.
    rows = cs.germany_forecast_rows(tmp_path, [cs.REFERENCE])
    assert rows.empty
    assert list(rows.columns) == cs.COLUMNS


def test_the_bdbv_tasks_carry_the_zones_as_a_submission_does(cs) -> None:
    # Sam (#314): Chronos-2 sees provinces, zones and national in its
    # block, so the tasks hold the zone tables and codes.
    from bdbvhubrenewal.evaluation import backtest as bt

    data = Path(__file__).parent / "fixtures" / "data"
    tasks = cs.bdbv_tasks(data, bt.load_config())
    task = next(iter(tasks.values()))
    assert task.zone_codes
    assert task.zone_daily_history is not None


def test_a_missing_score_table_starts_empty(cs, tmp_path) -> None:
    rows = cs.existing_rows(tmp_path / "none.csv", cs.COLUMNS)
    assert rows.empty
    assert list(rows.columns) == cs.COLUMNS


def test_the_bdbv_tasks_carry_zones_only_when_asked(cs) -> None:
    from bdbvhubrenewal.evaluation import backtest as bt

    data = Path(__file__).parent / "fixtures" / "data"
    plain = next(
        iter(cs.bdbv_tasks(data, bt.load_config(), zones=False).values())
    )
    assert not plain.zone_codes
    assert plain.zone_daily_history is None
