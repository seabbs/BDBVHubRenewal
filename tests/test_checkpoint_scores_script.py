"""Tests of scripts/checkpoint_scores.py's pure parts."""

from __future__ import annotations

import importlib.util
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


def test_a_slots_cache_is_also_reachable_by_its_bare_hash(cs, tmp_path):
    # The dashboard mirrors cache/<study>/<hash>/ as dates finish.
    root = cs.unit_root(tmp_path, "germany", "slots-abcdef123456")
    assert root == tmp_path / "germany" / "slots-abcdef123456"
    assert (tmp_path / "germany" / "abcdef123456").resolve() == root.resolve()


def test_rows_are_built_only_from_finished_units(cs, tmp_path):
    (tmp_path / "heldout").mkdir()
    ready = cs.ready_labels(tmp_path, "germany", ["reference", "abc"])
    assert ready == []
