"""Tests of scripts/heldout_scores.py: proper scores on held-out sims."""

from __future__ import annotations

import datetime as dt
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from bdbvhubrenewal.evaluation.scoring import (
    QUANTILE_LEVELS,
    crps_sample,
    wis_quantile,
)

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"

LAYOUT_TARGETS = {
    ("daily", "daily"),
    ("weekly", "7day"),
    ("3day", "3day"),
    ("daily+weekly", "daily"),
    ("daily+weekly", "7day"),
    ("daily+3day", "daily"),
    ("daily+3day", "3day"),
}
"""Each layout's targets: its count rows, and a mixed layout's totals."""


def _load(name: str):
    sys.path.insert(0, str(SCRIPTS))
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
    assert spec is not None
    assert spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def hs():
    return _load("heldout_scores")


@pytest.fixture(scope="module")
def cs():
    return _load("checkpoint_scores")


def toy_sims(root: Path, seed: int = 5001, n: int = 2) -> list[str]:
    """Write `n` simulations of three places, two draws each, to `root`.

    Counts rise and fall over 200 days and are reported over up to 20
    days, as the simulator's `counts-<seed>.csv.gz` and
    `settings-<seed>.csv` hold them.
    """
    rng = np.random.default_rng(seed)
    first = dt.date(2026, 1, 1)
    days = np.arange(200)
    rows, ids = [], []
    for i in range(1, n + 1):
        sim = f"{seed}-{i:04d}"
        ids.append(sim)
        for draw in (1, 2):
            for p in range(1, 4):
                rate = 2 + 30 * p * np.exp(-(((days - 100) / 40) ** 2))
                for d, count in zip(days, rng.poisson(rate), strict=True):
                    delays = rng.geometric(0.3, int(count)) - 1
                    for lag, k in zip(
                        *np.unique(np.minimum(delays, 20), return_counts=True),
                        strict=True,
                    ):
                        rows.append(
                            {
                                "outbreak": f"{sim}-{draw}",
                                "timestep": "day",
                                "location": f"P{p:02d}",
                                "reference_date": first
                                + dt.timedelta(days=int(d)),
                                "report_date": first
                                + dt.timedelta(days=int(d + lag)),
                                "count": int(k),
                            }
                        )
    pd.DataFrame(rows).to_csv(root / f"counts-{seed}.csv.gz", index=False)
    pd.DataFrame(
        {
            "outbreak": ids,
            "n_places": 3,
            "n_ages": 1,
            "max_delay": 21,
            # Two of grid r5's factors, a level each simulation.
            "process": ["growth", "renewal_7"] * (n // 2)
            + ["growth"] * (n % 2),
            "shocks": "none",
        }
    ).to_csv(root / f"settings-{seed}.csv", index=False)
    return ids


@pytest.fixture(scope="module")
def sims(tmp_path_factory):
    root = tmp_path_factory.mktemp("sims")
    return root, toy_sims(root)


def _sha(ids: list[str]) -> str:
    return hashlib.sha256(("\n".join(ids) + "\n").encode()).hexdigest()


# The block cut ----------------------------------------------------------


def test_the_cut_is_the_trainer_s_and_deterministic(hs, sims) -> None:
    import finetune_stream as fs

    root, ids = sims
    blocks = hs.cut_blocks(root, ids, per_simulation=64)
    again = hs.cut_blocks(root, ids, per_simulation=64)
    # The trainer's own cut, with its settings for the held-out set.
    fs.init_worker(*hs.cut_settings())
    trainer = [b for _, b in fs.cut(root, ids, hs.CUT_SEED, 64)]
    assert len(blocks) == len(again) == len(trainer) > 0
    for a, b, c in zip(blocks, again, trainer, strict=True):
        np.testing.assert_array_equal(a["context"], c["context"])
        np.testing.assert_array_equal(a["context"], b["context"])
        assert a["source"] == c["source"]
        assert a["places"] == c["places"]
        # Each block knows its cut-off and horizon, for its windows and
        # for randomwalknowcast's task.
        assert a["meta"]["cutoffs"][-1] == b["meta"]["cutoffs"][-1]
        assert a["meta"]["horizon"] <= a["prediction_length"]


def test_every_layout_and_target_is_cut(hs, sims) -> None:
    root, ids = sims
    blocks = hs.cut_blocks(root, ids, per_simulation=160)
    cells = {(b["source"], t) for b in blocks for t in hs.row_targets(b)}
    assert cells == LAYOUT_TARGETS
    for b in blocks:
        targets = hs.row_targets(b)
        assert len(targets) == b["n_targets"]
        if "+" in b["source"]:
            # The count rows, then each place's period totals.
            half = b["n_targets"] // 2
            assert set(targets[:half]) == {"daily"}
            assert len(set(targets[half:])) == 1


def test_the_sample_is_fixed_and_the_same_for_every_label(
    hs, sims, tmp_path
) -> None:
    root, ids = sims
    sizes = dict.fromkeys(hs.LAYOUTS, 2)
    first = hs.prepare(
        tmp_path, root, ids, _sha(ids), sizes=sizes, per_simulation=64
    )
    again = hs.prepare(
        tmp_path, root, ids, _sha(ids), sizes=sizes, per_simulation=64
    )
    assert [b["index"] for b in first] == [b["index"] for b in again]
    assert {b["source"] for b in first} == set(hs.LAYOUTS)
    for layout in hs.LAYOUTS:
        assert sum(b["source"] == layout for b in first) == 2
    # Written once: one blocks file, keyed by the held-out set.
    assert len(list((tmp_path / "heldout_scores").glob("blocks-*.pkl"))) == 1
    # A held-out set that is not the trainer's is refused.
    with pytest.raises(ValueError, match="held-out"):
        hs.prepare(tmp_path / "x", root, ids, "0" * 64, sizes=sizes)


def test_windows_split_nowcast_from_forecast_at_the_cutoff(hs) -> None:
    cut = "2026-03-04"  # a Wednesday; its week ends Saturday 03-07
    end = dt.date(2026, 3, 7) + dt.timedelta(weeks=4)
    weekly = {"source": "weekly", "meta": {"cutoffs": [cut], "horizon": 6}}
    assert list(hs.windows(weekly)) == ["nowcast"] * 2 + ["forecast"] * 4
    assert hs.step_ends(weekly)[-1] == end
    daily = {
        "source": "daily+weekly",
        "meta": {"cutoffs": [cut], "horizon": 40},
    }
    w = hs.windows(daily)
    # Days up to the cut-off are nowcast: 40 days end 4 weeks after the
    # Saturday, so the cut-off is the 9th.
    assert list(w).count("nowcast") == 9
    assert hs.step_ends(daily)[8] == dt.date(2026, 3, 4)
    three = {"source": "3day", "meta": {"cutoffs": [cut], "horizon": 12}}
    ends = hs.step_ends(three)
    # Bins end on the cut-off and every 3 days from it.
    assert all((e - dt.date(2026, 3, 4)).days % 3 == 0 for e in ends)
    assert ends[-1] <= end
    # The bins ending 03-01 and on the cut-off hold days up to it.
    assert list(hs.windows(three)).count("nowcast") == 2


# Scores -----------------------------------------------------------------


def test_wis_comes_from_native_quantiles_read_at_the_hub_levels(hs) -> None:
    native = np.array([0.01, *np.round(np.arange(0.05, 0.951, 0.05), 2), 0.99])
    # A uniform(0, 100) forecast: its quantile function is linear, so
    # linear interpolation between native levels is exact.
    q = np.broadcast_to(100 * native, (2, 3, len(native))).copy()
    q[0, 0] = q[0, 0][::-1]  # crossed quantiles are sorted first
    truth = np.array([[10.0, 50.0, np.nan], [99.0, 0.0, 30.0]])
    got = hs.quantile_scores(truth, hs.at_levels(q, native))
    want = wis_quantile(
        truth.ravel(),
        np.broadcast_to(100 * QUANTILE_LEVELS, (6, len(QUANTILE_LEVELS))),
        QUANTILE_LEVELS,
    )["wis"].reshape(2, 3)
    np.testing.assert_allclose(got["wis"], want)
    assert np.isnan(got["wis"][0, 2])
    # Log scale: the same rule on log1p of the truth and the quantiles.
    log_want = wis_quantile(
        np.log1p(truth.ravel()),
        np.log1p(
            np.broadcast_to(100 * QUANTILE_LEVELS, (6, len(QUANTILE_LEVELS)))
        ),
        QUANTILE_LEVELS,
    )["wis"].reshape(2, 3)
    np.testing.assert_allclose(got["log_wis"], log_want)
    # Coverage of the central intervals.
    assert got["coverage_90"][1, 0] == 0.0  # 99 is above the 95% quantile
    assert got["coverage_50"][0, 1] == 1.0


def test_sample_scores_are_crps_and_wis_of_the_draws(hs) -> None:
    rng = np.random.default_rng(1)
    draws = rng.poisson(20, (1000, 2, 3)).astype(float)
    truth = np.array([[18.0, 25.0, 3.0], [20.0, 0.0, 40.0]])
    got = hs.sample_scores(truth, draws)
    for i in range(2):
        for j in range(3):
            assert got["crps"][i, j] == pytest.approx(
                crps_sample(truth[i, j], draws[:, i, j])
            )
            q = np.quantile(draws[:, i, j], QUANTILE_LEVELS)
            assert got["wis"][i, j] == pytest.approx(
                wis_quantile([truth[i, j]], q[None], QUANTILE_LEVELS)["wis"][0]
            )


def _scored(
    model: str, checkpoint: str, layout: str, target: str, value: float
) -> pd.DataFrame:
    """Two blocks' scores of one cell, `value` everywhere."""
    rows = []
    for block in (0, 1):
        for step, window in enumerate(("nowcast", "forecast")):
            rows.append(
                {
                    "block": block,
                    "layout": layout,
                    "target": target,
                    "level": "leaf" if block else "national",
                    "window": window,
                    "row": 0,
                    "step": step,
                    "truth": 1.0,
                    "model": model,
                    "checkpoint": checkpoint,
                    "wis": value,
                    "log_wis": value / 10,
                    "crps": np.nan,
                    "coverage_50": 1.0,
                    "coverage_90": 1.0,
                    "coverage_95": 1.0,
                }
            )
    return pd.DataFrame(rows)


def test_rows_are_relative_to_randomwalknowcast_where_it_is(hs, cs) -> None:
    scores = pd.concat(
        [
            _scored("fm_chronos", "zero-shot", "weekly", "7day", 4.0),
            _scored("fm_chronos", "abc", "weekly", "7day", 2.0),
            _scored("randomwalknowcast", "zero-shot", "weekly", "7day", 8.0),
            _scored("fm_chronos", "zero-shot", "daily", "daily", 5.0),
            _scored("fm_chronos", "abc", "daily", "daily", 4.0),
        ],
        ignore_index=True,
    )
    # randomwalknowcast missed a block: every model's rows of that cell
    # leave it out.
    scores = scores.loc[
        ~((scores["model"] == "randomwalknowcast") & (scores["block"] == 1))
    ]
    rows = hs.relative_rows(scores)
    assert set(cs.COLUMNS) >= {"layout", "target", "window", "rel_to"}
    assert set(rows.columns) <= set(cs.COLUMNS)
    assert (rows["study"] == "heldout").all()
    weekly = rows.loc[rows["layout"] == "weekly"]
    assert (weekly["rel_to"] == "randomwalknowcast").all()
    assert (weekly["n_blocks"] == 1).all()
    ft = weekly.loc[
        (weekly["checkpoint"] == "abc")
        & (weekly["level"] == "all")
        & (weekly["window"] == "all")
    ]
    assert ft["rel_wis"].item() == pytest.approx(0.25)
    daily = rows.loc[rows["layout"] == "daily"]
    assert (daily["rel_to"] == "zero-shot").all()
    one = daily.loc[
        (daily["checkpoint"] == "abc")
        & (daily["level"] == "all")
        & (daily["window"] == "all")
    ]
    assert one["rel_wis"].item() == pytest.approx(0.8)
    assert one["wis"].item() == pytest.approx(4.0)
    # By level and window too.
    assert set(rows["level"]) == {"all", "leaf", "national"}
    assert set(rows["window"]) == {"all", "nowcast", "forecast"}


def test_the_reference_is_randomwalknowcast_only_where_it_forecast(hs) -> None:
    assert hs.reference_of({("randomwalknowcast", "zero-shot")}) == (
        "randomwalknowcast",
        "zero-shot",
    )
    assert hs.reference_of({("fm_chronos", "zero-shot")}) == (
        "fm_chronos",
        "zero-shot",
    )
    # randomwalknowcast forecasts the weekly layout's 7-day target only.
    assert {("weekly", "7day")} == hs.BASELINE_CELLS


# The unit ---------------------------------------------------------------


def test_the_unit_scores_every_block_for_each_label_and_resumes(
    hs, sims, tmp_path
) -> None:
    root, ids = sims
    sizes = dict.fromkeys(hs.LAYOUTS, 2)
    blocks = hs.prepare(
        tmp_path, root, ids, _sha(ids), sizes=sizes, per_simulation=64
    )
    native = np.array([0.01, *np.round(np.arange(0.05, 0.951, 0.05), 2), 0.99])
    calls = []

    def chronos(block):
        calls.append(block["index"])
        n, h = block["n_targets"], block["meta"]["horizon"]
        return native, np.broadcast_to(10 * native, (n, h, len(native)))

    def walk(block):
        n, h = block["n_targets"], block["meta"]["horizon"]
        return np.full((100, n, h), 5.0)

    for label, models in (
        ("reference", {"fm_chronos": chronos, "randomwalknowcast": walk}),
        ("abc", {"fm_chronos": chronos}),
    ):
        hs.unit(tmp_path, label, models, blocks)
        assert hs.unit_done(tmp_path, label)
    assert not hs.unit_done(tmp_path, "def")
    a = hs.label_scores(tmp_path, "reference")
    b = hs.label_scores(tmp_path, "abc")
    keys = ["block", "row", "step"]
    same = a.loc[a["model"] == "fm_chronos", keys].reset_index(drop=True)
    pd.testing.assert_frame_equal(same, b[keys].reset_index(drop=True))
    # randomwalknowcast runs on the weekly layout only.
    walked = a.loc[a["model"] == "randomwalknowcast"]
    assert set(walked["layout"]) == {"weekly"}
    assert walked["crps"].notna().all()
    assert {(r.layout, r.target) for r in a.itertuples()} == LAYOUT_TARGETS
    # A rerun reads the cache and predicts nothing.
    calls.clear()
    hs.unit(tmp_path, "abc", {"fm_chronos": chronos}, blocks)
    assert calls == []


def test_checkpoint_scores_dispatch_the_unit(cs, tmp_path) -> None:
    assert not cs.unit_done(tmp_path, "heldout_scores", "reference")


# Grid r5's held-out round ----------------------------------------------


def test_the_r5_set_is_round_9_whole(hs) -> None:
    r5 = hs.SETS["r5"]
    assert r5.sim_dir == Path("~/bdbv-rounds/rounds/r9/sims")
    assert r5.seeds == (30900, 30939)
    assert r5.holdout == 840
    # The trainer monitors the first 90; every simulation is scored.
    assert r5.checked == 90
    assert r5.sizes is None
    assert r5.draws == 1
    # Its own cache folder and rows, so the r4 set's cache is untouched.
    r4 = hs.SETS["r4"]
    assert r4.name == "heldout_scores"
    assert r5.name != r4.name
    assert (r4.study, r5.study) == ("heldout", "heldout_r5")


def test_a_whole_set_keeps_every_block_and_checks_the_trainer_s_ids(
    hs, sims, tmp_path
) -> None:
    root, ids = sims
    held = hs.HeldSet(
        name="toy", sim_dir=root, seeds=(5001, 5001), holdout=len(ids),
        checked=1, draws=2, sizes=None, study="heldout_toy",
    )  # fmt: skip
    every = hs.cut_blocks(root, ids, per_simulation=64)
    got = hs.prepare(
        tmp_path, held=held, ids=ids, sha=_sha(ids[:1]), per_simulation=64
    )
    assert [b["index"] for b in got] == list(range(len(every)))
    assert (tmp_path / "toy" / "sample.json").exists()
    assert not (tmp_path / "heldout_scores").exists()
    with pytest.raises(ValueError, match="held-out"):
        hs.prepare(
            tmp_path / "x", held=held, ids=ids, sha=_sha(ids),
            per_simulation=64,
        )  # fmt: skip


def test_the_factors_are_the_simulator_s() -> None:
    import re

    hs = _load("heldout_scores")
    text = (SCRIPTS / "finetune_simulate.R").read_text()
    body = text[
        text.index("factors <- list(") : text.index(
            "\n)\n", text.index("factors <- list(")
        )
    ]
    names = re.findall(r"^  (\w+) = c\(", body, flags=re.M)
    assert tuple(names) == hs.FACTORS


def test_factor_rows_split_each_cell_by_the_simulations_levels(
    hs, sims, tmp_path
) -> None:
    root, ids = sims
    held = hs.HeldSet(
        name="toy", sim_dir=root, seeds=(5001, 5001), holdout=len(ids),
        checked=len(ids), draws=2, sizes=None, study="heldout_toy",
    )  # fmt: skip
    blocks = hs.prepare(
        tmp_path, held=held, ids=ids, sha=_sha(ids), per_simulation=16
    )
    native = np.array([0.01, *np.round(np.arange(0.05, 0.951, 0.05), 2), 0.99])

    def chronos(scale):
        def predict(block):
            n, h = block["n_targets"], block["meta"]["horizon"]
            q = np.broadcast_to(scale * native, (n, h, len(native)))
            return native, q

        return predict

    hs.unit(
        tmp_path, "reference", {"fm_chronos": chronos(10)}, blocks, held=held
    )
    hs.unit(tmp_path, "abc", {"fm_chronos": chronos(20)}, blocks, held=held)
    scored = hs.label_scores(tmp_path, "abc", held)["outbreak"]
    assert set(hs.simulation_of(scored)) == set(ids)
    factors = hs.factor_table(root, ids)
    assert list(factors.loc[ids[0]][["process", "shocks"]]) == [
        "growth",
        "none",
    ]
    rows = hs.rows(tmp_path, ["reference", "abc"], "reference", held)
    assert set(rows["study"]) == {"heldout_toy"}
    by = rows.loc[rows["factor"] == "process"]
    assert set(by["factor_level"]) == {"growth", "renewal_7"}
    # Factor rows are over every level and window of a (layout, target).
    assert set(by["level"]) == {"all"}
    assert set(by["window"]) == {"all"}
    whole = rows.loc[rows["factor"].isna()]
    assert not whole.empty
    # The levels' blocks add up to the cell's.
    cell = by.loc[(by["layout"] == "weekly") & (by["checkpoint"] == "abc")]
    all_ = whole.loc[
        (whole["layout"] == "weekly") & (whole["checkpoint"] == "abc")
        & (whole["level"] == "all") & (whole["window"] == "all")
    ]  # fmt: skip
    assert cell["n"].sum() == all_["n"].sum()
    base = rows.loc[rows["checkpoint"] == "zero-shot", "rel_wis"]
    assert np.allclose(base, 1.0)


def _settings(cache: Path, label: str, **kw) -> None:
    root = cache / "heldout"
    root.mkdir(parents=True, exist_ok=True)
    base = {
        "grid": "r5",
        "held_out_sha256": "a" * 64,
        "memory_gb": 36.0,
        "seed": 1,
        "blocks_per_simulation": 32,
        "draws_per_simulation": 1,
    }
    (root / f"{label}.json").write_text(json.dumps({**base, **kw}))
    (root / f"{label}-release.json").write_text("{}")


def test_the_r5_trainer_hash_is_read_from_the_r5_releases(cs, tmp_path) -> None:
    # The old run's settings and release files sit beside them.
    _settings(tmp_path, "slots-old", grid="r4", held_out_sha256="b" * 64)
    with pytest.raises(SystemExit, match="no r5 checkpoint"):
        cs.trainer_sha(tmp_path, "r5")
    _settings(tmp_path, "slots-n250")
    _settings(tmp_path, "slots-n500")
    assert cs.trainer_sha(tmp_path, "r5") == "a" * 64
    assert cs.trainer_sha(tmp_path, "r4") == _load("heldout_scores").HELD_SHA256
    # Checkpoints that disagree on the held-out set, or were cut another
    # way, stop the unit rather than score a different set.
    _settings(tmp_path, "slots-n750", held_out_sha256="c" * 64)
    with pytest.raises(SystemExit, match="held-out"):
        cs.trainer_sha(tmp_path, "r5")
    (tmp_path / "heldout" / "slots-n750.json").unlink()
    _settings(tmp_path, "slots-n750", memory_gb=64.0)
    with pytest.raises(SystemExit, match="memory_gb"):
        cs.trainer_sha(tmp_path, "r5")


def test_each_checkpoint_s_held_out_rows_come_from_its_grid_s_set(cs) -> None:
    assert cs.held_set_of({"grid": "r5"}) == "r5"
    assert cs.held_set_of({"grid": "r4"}) == "r4"
    assert cs.held_set_of({}) == "r4"


def test_the_r5_unit_is_its_own_study(cs, tmp_path) -> None:
    assert not cs.unit_done(tmp_path, "heldout_scores_r5", "reference")
    hs = _load("heldout_scores")
    root = tmp_path / "heldout_scores_r5"
    root.mkdir()
    (root / "sample.json").write_text(json.dumps({"key": "k", "n_blocks": 1}))
    (root / "reference").mkdir()
    (root / "reference" / "k-000.csv.gz").write_bytes(b"")
    assert cs.unit_done(tmp_path, "heldout_scores_r5", "reference")
    # The r4 set's unit is not done by the r5 set's files.
    assert not cs.unit_done(tmp_path, "heldout_scores", "reference")
    assert hs.SETS["r5"].name == "heldout_scores_r5"
