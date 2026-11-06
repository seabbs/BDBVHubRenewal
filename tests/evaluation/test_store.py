from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from bdbvhubrenewal.evaluation import store
from bdbvhubrenewal.evaluation.backtest import BacktestConfig, real_vintages
from bdbvhubrenewal.forecasters import ForecastTask

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def fitted_flat(flat_model: str, monkeypatch: pytest.MonkeyPatch) -> str:
    """Register the test stand-in as a fitted model, the only one stored."""
    from bdbvhubrenewal.forecasters import base

    factory = base._REGISTRY[flat_model].factory

    def make() -> object:
        model = factory()
        model.name = "flatfit"
        return model

    monkeypatch.setattr(
        base,
        "_REGISTRY",
        {"flatfit": base.ModelSpec("flatfit", make, "Fitted.", fitted=True)},
    )
    return "flatfit"


@pytest.fixture
def tasks(
    target: pd.DataFrame,
    delays: pd.DataFrame,
    daily: pd.DataFrame,
    config: BacktestConfig,
) -> dict[str, ForecastTask]:
    return store.origin_tasks(target, delays, config, daily)


def test_code_digest_follows_the_inputs(tmp_path: Path) -> None:
    for name in store.STORE_INPUTS:
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        if "." in path.name:
            path.write_text("x")
        else:
            path.mkdir()
            (path / "a.py").write_text("x")
    before = store.code_digest(tmp_path)
    (tmp_path / "src/bdbvhubrenewal/model/__pycache__").mkdir()
    (tmp_path / "src/bdbvhubrenewal/model/__pycache__/a.pyc").write_text("y")
    assert store.code_digest(tmp_path) == before
    (tmp_path / "src/bdbvhubrenewal/model/a.py").write_text("y")
    assert store.code_digest(tmp_path) != before


def test_samplers_come_from_the_policy() -> None:
    table = store.samplers(ROOT / "ci" / "policy.toml")
    assert set(table) == {"renewal", "renewalbvd"}
    assert table["renewalbvd"]["num_samples"] == 1000


def test_a_new_release_keeps_older_keys(
    target: pd.DataFrame,
    delays: pd.DataFrame,
    daily: pd.DataFrame,
    config: BacktestConfig,
    tasks: dict[str, ForecastTask],
) -> None:
    latest = real_vintages(target)[-1]
    older = store.origin_tasks(
        target.loc[target["vintage"] != latest],
        delays.loc[delays["vintage"] != latest],
        config,
        daily.loc[daily["vintage"] != latest],
    )
    assert set(older) == set(tasks) - {latest}
    for origin, task in older.items():
        assert store.fit_key("m", task, "c", {}) == store.fit_key(
            "m", tasks[origin], "c", {}
        )
    task = tasks[latest]
    key = store.fit_key("m", task, "c", {"seed": 1})
    assert store.fit_key("m", task, "c", {"seed": 2}) != key
    assert store.fit_key("m", task, "d", {"seed": 1}) != key
    assert store.fit_key("n", task, "c", {"seed": 1}) != key


def test_plan_fit_and_load(
    fitted_flat: str,
    tasks: dict[str, ForecastTask],
    config: BacktestConfig,
    tmp_path: Path,
) -> None:
    config.model_choice.write_text(
        "choices:\n  - {date: 2026-10-10, model: flatfit}\n"
    )
    root = tmp_path / "store"
    models = store.stored_models()
    assert models == [fitted_flat]
    missing = store.plan(tasks, models, "c", {}, root)
    assert [m["origin"] for m in missing] == sorted(tasks, reverse=True)
    # The newest and oldest origins are fitted at the current key; one
    # origin only has a fit of older code, and one failed.
    origins = sorted(tasks)
    oldest, stale, broken, newest = (
        origins[0],
        origins[1],
        origins[2],
        origins[-1],
    )
    for origin in (newest, oldest):
        stored = store.fit(fitted_flat, tasks[origin], "c", {})
        assert stored.meta["status"] == "ok"
        path = store.write_fit(root, stored)
        back = store.read_fit(path)
        assert back.meta == stored.meta
        pd.testing.assert_frame_equal(
            back.forecasts, stored.forecasts, check_dtype=False
        )
    old = store.fit(fitted_flat, tasks[stale], "old", {})
    store.write_fit(root, old)
    store.write_fit(root, store.fit(fitted_flat, tasks[broken], "old", {}))
    store.write_fit(
        root,
        store.failed(fitted_flat, tasks[broken], "c", {}, RuntimeError("x")),
    )
    left = {m["origin"] for m in store.plan(tasks, models, "c", {}, root)}
    assert left == set(origins) - {newest, oldest, broken}

    results = tmp_path / "results"
    status = store.load(root, tasks, models, "c", {}, config, results)
    by_origin = {s.origin: s for s in status}
    assert by_origin[newest].current
    assert by_origin[oldest].current
    assert not by_origin[stale].current
    assert by_origin[stale].used == old.meta["key"]
    assert by_origin[broken].failed
    assert by_origin[broken].used is not None
    assert by_origin[origins[3]].used is None
    dev = pd.read_csv(results / "forecasts.csv.gz")
    held = pd.read_csv(results / "heldout-forecasts.csv.gz")
    assert set(dev["vintage"]) == {oldest, stale}
    assert newest in set(held["vintage"])
    assert set(held["chosen_model"]) == {fitted_flat}
    recorded = json.loads((results / store.STATUS_FILE).read_text())
    assert len(recorded) == len(tasks)
