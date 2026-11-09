from __future__ import annotations

import json
import shutil
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


def test_code_digest_ignores_docstrings_and_comments(tmp_path: Path) -> None:
    for name in ("pyproject.toml", "uv.lock"):
        shutil.copy(ROOT / name, tmp_path / name)
    code = tmp_path / "src/bdbvhubrenewal/model/a.py"
    code.parent.mkdir(parents=True)
    code.write_text('"""Doc."""\n\n\ndef f():\n    """Doc."""\n    return 1\n')
    before = store.code_digest(tmp_path)
    code.write_text('"""Other."""\n\n\ndef f():\n    # A note.\n    return 1\n')
    (code.parent / "__pycache__").mkdir()
    (code.parent / "__pycache__" / "a.pyc").write_text("y")
    assert store.code_digest(tmp_path) == before
    code.write_text("def f():\n    return 2\n")
    assert store.code_digest(tmp_path) != before


def test_samplers_come_from_the_policy() -> None:
    table = store.samplers(ROOT / "ci" / "policy.toml")
    assert set(table) == {"renewal", "renewalbvd", "renewalsp"}
    assert table["renewalbvd"]["num_samples"] == 1000


@pytest.mark.usefixtures("in_root")
def test_keys_follow_the_task_settings_and_bvd_release(
    target: pd.DataFrame,
    delays: pd.DataFrame,
    daily: pd.DataFrame,
    config: BacktestConfig,
    tasks: dict[str, ForecastTask],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sampler = store.samplers()
    models = ["renewal", "renewalbvd"]
    keys = store.current_keys(tasks, models, sampler, "c")
    # A new release leaves the older origins' keys alone.
    latest = real_vintages(target)[-1]
    older = store.origin_tasks(
        target.loc[target["vintage"] != latest],
        delays.loc[delays["vintage"] != latest],
        config,
        daily.loc[daily["vintage"] != latest],
    )
    assert store.current_keys(older, models, sampler, "c") == {
        k: v for k, v in keys.items() if k[1] != latest
    }
    # Code, sampler settings and the BVD release each change the key.
    assert store.current_keys(tasks, models, sampler, "d") != keys
    faster = {**sampler, "renewal": {**sampler["renewal"], "num_samples": 9}}
    changed = store.current_keys(tasks, models, faster, "c")
    assert changed[("renewal", latest)] != keys[("renewal", latest)]
    assert changed[("renewalbvd", latest)] == keys[("renewalbvd", latest)]
    monkeypatch.setattr(
        "bdbvhubrenewal.forecasters.renewal.bvd_record",
        lambda _: {"tag": "other"},
    )
    moved = store.current_keys(tasks, models, sampler, "c")
    assert moved[("renewal", latest)] == keys[("renewal", latest)]
    assert moved[("renewalbvd", latest)] != keys[("renewalbvd", latest)]


@pytest.fixture
def in_root(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(ROOT)


def test_plan_fit_and_load(
    fitted_flat: str,
    tasks: dict[str, ForecastTask],
    tmp_path: Path,
) -> None:
    root = tmp_path / "store"
    keys = store.current_keys(tasks, [fitted_flat], {}, "c")
    to_fit = store.plan(keys, root)
    assert [m["origin"] for m in to_fit] == sorted(tasks, reverse=True)
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
        stored = store.fit(fitted_flat, tasks[origin], {}, "c")
        assert stored.meta["status"] == "ok"
        assert stored.meta["key"] == keys[(fitted_flat, origin)]
        path = store.write_fit(root, stored)
        back = store.read_fit(path)
        assert back.meta == stored.meta
        pd.testing.assert_frame_equal(
            back.forecasts, stored.forecasts, check_dtype=False
        )
    old = store.fit(fitted_flat, tasks[stale], {}, "old")
    store.write_fit(root, old)
    store.write_fit(root, store.fit(fitted_flat, tasks[broken], {}, "old"))
    meta = {**old.meta, "origin": broken, "key": keys[(fitted_flat, broken)]}
    store.write_fit(root, store.failed(meta, RuntimeError("x")))
    left = {m["origin"] for m in store.plan(keys, root)}
    assert left == set(origins) - {newest, oldest, broken}
    again = {m["origin"] for m in store.plan(keys, root, retry_failed=True)}
    assert again == left | {broken}

    results = tmp_path / "results"
    status = store.load(root, keys, results)
    by_origin = {s.origin: s for s in status}
    assert by_origin[newest].current
    assert by_origin[oldest].current
    assert not by_origin[stale].current
    assert by_origin[stale].used == old.meta["key"]
    assert by_origin[broken].failed
    assert by_origin[broken].used is not None
    assert by_origin[origins[3]].used is None
    dev = pd.read_csv(results / "forecasts.csv.gz")
    # Every loaded origin is backtested.
    assert set(dev["vintage"]) == {oldest, stale, newest}
    recorded = json.loads((results / store.STATUS_FILE).read_text())
    assert len(recorded) == len(tasks)


def test_a_fit_that_raises_is_stored_as_failed(
    fitted_flat: str,
    tasks: dict[str, ForecastTask],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def boom(*_: object, **__: object) -> None:
        raise RuntimeError("no fit")

    monkeypatch.setattr(store, "run_forecasts", boom)
    stored = store.fit(fitted_flat, tasks[sorted(tasks)[-1]], {}, "c")
    assert stored.meta["status"] == "failed"
    assert "no fit" in stored.meta["error"]
    assert stored.forecasts.empty
