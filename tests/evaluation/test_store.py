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


def _copy_repo(tmp_path: Path) -> Path:
    for name in ("pyproject.toml", "uv.lock"):
        shutil.copy(ROOT / name, tmp_path / name)
    shutil.copytree(
        ROOT / "src",
        tmp_path / "src",
        ignore=shutil.ignore_patterns("__pycache__"),
    )
    return tmp_path


def test_code_digest_ignores_docstrings_and_comments(tmp_path: Path) -> None:
    root = _copy_repo(tmp_path)
    code = root / "src/bdbvhubrenewal/model/delays.py"
    text = code.read_text()
    before = store.code_digest("renewal", root)
    code.write_text(text + '\n\ndef _f():\n    """Doc."""\n    return 1\n')
    after = store.code_digest("renewal", root)
    code.write_text(text + "\n\ndef _f():\n    # A note.\n    return 1\n")
    assert store.code_digest("renewal", root) == after != before
    code.write_text(text + "\n\ndef _f():\n    return 2\n")
    assert store.code_digest("renewal", root) != after


def test_code_digest_follows_each_models_own_code(tmp_path: Path) -> None:
    root = _copy_repo(tmp_path)
    models = ("renewal", "renewalsp", "pprandomwalknowcast")

    def digests() -> dict[str, str]:
        return {m: store.code_digest(m, root) for m in models}

    def edit(rel: str) -> None:
        path = root / rel
        path.write_text(path.read_text() + "\n_EDITED = 1\n")

    src = "src/bdbvhubrenewal"
    before = digests()
    # The three renewal variants share a module, so share a key.
    assert before["renewal"] == before["renewalsp"]
    assert before["renewal"] != before["pprandomwalknowcast"]
    edit(f"{src}/forecasters/pprandomwalk.py")
    after = digests()
    assert after["renewal"] == before["renewal"]
    assert after["pprandomwalknowcast"] != before["pprandomwalknowcast"]
    before = after
    edit(f"{src}/model/observation.py")
    after = digests()
    assert after["renewal"] != before["renewal"]
    before = after
    # The CLI is not fit code.
    edit(f"{src}/cli.py")
    assert digests() == before
    # The store's own code refits every model.
    edit(f"{src}/evaluation/store.py")
    after = digests()
    assert all(after[m] != before[m] for m in models)


def test_fit_code_has_the_store_and_no_other_model() -> None:
    renewal = set(store.fit_code("renewal", ROOT))
    assert "src/bdbvhubrenewal/evaluation/store.py" in renewal
    assert "src/bdbvhubrenewal/model/observation.py" in renewal
    assert "src/bdbvhubrenewal/forecasters/pprandomwalk.py" not in renewal
    pp = set(store.fit_code("pprandomwalknowcast", ROOT))
    assert "src/bdbvhubrenewal/forecasters/renewal.py" not in pp
    assert "src/bdbvhubrenewal/evaluation/store.py" in pp


def test_samplers_come_from_the_policy() -> None:
    table = store.samplers(ROOT / "ci" / "policy.toml")
    assert set(table) == {"renewal", "renewalbvd", "renewalsp"}
    assert table["renewalbvd"]["num_samples"] == 750


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


@pytest.mark.usefixtures("in_root")
def test_the_submission_task_has_the_stored_key(
    target: pd.DataFrame, delays: pd.DataFrame, daily: pd.DataFrame
) -> None:
    """Same code and data give the submission the stored fit's key."""
    import argparse

    from bdbvhubrenewal import cli
    from bdbvhubrenewal.evaluation.backtest import load_config

    config = load_config("config/backtest.toml")
    tasks = store.origin_tasks(target, delays, config, daily)
    sampler = store.samplers()
    models = {m: store.build_model(m, sampler) for m in sampler}
    # Every real vintage, including 2026-08-18, which has no delays of
    # its own.
    for vintage in real_vintages(target):
        args = argparse.Namespace(
            vintage=vintage,
            reference_date=None,
            data=str(ROOT / "data"),
            backtest_config="config/backtest.toml",
        )
        task, _ = cli._task(args)
        stored = tasks[vintage]
        assert task.target_end_dates == stored.target_end_dates
        for name, model in models.items():
            assert store.fit_key(model, task, "c") == store.fit_key(
                model, tasks[vintage], "c"
            ), (name, vintage)


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
    assert set(dev["vintage"]) == {oldest, stale, broken, newest}
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
