from __future__ import annotations

import dataclasses
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


def test_the_zone_split_does_not_key_a_fit(tmp_path: Path) -> None:
    # A stored fit holds national and province samples only. Zones are
    # split from them at load (`zones_for_model`), so the zone code is not
    # fit code.
    root = _copy_repo(tmp_path)
    models = ("renewal", "pprandomwalknowcast")

    def digests() -> dict[str, str]:
        return {m: store.code_digest(m, root) for m in models}

    def edit(rel: str) -> None:
        path = root / "src/bdbvhubrenewal" / rel
        path.write_text(path.read_text() + "\n_EDITED = 1\n")

    before = digests()
    for rel in ("forecasters/zones.py", "evaluation/zones.py"):
        edit(rel)
        assert digests() == before, rel
    # Introductions are drawn in the fit, for provinces with no recent case.
    edit("forecasters/introductions.py")
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


def test_fit_time_data_files_key_a_fit(tmp_path: Path) -> None:
    # The gravity kernel reads the province centres at fit time.
    root = _copy_repo(tmp_path)
    before = store.code_digest("renewal", root)
    path = root / "src/bdbvhubrenewal/data/province_centres.csv"
    path.write_text(path.read_text() + "CD-XX,0.0,0.0\n")
    assert store.code_digest("renewal", root) != before


def test_report_code_does_not_key_a_fit(tmp_path: Path) -> None:
    root = _copy_repo(tmp_path)
    models = ("renewal", "pprandomwalknowcast")

    def digests() -> dict[str, str]:
        return {m: store.code_digest(m, root) for m in models}

    before = digests()
    for rel in ("evaluation/digest.py", "reporting/plots.py"):
        path = root / "src/bdbvhubrenewal" / rel
        path.write_text(path.read_text() + "\n_EDITED = 1\n")
        assert digests() == before, rel


def _workflow_paths(event: str) -> list[str]:
    import yaml

    workflow = yaml.safe_load(
        (ROOT / ".github/workflows/fit-store.yml").read_text()
    )
    # PyYAML reads the `on` key as True.
    return list(workflow[True][event]["paths"])


@pytest.mark.parametrize("event", ["push", "pull_request"])
def test_fit_store_runs_on_every_file_in_a_key(event: str) -> None:
    import fnmatch

    patterns = _workflow_paths(event)
    files = {f for m in store.stored_models() for f in store.fit_code(m, ROOT)}
    missed = sorted(
        f
        for f in files
        if not any(fnmatch.fnmatch(f, p.replace("**", "*")) for p in patterns)
    )
    assert not missed, f"fit-store.yml {event} paths miss {missed}"


def test_samplers_refuse_two_candidates_for_one_model(
    tmp_path: Path,
) -> None:
    policy = tmp_path / "policy.toml"
    policy.write_text(
        "\n".join(
            f'[[candidates]]\nid = "{i}"\nmodel = "renewal"\n'
            f"[candidates.sampler]\nnum_samples = {n}\n"
            for i, n in (("a", 10), ("b", 20))
        )
    )
    with pytest.raises(ValueError, match="renewal"):
        store.samplers(policy)


def test_renewal_key_has_every_model_module() -> None:
    # `fit_code` does not follow a package's `__init__` imports.
    renewal = set(store.fit_code("renewal", ROOT))
    model = {
        p.relative_to(ROOT).as_posix()
        for p in (ROOT / "src/bdbvhubrenewal/model").glob("*.py")
    }
    assert model <= renewal, sorted(model - renewal)


def test_samplers_come_from_the_policy() -> None:
    table = store.samplers(ROOT / "ci" / "policy.toml")
    assert set(table) == {"renewal", "renewalbvd", "renewalsp"}
    assert table["renewalbvd"]["num_samples"] == 750


@pytest.fixture
def offline_bvd(monkeypatch: pytest.MonkeyPatch) -> None:
    """Choose no BVDOutbreakSize release, rather than list them online.

    The fixture data hold no release cache, so the choice would list the
    releases from GitHub.
    """
    monkeypatch.setattr(
        "bdbvhubrenewal.forecasters.renewal.bvd_release", lambda *_: None
    )


@pytest.mark.usefixtures("in_root", "offline_bvd")
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


@pytest.mark.usefixtures("in_root")
def test_skipped_origins_leave_the_other_keys_alone(
    target: pd.DataFrame,
    delays: pd.DataFrame,
    daily: pd.DataFrame,
    config: BacktestConfig,
    tasks: dict[str, ForecastTask],
) -> None:
    sampler = store.samplers()
    models = ["renewal"]
    keys = store.current_keys(tasks, models, sampler, "c")
    skip = dataclasses.replace(config, skip_origins=("2026-08-18",))
    kept = store.origin_tasks(target, delays, skip, daily)
    assert "2026-08-18" not in kept
    assert store.current_keys(kept, models, sampler, "c") == {
        k: v for k, v in keys.items() if k[1] != "2026-08-18"
    }


@pytest.fixture
def in_root(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(ROOT)


@pytest.mark.usefixtures("in_root", "offline_bvd")
def test_the_stored_task_covers_the_submissions_rounds(
    target: pd.DataFrame,
    delays: pd.DataFrame,
    daily: pd.DataFrame,
    tasks_json: Path,
) -> None:
    """The submission forecasts from the stored fit at its vintage.

    Its task, for the vintage's own round or the next one (a round with
    no new release), is the stored task but for the weeks, which the
    stored task covers. So the stored fit of the same code and data is
    the submission's fit.
    """
    import argparse
    import dataclasses
    import datetime as dt

    from bdbvhubrenewal import cli
    from bdbvhubrenewal.evaluation.backtest import load_config

    config = load_config("config/backtest.toml")
    tasks = store.origin_tasks(target, delays, config, daily)
    sampler = store.samplers()
    models = {m: store.build_model(m, sampler) for m in sampler}
    assert tasks
    for vintage, stored in tasks.items():
        for later in (0, 1):
            ref = stored.reference_date + dt.timedelta(weeks=later)
            args = argparse.Namespace(
                vintage=vintage,
                reference_date=ref.isoformat(),
                data=str(tasks_json.parents[1]),
                backtest_config="config/backtest.toml",
            )
            task, _ = cli._task(args)
            assert set(task.target_end_dates) <= set(stored.target_end_dates)
            same = dataclasses.replace(
                task,
                reference_date=stored.reference_date,
                target_end_dates=stored.target_end_dates,
            )
            for name, model in models.items():
                assert store.fit_key(model, same, "c") == store.fit_key(
                    model, stored, "c"
                ), (name, vintage, later)


def test_plan_fit_and_load(
    fitted_flat: str,
    tasks: dict[str, ForecastTask],
    config: BacktestConfig,
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
    status = store.load(
        root, keys, results, config.min_horizon, config.max_horizon
    )
    by_origin = {s.origin: s for s in status}
    assert by_origin[newest].current
    assert by_origin[oldest].current
    assert not by_origin[stale].current
    assert by_origin[stale].used == old.meta["key"]
    assert by_origin[broken].failed
    assert by_origin[broken].used is not None
    assert by_origin[origins[3]].used is None
    dev = pd.read_csv(results / "forecasts.csv.gz")
    # Every loaded origin is backtested, from the backtest's first week.
    assert set(dev["vintage"]) == {oldest, stale, broken, newest}
    assert stored.forecasts["horizon"].min() < config.min_horizon
    assert stored.forecasts["horizon"].max() > config.max_horizon
    assert dev["horizon"].min() == config.min_horizon
    assert dev["horizon"].max() == config.max_horizon
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


@pytest.fixture
def candidate_flat(fitted_flat: str, monkeypatch: pytest.MonkeyPatch) -> str:
    """Make the stand-in a submission candidate, config `flatcfg`."""
    from bdbvhubrenewal.forecasters import base, renewal
    from bdbvhubrenewal.submission import fitting

    spec = base._REGISTRY[fitted_flat]
    make = spec.factory

    def write_fit_dir(
        task: ForecastTask, fit_dir: Path, config_name: str
    ) -> dict[str, object]:
        del task
        (Path(fit_dir) / "samples.csv.gz").write_bytes(b"draws")
        return {"sampler": "nuts", "config": config_name, "worst_rhat": 1.0}

    def candidate() -> object:
        model = make()
        model.write_fit_dir = write_fit_dir
        return model

    monkeypatch.setitem(
        base._REGISTRY,
        fitted_flat,
        base.ModelSpec(fitted_flat, candidate, "Fitted.", fitted=True),
    )
    monkeypatch.setitem(renewal.FIT_VARIANTS, "flatcfg", fitted_flat)
    monkeypatch.setitem(
        fitting._CONFIGS,
        "flatcfg",
        fitting.FitConfig(
            "flatcfg", "Stand-in.", lambda *_: {}, lambda *_: None
        ),
    )
    return fitted_flat


def test_a_candidate_fit_carries_what_the_submission_reads(
    candidate_flat: str,
    tasks: dict[str, ForecastTask],
    tmp_path: Path,
) -> None:
    from bdbvhubrenewal.submission.fitting import read_fit as read_record

    root = tmp_path / "store"
    keys = store.current_keys(tasks, [candidate_flat], {}, "c")
    newest, older, broken = (
        sorted(tasks)[-1],
        sorted(tasks)[-2],
        sorted(tasks)[0],
    )
    stored = store.fit(candidate_flat, tasks[newest], {}, "c")
    assert set(stored.files) == {
        "diagnostics.json",
        "fit.json",
        "samples.csv.gz",
    }
    store.write_fit(root, stored)
    meta = {
        **stored.meta,
        "origin": broken,
        "key": keys[(candidate_flat, broken)],
    }
    store.write_fit(root, store.failed(meta, RuntimeError("x")))
    (root / ".commit").write_text("abc123\n")

    out = tmp_path / "fit"
    state, found = store.unpack(root, keys, candidate_flat, newest, out)
    assert state == "ok"
    assert found is not None
    assert found.meta["key"] == keys[(candidate_flat, newest)]
    assert (out / "samples.csv.gz").read_bytes() == b"draws"
    record = read_record(out)
    assert record["config"] == "flatcfg"
    assert record["vintage"] == newest
    diag = json.loads((out / "diagnostics.json").read_text())
    assert diag["worst_rhat"] == 1.0
    where = json.loads((out / store.UNPACKED_FILE).read_text())
    assert where["key"] == keys[(candidate_flat, newest)]
    assert where["store_commit"] == "abc123"
    assert where["commit"] == stored.meta["commit"]
    # No fit at the current key, or one that failed.
    assert store.unpack(root, keys, candidate_flat, older, tmp_path / "a")[
        0
    ] == ("missing")
    assert store.unpack(root, keys, candidate_flat, broken, tmp_path / "b")[
        0
    ] == ("failed")


def test_a_model_with_no_config_stores_no_files(
    fitted_flat: str, tasks: dict[str, ForecastTask]
) -> None:
    stored = store.fit(fitted_flat, tasks[sorted(tasks)[-1]], {}, "c")
    assert stored.files == {}


def test_the_archive_holds_every_models_forecast_for_a_round(
    fitted_flat: str,
    tasks: dict[str, ForecastTask],
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from bdbvhubrenewal.evaluation import store_archive
    from bdbvhubrenewal.forecasters import base

    # `plain` is a model the store does not hold, as `randomwalk` is.
    factory = base._REGISTRY[fitted_flat].factory

    def plain() -> object:
        model = factory()
        model.name = "plain"
        return model

    monkeypatch.setitem(
        base._REGISTRY, "plain", base.ModelSpec("plain", plain, "Not fitted.")
    )
    root = tmp_path / "store"
    models = [fitted_flat, "plain"]
    code = dict.fromkeys(models, "c")
    current, stale, broken = sorted(tasks)[-3:]
    round_tasks = {o: tasks[o] for o in (current, stale, broken)}
    keys = store.current_keys(round_tasks, models, {}, code)
    now = store.fit(fitted_flat, tasks[current], {}, "c")
    store.write_fit(root, now)
    old = store.fit(fitted_flat, tasks[stale], {}, "old")
    store.write_fit(root, old)
    meta = {**old.meta, "origin": broken, "key": keys[(fitted_flat, broken)]}
    store.write_fit(root, store.failed(meta, RuntimeError("x")))

    table = store_archive.archive(root, round_tasks, models, {}, code)
    assert list(table.columns) == list(store_archive.COLUMNS)
    fits = table.drop_duplicates(["model", "vintage"]).set_index(
        ["model", "vintage"]
    )
    # The stored fits as the store holds them; a failed fit has no rows.
    assert set(fits.loc[fitted_flat].index) == {current, stale}
    got = fits.loc[(fitted_flat, current)]
    assert got["key"] == now.meta["key"]
    assert got["fitted_at"] == now.meta["fitted_at"]
    assert bool(got["current"])
    got = fits.loc[(fitted_flat, stale)]
    assert got["key"] == old.meta["key"]
    assert not bool(got["current"])
    rows = table.loc[
        (table["model"] == fitted_flat) & (table["vintage"] == current),
        now.forecasts.columns,
    ]
    pd.testing.assert_frame_equal(
        rows.reset_index(drop=True), now.forecasts, check_dtype=False
    )
    # The model the store does not hold is forecast at every origin, at
    # its current key, and nothing is written to the store.
    assert set(fits.loc["plain"].index) == set(round_tasks)
    assert fits.loc["plain", "current"].all()
    assert fits.loc[("plain", current), "key"] == keys[("plain", current)]
    assert not (root / "plain").exists()
    # With nothing to archive, the table still has its columns.
    empty = store_archive.archive(root, round_tasks, [], {}, {})
    assert empty.empty
    assert list(empty.columns) == list(store_archive.COLUMNS)


def test_the_candidate_models_are_the_stored_and_the_policy_models() -> None:
    from bdbvhubrenewal.evaluation import store_archive

    models = store_archive.candidate_models(ROOT / "ci/policy.toml")
    assert "randomwalk" in models
    assert "randomwalknowcast" in models
    assert set(store.stored_models()) <= set(models)
    assert len(models) == len(set(models))


def test_a_model_that_cannot_be_built_leaves_the_others_archived(
    fitted_flat: str,
    tasks: dict[str, ForecastTask],
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from bdbvhubrenewal.evaluation import store_archive
    from bdbvhubrenewal.forecasters import base

    # As a fine-tuned model whose checkpoint's release cannot be read.
    def broken() -> object:
        raise ConnectionError("no release")

    monkeypatch.setitem(
        base._REGISTRY, "broken", base.ModelSpec("broken", broken, "Broken.")
    )
    origin = sorted(tasks)[-1]
    round_tasks = {origin: tasks[origin]}
    root = tmp_path / "store"
    store.write_fit(root, store.fit(fitted_flat, tasks[origin], {}, "c"))
    models = ["broken", fitted_flat]
    code = dict.fromkeys(models, "c")
    table = store_archive.archive(root, round_tasks, models, {}, code)
    assert set(table["model"]) == {fitted_flat}
