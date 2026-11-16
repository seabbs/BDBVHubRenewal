from __future__ import annotations

import base64
import dataclasses
import json
import shutil
import tomllib
from pathlib import Path

import pandas as pd
import pytest

from bdbvhubrenewal.evaluation import store
from bdbvhubrenewal.evaluation.backtest import BacktestConfig, real_vintages
from bdbvhubrenewal.forecasters import ForecastTask
from bdbvhubrenewal.forecasters.base import NATIONAL, SAMPLE_COLUMNS
from bdbvhubrenewal.submission.fitting import Forecast

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
    models = ("renewal", "renewalbvd", "renewalsp", "pprandomwalknowcast")

    def digests() -> dict[str, str]:
        return {m: store.code_digest(m, root) for m in models}

    def edit(rel: str) -> None:
        path = root / rel
        path.write_text(path.read_text() + "\n_EDITED = 1\n")

    src = "src/bdbvhubrenewal"
    before = digests()
    # The zone variants share a module, so share a key, and renewalsp,
    # in the province module they build on, has its own.
    assert before["renewal"] == before["renewalbvd"]
    assert before["renewal"] != before["renewalsp"]
    assert before["renewal"] != before["pprandomwalknowcast"]
    edit(f"{src}/forecasters/pprandomwalk.py")
    after = digests()
    assert after["renewal"] == before["renewal"]
    assert after["pprandomwalknowcast"] != before["pprandomwalknowcast"]
    before = after
    edit(f"{src}/forecasters/renewal.py")
    after = digests()
    assert after["renewal"] != before["renewal"]
    assert after["renewalsp"] != before["renewalsp"]
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


def test_zone_code_keys_only_the_zone_fits(tmp_path: Path) -> None:
    # The zone backtest is not fit code. The zone split module is, for
    # the zone fits of renewal and renewalbvd, which call its zone_task.
    root = _copy_repo(tmp_path)
    models = ("renewal", "pprandomwalknowcast")

    def digests() -> dict[str, str]:
        return {m: store.code_digest(m, root) for m in models}

    def edit(rel: str) -> None:
        path = root / "src/bdbvhubrenewal" / rel
        path.write_text(path.read_text() + "\n_EDITED = 1\n")

    before = digests()
    edit("evaluation/zones.py")
    assert digests() == before
    edit("forecasters/zones.py")
    after = digests()
    assert after["renewal"] != before["renewal"]
    assert after["pprandomwalknowcast"] == before["pprandomwalknowcast"]
    before = after
    # Introductions are drawn in the fit, for provinces with no recent case.
    edit("forecasters/introductions.py")
    after = digests()
    assert all(after[m] != before[m] for m in models)


def test_zone_data_keys_the_zone_fits(
    tasks: dict[str, ForecastTask], monkeypatch: pytest.MonkeyPatch
) -> None:
    from bdbvhubrenewal.forecasters import zonerenewal as zr

    sampler = store.samplers()
    models = ["renewal", "renewalsp"]
    keys = store.current_keys(tasks, models, sampler, "c")
    daily = zr._daily_zones()
    monkeypatch.setattr(
        zr, "_daily_zones", lambda: daily.assign(observed=daily["observed"] + 1)
    )
    moved = store.current_keys(tasks, models, sampler, "c")
    for (model, origin), key in keys.items():
        assert (moved[(model, origin)] != key) == (model == "renewal")


def test_fit_time_data_files_key_a_fit(tmp_path: Path) -> None:
    # The gravity kernel reads the province centres at fit time.
    root = _copy_repo(tmp_path)
    before = store.code_digest("renewal", root)
    path = root / "src/bdbvhubrenewal/data/province_centres.csv"
    path.write_text(path.read_text() + "CD-XX,0.0,0.0\n")
    assert store.code_digest("renewal", root) != before


def test_report_code_does_not_key_a_fit(tmp_path: Path) -> None:
    # `fit_code` runs `digest.model_code`, so a change to its import
    # closure can move a key. Its own source is not hashed (`STORE_SKIP`).
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


def _key_files(monkeypatch: pytest.MonkeyPatch) -> set[str]:
    """List the repository files a fit key depends on."""
    import inspect
    from collections.abc import Callable

    from bdbvhubrenewal import cli
    from bdbvhubrenewal.evaluation import digest

    read: set[Path] = set()
    open_ = Path.open
    getsource = inspect.getsource

    def record_open(self: Path, *args: object, **kwargs: object) -> object:
        read.add(self.resolve())
        return open_(self, *args, **kwargs)

    def record_source(obj: Callable[..., object]) -> str:
        src = inspect.getsourcefile(obj)
        assert src is not None
        read.add(Path(src).resolve())
        return getsource(obj)

    with monkeypatch.context() as mp:
        mp.setattr(Path, "open", record_open)
        mp.setattr(store.inspect, "getsource", record_source)
        for model in store.stored_models():
            store.code_digest(model, ROOT)
    # The key hashes the Python version. `fit-store plan` reads the config
    # and policy by default, and the CLI picks the BVD release a model is
    # keyed on. `digest.py` decides what `fit_code` lists.
    plan = cli.build_parser().parse_args(["fit-store", "plan"])
    read |= {
        (ROOT / f).resolve()
        for f in (plan.config, plan.policy, ".python-version")
    }
    for module in (cli, digest):
        src = inspect.getsourcefile(module)
        assert src is not None
        read.add(Path(src).resolve())
    root = ROOT.resolve()
    return {
        p.relative_to(root).as_posix() if p.is_relative_to(root) else str(p)
        for p in read
    }


@pytest.mark.parametrize("event", ["push", "pull_request"])
def test_fit_store_runs_on_every_file_in_a_key(
    event: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    import fnmatch

    patterns = _workflow_paths(event)
    files = _key_files(monkeypatch)
    # Fails if the spies stop recording reads.
    required = {
        "pyproject.toml",
        "uv.lock",
        "src/bdbvhubrenewal/evaluation/backtest.py",
    }
    assert required <= files
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
    assert table["renewalbvd"]["num_samples"] == 1500
    assert table["renewalbvd"]["thinning"] == 2


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


def test_load_writes_the_zone_fits_zone_samples(tmp_path: Path) -> None:
    import base64
    import gzip

    zones = pd.DataFrame(
        {
            "location": ["CD-IT-01", "CD-IT-01"],
            "target_end_date": ["2026-10-24", "2026-10-24"],
            "sample": [1, 1],
            "value": [3.0, 1.0],
            "as_of": [None, "2026-10-24"],
        }
    )
    body = base64.b64encode(
        gzip.compress(zones.to_csv(index=False).encode())
    ).decode()
    meta = {
        "model": "renewal",
        "origin": "2026-10-17",
        "key": "k",
        "commit": "c",
        "fitted_at": "2026-11-13T00:00:00+00:00",
        "settings": "",
        "status": "ok",
    }
    empty = pd.DataFrame()
    fit = store.StoredFit(
        meta, empty, empty, empty, files={"zones.csv.gz": body}
    )
    store.write_fit(tmp_path / "store", fit)
    # A fit with no zone samples adds no rows.
    plain = store.StoredFit(
        {**meta, "model": "randomwalknowcast", "key": "p"}, empty, empty, empty
    )
    store.write_fit(tmp_path / "store", plain)
    store.load(
        tmp_path / "store",
        {
            ("renewal", "2026-10-17"): "k",
            ("randomwalknowcast", "2026-10-17"): "p",
        },
        tmp_path / "res",
    )
    written = pd.read_csv(tmp_path / "res" / store.ZONE_FITS_FILE)
    # Final counts only, by model and origin.
    assert written[
        ["model", "vintage", "location", "value"]
    ].to_numpy().tolist() == [["renewal", "2026-10-17", "CD-IT-01", 3.0]]


class _Hub:
    """A stand-in for `cli._hub_forecast`, recording each call."""

    def __init__(self, fail: str | None = None) -> None:
        self.calls: dict[str, dict] = {}
        self.fail = fail

    def __call__(
        self,
        model: str,
        *,
        config: str | None = None,
        fit_dir: Path | None = None,
        forecast: Forecast | None = None,
    ) -> pd.DataFrame:
        from bdbvhubrenewal.submission.hub import HUB_COLUMNS

        if model == self.fail:
            raise ValueError("no forecast")
        files = sorted(p.name for p in fit_dir.iterdir()) if fit_dir else []
        self.calls[model] = {
            "config": config,
            "files": files,
            "forecast": forecast,
        }
        row = dict.fromkeys(HUB_COLUMNS, "x") | {"location": "CD"}
        return pd.DataFrame([row])


def _plain(fitted_flat: str, monkeypatch: pytest.MonkeyPatch) -> None:
    """Register `plain`, a model the store does not hold."""
    from bdbvhubrenewal.forecasters import base

    factory = base._REGISTRY[fitted_flat].factory
    monkeypatch.setitem(
        base._REGISTRY, "plain", base.ModelSpec("plain", factory, "Not fitted.")
    )


def test_the_archive_holds_the_fit_the_store_holds_at_record_time(
    fitted_flat: str,
    tasks: dict[str, ForecastTask],
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from bdbvhubrenewal.evaluation import store_archive

    _plain(fitted_flat, monkeypatch)
    root = tmp_path / "store"
    origin = sorted(tasks)[-1]
    task = tasks[origin]
    models = {fitted_flat: None, "plain": None}

    def code(_: str) -> str:
        return "c"

    # Only a fit of older code: it is archived, and is not current.
    old = store.fit(fitted_flat, task, {}, "old")
    store.write_fit(root, old)
    hub = _Hub()
    table = store_archive.archive(root, task, task, models, {}, hub, code)
    assert list(table.columns) == list(store_archive.COLUMNS)
    got = table.set_index("model")
    assert got.loc[fitted_flat, "key"] == old.meta["key"]
    assert got.loc[fitted_flat, "fitted_at"] == old.meta["fitted_at"]
    assert not got.loc[fitted_flat, "current"]
    assert set(got["vintage"]) == {origin}
    # A stored model that is no candidate submits its stored samples, for
    # the round's weeks only.
    samples = hub.calls[fitted_flat]["forecast"].samples
    assert list(samples.columns) == list(SAMPLE_COLUMNS)
    weeks = {str(d) for d in task.target_end_dates}
    assert set(samples["target_end_date"].astype(str)) == weeks
    # The model the store does not hold is forecast now and not stored.
    assert hub.calls["plain"] == {
        "config": None,
        "files": [],
        "forecast": None,
    }
    assert bool(got.loc["plain", "current"])
    assert not (root / "plain").exists()

    # The fit of the current code, with a fit directory, as a candidate:
    # it forecasts from that directory with the candidate's config.
    now = store.fit(fitted_flat, task, {}, "c")
    body = base64.b64encode(b"{}").decode()
    store.write_fit(root, dataclasses.replace(now, files={"fit.json": body}))
    hub = _Hub()
    models = {fitted_flat: "flatconfig"}
    table = store_archive.archive(root, task, task, models, {}, hub, code)
    assert hub.calls[fitted_flat]["config"] == "flatconfig"
    assert hub.calls[fitted_flat]["files"] == ["fit.json"]
    got = table.set_index("model").loc[fitted_flat]
    assert got["key"] == now.meta["key"]
    assert bool(got["current"])

    # With nothing to archive, the table still has its columns.
    empty = store_archive.archive(root, task, task, {}, {}, hub, code)
    assert empty.empty
    assert list(empty.columns) == list(store_archive.COLUMNS)


def test_a_model_that_fails_leaves_the_others_archived(
    fitted_flat: str,
    tasks: dict[str, ForecastTask],
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    from bdbvhubrenewal.evaluation import store_archive
    from bdbvhubrenewal.forecasters import base

    _plain(fitted_flat, monkeypatch)

    # As a fine-tuned model whose checkpoint's release cannot be read.
    def broken() -> object:
        raise ConnectionError("no release")

    monkeypatch.setitem(
        base._REGISTRY,
        "broken",
        base.ModelSpec("broken", broken, "Broken.", fitted=True),
    )
    monkeypatch.setitem(
        base._REGISTRY,
        "nocode",
        base.ModelSpec("nocode", base._REGISTRY[fitted_flat].factory, "",
                       fitted=True),
    )  # fmt: skip
    monkeypatch.setitem(base._REGISTRY, "fails", base._REGISTRY["plain"])
    task = tasks[sorted(tasks)[-1]]
    root = tmp_path / "store"
    store.write_fit(root, store.fit(fitted_flat, task, {}, "c"))

    def code(model: str) -> str:
        if model == "nocode":
            raise KeyError("nocode has no fit code")
        return "c"

    models = dict.fromkeys(
        ["broken", "nocode", "fails", "missing", fitted_flat, "plain"]
    )
    monkeypatch.setitem(
        base._REGISTRY,
        "missing",
        base.ModelSpec("missing", base._REGISTRY[fitted_flat].factory, "",
                       fitted=True),
    )  # fmt: skip
    hub = _Hub(fail="fails")
    table = store_archive.archive(root, task, task, models, {}, hub, code)
    assert set(table["model"]) == {fitted_flat, "plain"}
    err = capsys.readouterr().err
    assert "broken not archived: ConnectionError('no release')" in err
    assert "nocode not archived: KeyError" in err
    assert "fails not archived: ValueError('no forecast')" in err
    assert "missing not archived: LookupError" in err
    # Each failure prints its traceback, so a bug can be told from a
    # missing fit.
    assert err.count("Traceback (most recent call last)") == 4
    # No store task at the origin: the stored models are reported.
    table = store_archive.archive(root, task, None, models, {}, hub, code)
    assert set(table["model"]) == {"plain"}
    assert f"{fitted_flat} not archived" in capsys.readouterr().err


def test_the_candidate_models_are_the_policy_and_the_stored_models() -> None:
    from bdbvhubrenewal.evaluation import store_archive

    models = store_archive.candidate_models(ROOT / "ci/policy.toml")
    # A random walk candidate runs its model, as `candidate.sh` does.
    assert models["randomwalk"] is None
    assert models["randomwalknowcast"] is None
    # Any other candidate forecasts with its fit config.
    policy = tomllib.loads((ROOT / "ci/policy.toml").read_text())
    for c in policy["candidates"]:
        if c["kind"] != "randomwalk":
            assert models[c["model"]] == c["config"]
    assert set(store.stored_models()) <= set(models)


def test_stored_samples_keep_the_unassigned_national_part(
    fitted_flat: str,
    tasks: dict[str, ForecastTask],
    tmp_path: Path,
) -> None:
    from bdbvhubrenewal.evaluation import store_archive

    task = tasks[sorted(tasks)[-1]]
    fit = store.fit(fitted_flat, task, {}, "c")
    rows = fit.forecasts
    # The model fits the nation on its own: 5 cases no province holds.
    national = rows["location"] == NATIONAL
    rows.loc[national, "value"] = rows.loc[national, "value"] + 5
    # A nowcast of an earlier report date, which is no forecast.
    nowcast = rows.loc[~national].assign(as_of="2026-01-01")
    rows = pd.concat([rows, nowcast], ignore_index=True)
    root = tmp_path / "store"
    store.write_fit(root, dataclasses.replace(fit, forecasts=rows))
    hub = _Hub()
    store_archive.archive(
        root, task, task, {fitted_flat: None}, {}, hub, lambda _: "c"
    )
    forecast = hub.calls[fitted_flat]["forecast"]
    assert NATIONAL not in set(forecast.samples["location"])
    assert not forecast.samples.duplicated(
        ["location", "target_end_date", "sample"]
    ).any()
    assert (forecast.unassigned["value"] == 5).all()
    n_weeks = len(task.target_end_dates)
    n_samples = forecast.samples["sample"].nunique()
    assert len(forecast.unassigned) == n_weeks * n_samples


def test_stored_samples_missing_a_week_are_not_archived(
    fitted_flat: str,
    tasks: dict[str, ForecastTask],
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    from bdbvhubrenewal.evaluation import store_archive

    task = tasks[sorted(tasks)[-1]]
    fit = store.fit(fitted_flat, task, {}, "c")
    last = max(fit.forecasts["target_end_date"])
    rows = fit.forecasts.loc[fit.forecasts["target_end_date"] != last]
    root = tmp_path / "store"
    store.write_fit(root, dataclasses.replace(fit, forecasts=rows))
    table = store_archive.archive(
        root, task, task, {fitted_flat: None}, {}, _Hub(), lambda _: "c"
    )
    assert table.empty
    assert f"{fitted_flat} not archived: ValueError" in capsys.readouterr().err


def test_the_policy_candidates_are_the_policy_models() -> None:
    from bdbvhubrenewal.evaluation import store_archive

    policy = ROOT / "ci/policy.toml"
    candidates = store_archive.policy_candidates(policy)
    # Every candidate's model, and the random walk a candidate that names
    # none runs, whatever candidates the policy holds.
    raw = tomllib.loads(policy.read_text())["candidates"]
    want = {c.get("model", store_archive.RANDOM_WALK) for c in raw}
    assert set(candidates) == want
    assert {"renewal", "randomwalk", "randomwalknowcast"} <= want
    assert set(candidates) <= set(store_archive.candidate_models(policy))
