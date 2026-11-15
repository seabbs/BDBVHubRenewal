"""The Chronos-2 forecasters, with a stand-in for Chronos-2."""

from __future__ import annotations

import dataclasses
import hashlib
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from bdbvhubrenewal.data.zones import (
    ZoneData,
    hub_zones,
    is_zone,
    province_of,
    weekly_zone_counts,
)
from bdbvhubrenewal.evaluation.backtest import BacktestConfig, make_tasks
from bdbvhubrenewal.forecasters import (
    NATIONAL,
    ForecastTask,
    FoundationModel,
    ModelUnavailableError,
    add_national,
    check_samples,
    foundation,
    get_model,
    list_models,
)
from bdbvhubrenewal.forecasters.chain import NoWalkSpread
from bdbvhubrenewal.forecasters.copula import GaussianCopula
from bdbvhubrenewal.forecasters.foundation import (
    CHRONOS_COPULA,
    CHRONOS_DAY_RHO,
    LAYOUTS,
    MAX_LAG,
    SOURCES,
    chronos_sampler,
    hub_triangle,
    quantile_draws,
)
from bdbvhubrenewal.forecasters.randomwalk import RandomWalk
from bdbvhubrenewal.forecasters.zones import (
    ZoneAllocator,
    check_zone_coherence,
    zones_for_model,
)

FINETUNED_MODELS = tuple(f"fm_chronos_{s}_ragged_ft" for s in SOURCES)
MODELS = (
    *(f"fm_chronos_{s}_{lay}" for s in SOURCES for lay in LAYOUTS),
    *FINETUNED_MODELS,
)


def _constant(value: float):
    """A sampler whose every draw is `value`."""

    def sample(groups, horizon, n, seed):
        assert seed >= 0
        return [
            np.full((n, np.atleast_2d(g["target"]).shape[0], horizon), value)
            for g in groups
        ]

    return sample


def _independent(groups, horizon, n, seed):
    """A sampler of independent, widely spread draws."""
    rng = np.random.default_rng(seed)
    return [
        rng.gamma(
            2.0, 1e3, size=(n, np.atleast_2d(g["target"]).shape[0], horizon)
        )
        for g in groups
    ]


def _recording(store: list):
    """A sampler that keeps its inputs and draws zeros."""

    def sample(groups, horizon, n, seed):
        store.append((groups, horizon))
        return _constant(0.0)(groups, horizon, n, seed)

    return sample


@pytest.fixture(scope="module")
def task(
    target: pd.DataFrame,
    delays: pd.DataFrame,
    daily: pd.DataFrame,
    daily_zones: pd.DataFrame,
    health_zones: pd.DataFrame,
    tasks_json,
) -> ForecastTask:
    rows = target.loc[target["vintage"] == "2026-10-24"]
    real = target.loc[target["kind"] == "real"]
    return make_tasks(
        rows,
        delays,
        BacktestConfig(seed=3),
        history=real,
        daily=daily,
        zones=ZoneData(health_zones, daily_zones, hub_zones(tasks_json)),
    )[0]


def _zones(task: ForecastTask) -> list[str]:
    return [z for z in task.zone_codes if province_of(z) in task.locations]


def _reported(task: ForecastTask, out: pd.DataFrame) -> np.ndarray:
    keys = pd.MultiIndex.from_frame(out[["location", "target_end_date"]])
    assert task.zone_history is not None
    zones = task.zone_history.loc[task.zone_history["vintage"] == task.vintage]
    rows = pd.concat([task.target, zones])
    return (
        rows.groupby(["location", "target_end_date"])["total"]
        .sum()
        .reindex(keys)
        .fillna(0.0)
        .to_numpy()
    )


def _with_cases(task: ForecastTask) -> set[str]:
    """The task's places with a case in its release."""
    assert task.zone_history is not None
    zones = task.zone_history.loc[task.zone_history["vintage"] == task.vintage]
    rows = pd.concat([task.target, zones])
    places = {NATIONAL, *task.locations, *_zones(task)}
    return set(rows.loc[rows["total"] > 0, "location"]) & places


def test_registered_as_fitted_models() -> None:
    specs = {s.name: s for s in list_models()}
    for name in MODELS:
        assert specs[name].fitted
        assert not specs[name].baseline
        model = get_model(name)
        assert isinstance(model, FoundationModel)
        # The fit store keys a fit on the repr, so it must not hold an
        # address.
        assert "0x" not in repr(model)


@pytest.mark.parametrize(("source", "steps"), [("daily", 28), ("weekly", 4)])
def test_ragged_lags_stop_below_the_truth_delay(
    source: str, steps: int, task: ForecastTask
) -> None:
    # The lag series stop at 28 days or 4 weeks, short of the 35-day truth
    # delay. Running them to the truth delay (e43eda80c) was reverted: zone
    # releases began on 2026-10-09, so at 5 weeks no lag 0 to 2 value of a
    # zone fell in the context, Chronos-2 read partial counts as levels,
    # and the forecasts and scores were worse.
    store: list = []
    model = dataclasses.replace(
        get_model(f"fm_chronos_{source}_ragged"), sample=_recording(store)
    )
    model.forecast(task)
    groups, _ = store[0]
    n_places = 1 + len(task.locations)
    assert len(groups[0]["future_covariates"]) == n_places * steps
    assert MAX_LAG[source] == steps


def _history() -> pd.DataFrame:
    """Two weekly releases of one place, the second a week later."""
    return pd.DataFrame(
        {
            "vintage": ["2026-10-17"] * 2 + ["2026-10-24"] * 3,
            "location": ["A"] * 5,
            "target_end_date": [
                "2026-10-10",
                "2026-10-17",
                "2026-10-10",
                "2026-10-17",
                "2026-10-24",
            ],
            "total": [5, 1, 7, 4, 2],
            "cutoff": ["2026-10-17"] * 2 + ["2026-10-24"] * 3,
        }
    )


def test_weekly_triangle_puts_each_release_at_its_lag() -> None:
    tri = hub_triangle(_history(), "weekly", ("A",), "2026-10-31")
    assert tri.times == ["2026-10-10", "2026-10-17", "2026-10-24", "2026-10-31"]
    assert tri.last == 2
    assert tri.final[0].tolist() == [7.0, 4.0, 2.0, 0.0]
    partial = tri.partial[0]
    assert partial.shape == (4, MAX_LAG["weekly"])
    # Week 10-10 was 5 at lag 1 and 7 at lag 2; 10-17 1 then 4.
    assert partial[0, 1] == 5.0
    assert partial[0, 2] == 7.0
    assert partial[1, 0] == 1.0
    assert partial[1, 1] == 4.0
    assert partial[2, 0] == 2.0
    # No release saw week 10-10 at lag 0, or any week after the data.
    assert np.isnan(partial[0, 0])
    assert np.isnan(partial[3]).all()
    assert tri.n_context == tri.last - MAX_LAG["weekly"] + 1


def test_daily_triangle_lags_are_days() -> None:
    daily = pd.DataFrame(
        {
            "vintage": ["2026-10-17", "2026-10-24", "2026-10-24"],
            "location": ["A"] * 3,
            "onset_date": ["2026-10-15", "2026-10-15", "2026-10-20"],
            "observed": [1, 2, 1],
            "imputed": [0, 1, 0],
            "cutoff": ["2026-10-16", "2026-10-23", "2026-10-23"],
        }
    )
    tri = hub_triangle(daily, "daily", ("A",), "2026-10-24")
    # The latest release gives the final counts and the last step.
    assert tri.times[0] == "2026-10-15"
    assert tri.times[-1] == "2026-10-24"
    assert tri.last == tri.times.index("2026-10-23")
    # Observed plus imputed, at lag 1 day then 8 days.
    assert tri.partial[0, 0, 1] == 1.0
    assert tri.partial[0, 0, 8] == 3.0
    assert tri.final[0, 0] == 3.0


def test_quantile_draws_invert_the_quantile_function() -> None:
    levels = np.array([0.1, 0.5, 0.9])
    values = np.array([[[0.0, 1.0, 3.0]], [[2.0, 2.0, 2.0]]])
    u = np.array([[0.5], [0.3]])
    out = quantile_draws(levels, values, u)
    assert out.shape == (2, 1)
    assert out[0, 0] == pytest.approx(1.0)
    assert out[1, 0] == pytest.approx(2.0)
    # Outside the levels the draw is the outermost quantile.
    low = quantile_draws(levels, values[:1], np.array([[0.01]]))
    assert low[0, 0] == pytest.approx(0.0)


@pytest.mark.parametrize("name", MODELS)
def test_draws_are_counts_at_least_the_reported(
    name: str, task: ForecastTask
) -> None:
    for value in (-50.0, 0.0, 3.0):
        model = dataclasses.replace(get_model(name), sample=_constant(value))
        out = model.forecast(task)
        check_samples(out, task)
        assert set(out["location"]) == {
            NATIONAL,
            *task.locations,
            *_zones(task),
        }
        values = out["value"].to_numpy()
        assert (values >= 0).all()
        assert np.array_equal(values, np.round(values))
        reported = _reported(task, out)
        assert (values >= reported).all()
        if value < 0:
            assert np.array_equal(values, reported)
    # Draws above the reported counts raise the forecast past them.
    high = dataclasses.replace(get_model(name), sample=_constant(1e4))
    out = high.forecast(task)
    assert (out["value"].to_numpy() > _reported(task, out)).any()


def _wide(out: pd.DataFrame) -> pd.DataFrame:
    return out.pivot_table(
        index="sample",
        columns=["location", "target_end_date"],
        values="value",
    )


@pytest.mark.parametrize("name", MODELS)
def test_draws_are_paired_by_the_copula(name: str, task: ForecastTask) -> None:
    model = get_model(name)
    if "_weekly_" in name:
        assert model.copula == CHRONOS_COPULA
    else:
        # Days apart by one are paired by the daily prior.
        per_day = model.copula.rho_time ** (1 / 7)
        assert per_day == pytest.approx(CHRONOS_DAY_RHO)
        assert model.copula.rho_provinces == CHRONOS_COPULA.rho_provinces
    paired = dataclasses.replace(model, sample=_independent).forecast(task)
    unpaired = dataclasses.replace(
        model, sample=_independent, copula=GaussianCopula()
    ).forecast(task)
    wide, flat = _wide(paired), _wide(unpaired)
    weeks = sorted(paired["target_end_date"].unique())
    nat = [(NATIONAL, w) for w in weeks[-2:]]
    corr = wide[nat].corr(method="spearman").iloc[0, 1]
    assert corr > flat[nat].corr(method="spearman").iloc[0, 1] + 0.1
    if name.startswith("fm_chronos_weekly"):
        # Each weekly draw of the country and provinces is kept, only
        # paired anew. Zones, if any, are then reconciled to their
        # province draw by draw.
        places = [c for c in wide.columns if c[0].count("-") < 2]
        assert np.array_equal(
            np.sort(wide[places].to_numpy(), axis=0),
            np.sort(flat[places].to_numpy(), axis=0),
        )
    else:
        # Days of a week are paired, so the week's sum is wider.
        last = (NATIONAL, weeks[-1])
        assert wide[last].std() > 1.1 * flat[last].std()


@pytest.mark.parametrize("name", MODELS)
def test_paired_forecasts_are_reproducible(
    name: str, task: ForecastTask
) -> None:
    model = dataclasses.replace(get_model(name), sample=_independent)
    pd.testing.assert_frame_equal(model.forecast(task), model.forecast(task))


@pytest.mark.parametrize("name", MODELS)
def test_gives_no_delay(name: str, task: ForecastTask) -> None:
    model = dataclasses.replace(get_model(name), sample=_constant(0.0))
    model.forecast(task)
    # With no delay the release backtest scores it on the complete
    # target only, as for any model with none.
    assert getattr(model, "delay_summary", None) is None


@pytest.mark.parametrize("source", SOURCES)
def test_one_block_holds_every_target(source: str, task: ForecastTask) -> None:
    lags = MAX_LAG[source]
    # The country, the provinces and their zones are each a target.
    n_places = 1 + len(task.locations) + len(_zones(task))
    seen: dict[str, tuple] = {}
    for layout in LAYOUTS:
        store: list = []
        model = dataclasses.replace(
            get_model(f"fm_chronos_{source}_{layout}"),
            sample=_recording(store),
        )
        model.forecast(task)
        groups, horizon = store[0]
        assert len(groups) == 1
        seen[layout] = (groups[0], horizon)
    complete, _ = seen["complete"]
    assert complete["target"].shape[0] == n_places
    assert "future_covariates" not in complete
    ragged, horizon = seen["ragged"]
    # The targets are the hub-target series only; the lag series are
    # covariates, with no copy of a target, of the places with cases.
    assert ragged["target"].shape[0] == n_places
    future = ragged["future_covariates"]
    assert ragged["past_covariates"].keys() == future.keys()
    lagged = {name.rsplit("_lag_", 1)[0] for name in future}
    assert lagged == _with_cases(task)
    assert len(lagged) < n_places
    assert len(future) == len(lagged) * lags
    for name, values in future.items():
        d = int(name.rsplit("_", 1)[1])
        assert len(values) == horizon
        # Lag d of a step after the context is known only if it is no
        # later than the cut-off, the last `lags` steps of the context.
        assert np.isnan(values[lags - d :]).all()
    assert any(not np.isnan(v).all() for v in future.values())


@pytest.mark.parametrize("source", SOURCES)
def test_complete_starts_from_the_random_walk_week(
    source: str, task: ForecastTask
) -> None:
    store: list = []
    model = dataclasses.replace(
        get_model(f"fm_chronos_{source}_complete"), sample=_recording(store)
    )
    model.forecast(task)
    origin = RandomWalk().origin_at(task, errors=False)
    # randomwalk's start: the latest week 95% complete, the earliest
    # over places.
    start = origin.start.min()
    week = pd.Timestamp(origin.snap.week_ends[start])
    groups, _ = store[0]
    n_context = groups[0]["target"].shape[1]
    rows = task.target_history if source == "weekly" else task.daily_history
    assert rows is not None
    column = "target_end_date" if source == "weekly" else "onset_date"
    first = pd.Timestamp(rows[column].min())
    step = 7 if source == "weekly" else 1
    assert n_context == (week - first).days // step + 1


def test_weekly_ragged_needs_no_daily_releases(task: ForecastTask) -> None:
    bare = dataclasses.replace(task, daily_history=None)
    model = dataclasses.replace(
        get_model("fm_chronos_weekly_ragged"), sample=_constant(0.0)
    )
    # It reads no release snapshots, so it forecasts without them.
    assert not model.needs_snapshots
    check_samples(model.forecast(bare), bare)


def _flat(task: ForecastTask) -> ForecastTask:
    """The task with one case a day in every place and release."""
    daily = task.daily_history
    assert daily is not None
    first = daily["onset_date"].min()
    rows = [
        pd.DataFrame(
            {
                "vintage": vintage,
                "location": place,
                "onset_date": pd.date_range(first, cutoff).strftime("%Y-%m-%d"),
                "observed": 1,
                "imputed": 0,
                "cutoff": cutoff,
            }
        )
        for (vintage, cutoff), _ in daily.groupby(["vintage", "cutoff"])
        for place in daily["location"].unique()
    ]
    return dataclasses.replace(
        task, daily_history=pd.concat(rows, ignore_index=True)
    )


@pytest.mark.parametrize("source", SOURCES)
def test_complete_forecasts_with_no_walk_spread(
    source: str, task: ForecastTask
) -> None:
    flat = _flat(task)
    # A flat series gives the walk no spread, so randomwalk cannot run.
    with pytest.raises(NoWalkSpread):
        RandomWalk().origin_at(flat, errors=False)
    model = dataclasses.replace(
        get_model(f"fm_chronos_{source}_complete"), sample=_constant(0.0)
    )
    check_samples(model.forecast(flat), flat)


def test_no_daily_releases_is_unavailable(task: ForecastTask) -> None:
    bare = dataclasses.replace(task, daily_history=None)
    model = dataclasses.replace(
        get_model("fm_chronos_daily_complete"), sample=_constant(0.0)
    )
    with pytest.raises(ModelUnavailableError):
        model.forecast(bare)


class _FakePipeline:
    """Quantiles that grow with the level and the week."""

    quantiles = (0.1, 0.5, 0.9)

    def predict_quantiles(self, groups, prediction_length, quantile_levels):
        import torch

        out = []
        for g in groups:
            n_var = np.atleast_2d(g["target"]).shape[0]
            week = torch.arange(prediction_length, dtype=torch.float32)
            level = torch.tensor(quantile_levels, dtype=torch.float32)
            q = week[:, None] + level[None, :]
            out.append(q.expand(n_var, -1, -1))
        return out, None


@pytest.mark.skipif(
    importlib.util.find_spec("torch") is None, reason="needs torch"
)
def test_draws_are_independent_across_series_and_weeks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(foundation, "_chronos", lambda *_: _FakePipeline())
    groups = [{"target": np.zeros((2, 5))}, {"target": np.zeros((1, 5))}]
    draws = chronos_sampler(groups, 4, 200, 1)
    assert [d.shape for d in draws] == [(200, 2, 4), (200, 1, 4)]
    # Each draw is the week index plus a level between 0.1 and 0.9.
    level = draws[0] - np.arange(4)
    assert ((level > 0.1 - 1e-6) & (level < 0.9 + 1e-6)).all()
    assert abs(np.corrcoef(level[:, 0, 0], level[:, 0, 1])[0, 1]) < 0.2
    assert abs(np.corrcoef(level[:, 0, 0], level[:, 1, 0])[0, 1]) < 0.2


@pytest.mark.skipif(
    importlib.util.find_spec("chronos") is not None,
    reason="Chronos-2 is installed",
)
def test_chronos_missing_fails() -> None:
    # A fit without the chronos group fails, so the store records it as
    # failed, not as a fit with no forecast.
    with pytest.raises(ImportError):
        chronos_sampler([{"target": np.zeros((1, 5))}], 3, 10, 0)


@pytest.mark.parametrize("name", FINETUNED_MODELS)
def test_finetuned_models_are_the_ragged_models_with_the_checkpoint(
    name: str, fake_checkpoint
) -> None:
    model = get_model(name)
    zero_shot = get_model(name.removesuffix("_ft"))
    assert model.checkpoint == foundation.resolve_checkpoint("latest")
    assert model.checkpoint.hash == fake_checkpoint
    assert zero_shot.checkpoint is None
    assert (model.source, model.layout) == (zero_shot.source, "ragged")
    # The resolved checkpoint is in the repr, so it keys the stored fit.
    assert fake_checkpoint in repr(model)
    assert model.checkpoint.sha256[1][1] in repr(model)


def test_the_models_take_the_latest_release_by_default() -> None:
    assert foundation.finetuned_spec() == "latest"


def _listing(*releases: tuple[str, str, bool]) -> list[dict]:
    return [
        {"tag_name": tag, "published_at": at, "draft": draft}
        for tag, at, draft in releases
    ]


def test_latest_is_the_newest_published_checkpoint_release(
    releases: Path, monkeypatch: pytest.MonkeyPatch, write_release
) -> None:
    for tag_hash in ("old", "new", "draft"):
        write_release(releases, tag_hash, _ADAPTER, tag_hash)
    listing = _listing(
        ("chronos-ft-old", "2026-11-01T00:00:00Z", False),
        ("round-2026-11-14", "2026-11-20T00:00:00Z", False),
        ("chronos-ft-new", "2026-11-12T00:00:00Z", False),
        ("chronos-ft-draft", "2026-11-19T00:00:00Z", True),
    )
    monkeypatch.setattr(foundation, "finetuned_releases", lambda: listing)
    got = foundation.resolve_checkpoint("latest")
    assert got == _checkpoint("new")
    # A pinned hash takes its own release.
    assert foundation.resolve_checkpoint("old") == _checkpoint("old")


def test_a_new_release_changes_the_models_fit_key(
    releases: Path, monkeypatch: pytest.MonkeyPatch, write_release
) -> None:
    for tag_hash in ("one", "two"):
        write_release(releases, tag_hash, _ADAPTER, tag_hash)
    listing = _listing(("chronos-ft-one", "2026-11-01T00:00:00Z", False))
    monkeypatch.setattr(foundation, "finetuned_releases", lambda: listing)
    before = repr(get_model("fm_chronos_daily_ragged_ft"))
    listing += _listing(("chronos-ft-two", "2026-11-02T00:00:00Z", False))
    foundation.resolve_checkpoint.cache_clear()
    after = repr(get_model("fm_chronos_daily_ragged_ft"))
    assert "'one'" in before
    assert "'two'" in after
    assert before != after


def test_unreachable_github_fails_loudly(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, fake_checkpoint
) -> None:
    def offline() -> list[dict]:
        raise ConnectionError("GitHub cannot be reached")

    monkeypatch.setattr(foundation, "finetuned_releases", offline)
    # Even with a checkpoint cached, no older one is used in its place.
    cache = tmp_path / "cache"
    (cache / f"chronos-ft-{fake_checkpoint}").mkdir(parents=True)
    monkeypatch.setattr(foundation, "FINETUNED_CACHE", str(cache))
    foundation.resolve_checkpoint.cache_clear()
    with pytest.raises(ConnectionError):
        get_model("fm_chronos_weekly_ragged_ft")


def test_no_checkpoint_release_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    listing = _listing(("v0.1.0", "2026-11-04T00:00:00Z", False))
    monkeypatch.setattr(foundation, "finetuned_releases", lambda: listing)
    with pytest.raises(LookupError, match="chronos-ft-"):
        foundation.resolve_checkpoint("latest")


@pytest.mark.usefixtures("releases")
def test_a_release_with_no_checksums_fails() -> None:
    with pytest.raises(FileNotFoundError, match=r"sha256sums\.txt"):
        foundation.resolve_checkpoint("nosums")


def test_checksums_must_cover_the_adapter(
    releases: Path, write_release
) -> None:
    write_release(releases, "abc", _ADAPTER, "abc")
    sums = releases / "chronos-ft-abc" / "sha256sums.txt"
    sums.write_text(
        "".join(
            line + "\n"
            for line in sums.read_text().splitlines()
            if "adapter_config" not in line
        )
    )
    with pytest.raises(ValueError, match=r"adapter_config\.json"):
        foundation.resolve_checkpoint("abc")


@pytest.mark.skipif(
    importlib.util.find_spec("torch") is None, reason="needs torch"
)
def test_sampler_loads_the_model_of_the_checkpoint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    loaded: list = []

    def chronos(checkpoint=None):
        loaded.append(checkpoint)
        return _FakePipeline()

    monkeypatch.setattr(foundation, "_chronos", chronos)
    ckpt = _checkpoint("abc")
    chronos_sampler([{"target": np.zeros((1, 5))}], 3, 10, 0)
    chronos_sampler([{"target": np.zeros((1, 5))}], 3, 10, 0, ckpt)
    assert loaded == [None, ckpt]


_ADAPTER = {
    "adapter_config.json": b"{}",
    "adapter_model.safetensors": b"weights",
}


def _checkpoint(settings_hash: str) -> foundation.Checkpoint:
    """The checkpoint of the fake release's adapter files."""
    return foundation.Checkpoint(
        settings_hash,
        tuple(
            (name, hashlib.sha256(data).hexdigest())
            for name, data in _ADAPTER.items()
        ),
    )


@pytest.fixture
def releases(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """An empty fake release store and an empty adapter cache."""
    root = tmp_path / "releases"
    root.mkdir()
    monkeypatch.setattr(foundation, "FINETUNED_URL", root.as_uri())
    monkeypatch.setattr(foundation, "FINETUNED_CACHE", str(tmp_path / "cache"))
    foundation.resolve_checkpoint.cache_clear()
    return root


def test_adapter_is_downloaded_once_and_cached(
    releases: Path, tmp_path: Path, write_release
) -> None:
    write_release(releases, "abc", _ADAPTER, "abc")
    adapter = foundation.finetuned_adapter(_checkpoint("abc"))
    assert adapter == tmp_path / "cache" / "chronos-ft-abc"
    assert (adapter / "adapter_model.safetensors").read_bytes() == b"weights"
    assert (adapter / "adapter_config.json").exists()
    # With the release gone, the cached copy is used.
    for path in (releases / "chronos-ft-abc").iterdir():
        path.unlink()
    assert foundation.finetuned_adapter(_checkpoint("abc")) == adapter


@pytest.mark.usefixtures("releases")
def test_missing_release_fails_and_caches_nothing(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="chronos-ft-abc"):
        foundation.finetuned_adapter(_checkpoint("abc"))
    assert not (tmp_path / "cache" / "chronos-ft-abc").exists()


def test_release_of_another_hash_fails_and_caches_nothing(
    releases: Path, tmp_path: Path, write_release
) -> None:
    write_release(releases, "abc", _ADAPTER, "def")
    with pytest.raises(ValueError, match="def"):
        foundation.finetuned_adapter(_checkpoint("abc"))
    assert not (tmp_path / "cache" / "chronos-ft-abc").exists()


def test_adapter_of_another_sha256_fails_and_caches_nothing(
    releases: Path, tmp_path: Path, write_release
) -> None:
    write_release(releases, "abc", _ADAPTER, "abc")
    (releases / "chronos-ft-abc" / "adapter_model.safetensors").write_bytes(
        b"other weights"
    )
    with pytest.raises(ValueError, match=r"adapter_model\.safetensors"):
        foundation.finetuned_adapter(_checkpoint("abc"))
    assert not (tmp_path / "cache" / "chronos-ft-abc").exists()


def test_missing_adapter_is_stored_as_failed(
    task: ForecastTask, fake_checkpoint
) -> None:
    from bdbvhubrenewal.evaluation import store

    # The latest release lists its checksums but the adapter cannot be
    # downloaded. As a missing Chronos-2: a failed fit, not a fit with no
    # forecast.
    stored = store.fit("fm_chronos_weekly_ragged_ft", task, {}, "c")
    assert stored.meta["status"] == "failed"
    assert f"chronos-ft-{fake_checkpoint}" in stored.meta["error"]
    # The stored fit records the checkpoint it was keyed on.
    assert fake_checkpoint in stored.meta["settings"]


def test_interrupted_download_fails_and_caches_nothing(
    releases: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    write_release,
) -> None:
    write_release(releases, "abc", _ADAPTER, "abc")

    def drop(*_: object, **__: object) -> None:
        raise ConnectionResetError("reset")

    monkeypatch.setattr(foundation.urllib.request, "urlopen", drop)
    # A dropped connection is not a missing file.
    with pytest.raises(ConnectionError, match="chronos-ft-abc"):
        foundation.finetuned_adapter(_checkpoint("abc"))
    assert not (tmp_path / "cache" / "chronos-ft-abc").exists()


def test_settings_with_no_hash_fail(releases: Path, write_release) -> None:
    write_release(releases, "abc", _ADAPTER, "abc")
    (releases / "chronos-ft-abc" / "settings.json").write_text("{}")
    with pytest.raises(ValueError, match="chronos-ft-abc"):
        foundation.finetuned_adapter(_checkpoint("abc"))


def test_a_changed_cached_adapter_is_downloaded_again(
    releases: Path, write_release
) -> None:
    write_release(releases, "abc", _ADAPTER, "abc")
    adapter = foundation.finetuned_adapter(_checkpoint("abc"))
    (adapter / "adapter_model.safetensors").write_bytes(b"tampered")
    assert foundation.finetuned_adapter(_checkpoint("abc")) == adapter
    assert (adapter / "adapter_model.safetensors").read_bytes() == b"weights"


def test_the_release_list_is_read_once_per_process(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[int] = []
    listing = foundation.finetuned_releases()

    def count() -> list[dict]:
        calls.append(1)
        return listing

    monkeypatch.setattr(foundation, "finetuned_releases", count)
    foundation.resolve_checkpoint.cache_clear()
    for _ in range(3):
        for name in FINETUNED_MODELS:
            get_model(name)
    assert len(calls) == 1


def test_a_run_passes_its_resolved_checkpoint_down(
    monkeypatch: pytest.MonkeyPatch, fake_checkpoint: str
) -> None:
    # A job given the hash its run resolved never lists the releases.
    def offline() -> list[dict]:
        raise AssertionError("listed the releases")

    monkeypatch.setattr(foundation, "finetuned_releases", offline)
    monkeypatch.setenv(foundation.FINETUNED_ENV, fake_checkpoint)
    foundation.resolve_checkpoint.cache_clear()
    model = get_model("fm_chronos_daily_ragged_ft")
    assert model.checkpoint.hash == fake_checkpoint


def test_the_release_list_sends_a_token_when_set(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen: list = []

    class Response:
        def __enter__(self):  # type: ignore[no-untyped-def]
            return self

        def __exit__(self, *_: object) -> None:
            return None

        def read(self) -> bytes:
            return b"[]"

    def urlopen(request, **_: object):  # type: ignore[no-untyped-def]
        seen.append(request)
        return Response()

    monkeypatch.undo()
    monkeypatch.setattr(foundation.urllib.request, "urlopen", urlopen)
    monkeypatch.setenv("GH_TOKEN", "secret")
    assert foundation.finetuned_releases() == []
    assert seen[0].get_header("Authorization") == "Bearer secret"
    monkeypatch.delenv("GH_TOKEN")
    monkeypatch.delenv("GITHUB_TOKEN", raising=False)
    foundation.finetuned_releases()
    assert seen[1].get_header("Authorization") is None


def test_a_failed_lookup_says_how_to_set_a_token(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def refuse(*_: object, **__: object) -> None:
        raise OSError("HTTP Error 403: rate limit exceeded")

    monkeypatch.undo()
    monkeypatch.setattr(foundation, "RETRY_SECONDS", (0, 0, 0))
    monkeypatch.setattr(foundation.urllib.request, "urlopen", refuse)
    with pytest.raises(ConnectionError, match="GH_TOKEN"):
        foundation.finetuned_releases()


def _by_series(groups, horizon, n, _seed):
    """A sampler whose draws of series $i$ are $i$, so each differs."""
    return [
        np.broadcast_to(
            np.arange(len(g["target"]), dtype=float)[None, :, None],
            (n, len(g["target"]), horizon),
        ).copy()
        for g in groups
    ]


@pytest.fixture(scope="module")
def zoned(task: ForecastTask) -> pd.DataFrame:
    """Samples with the zones, from a stand-in that draws each series apart."""
    model = dataclasses.replace(
        get_model("fm_chronos_weekly_ragged"), sample=_by_series
    )
    return model.forecast(task)


def _provinces(samples: pd.DataFrame) -> pd.DataFrame:
    return samples.loc[
        ~samples["location"].map(is_zone) & (samples["location"] != NATIONAL)
    ]


def test_samples_hold_every_zone_of_the_tasks_provinces(
    task: ForecastTask, zoned: pd.DataFrame
) -> None:
    check_samples(zoned, task)
    zone = zoned["location"].map(is_zone)
    assert set(zoned.loc[zone, "location"]) == set(_zones(task))
    first = zoned.loc[zone, "location"].iloc[0]
    stray = zoned.copy()
    stray.loc[stray["location"] == first, "location"] = "CD-XX-01"
    with pytest.raises(ValueError, match="zones"):
        check_samples(stray, task)
    missing = zoned.loc[zoned["location"] != first]
    with pytest.raises(ValueError, match="zones"):
        check_samples(missing, task)


def test_national_sums_provinces_not_zones(zoned: pd.DataFrame) -> None:
    rows = zoned.loc[zoned["location"] != NATIONAL]
    out = add_national(rows)
    national = out.loc[out["location"] == NATIONAL, "value"].sum()
    assert national == _provinces(rows)["value"].sum()


def test_zone_samples_are_the_models_own_summed_to_its_provinces(
    task: ForecastTask,
    zoned: pd.DataFrame,
    daily_zones: pd.DataFrame,
    tasks_json,
) -> None:
    zones = hub_zones(tasks_json)
    counts = weekly_zone_counts(daily_zones, task.vintage)
    own = zoned.loc[zoned["location"].map(is_zone)]
    provinces = _provinces(zoned)
    # The model's zones sum to its provinces and keep the published counts.
    assert not check_zone_coherence(provinces, own)
    keys = ["location", "target_end_date"]
    floor = own.merge(counts, on=keys, how="left")["total"].fillna(0).to_numpy()
    assert (own["value"].to_numpy() >= floor).all()
    # The hub file takes them as they are.
    split = zones_for_model(
        "fm_chronos_weekly_ragged", task, zoned, counts, zones, daily_zones
    )
    keys = ["location", "target_end_date", "sample"]
    a = split.set_index(keys)["value"].sort_index()
    pd.testing.assert_series_equal(a, own.set_index(keys)["value"].sort_index())
    allocated = ZoneAllocator().allocate(provinces, task, counts, zones)
    b = allocated.set_index(keys)["value"].sort_index()
    assert not a.equals(b)


def test_a_passing_github_error_is_retried(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import urllib.error

    calls: list[int] = []

    class Response:
        def __enter__(self):  # type: ignore[no-untyped-def]
            return self

        def __exit__(self, *_: object) -> None:
            return None

        def read(self) -> bytes:
            return b"[]"

    def flaky(request, **_: object):  # type: ignore[no-untyped-def]
        calls.append(1)
        if len(calls) < 3:
            raise urllib.error.HTTPError(
                request.full_url, 502, "Bad Gateway", {}, None
            )
        return Response()

    monkeypatch.undo()
    monkeypatch.setattr(foundation, "RETRY_SECONDS", (0, 0, 0))
    monkeypatch.setattr(foundation.urllib.request, "urlopen", flaky)
    assert foundation.finetuned_releases() == []
    assert len(calls) == 3


def test_a_missing_asset_is_not_retried(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import urllib.error

    calls: list[int] = []

    def missing(request, **_: object):  # type: ignore[no-untyped-def]
        calls.append(1)
        raise urllib.error.HTTPError(
            request.full_url, 404, "Not Found", {}, None
        )

    monkeypatch.undo()
    monkeypatch.setattr(foundation, "RETRY_SECONDS", (0, 0, 0))
    monkeypatch.setattr(foundation.urllib.request, "urlopen", missing)
    with pytest.raises(FileNotFoundError):
        foundation.resolve_checkpoint("abc")
    assert len(calls) == 1


def _http_error(code: int):  # type: ignore[no-untyped-def]
    """An `urlopen` that always fails with HTTP `code`, counting calls."""
    import urllib.error

    calls: list[int] = []

    def fail(request, **_: object):  # type: ignore[no-untyped-def]
        calls.append(1)
        raise urllib.error.HTTPError(request.full_url, code, "no", {}, None)

    return fail, calls


@pytest.mark.usefixtures("releases")
def test_a_rate_limited_download_names_the_token(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fail, calls = _http_error(403)
    monkeypatch.setattr(foundation.urllib.request, "urlopen", fail)
    with pytest.raises(ConnectionError, match="GH_TOKEN") as err:
        foundation.finetuned_adapter(_checkpoint("abc"))
    assert "has no" not in str(err.value)
    assert len(calls) == 1


@pytest.mark.usefixtures("releases")
def test_a_missing_file_is_named_and_not_retried(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fail, calls = _http_error(404)
    monkeypatch.setattr(foundation.urllib.request, "urlopen", fail)
    with pytest.raises(FileNotFoundError, match="has no adapter_config"):
        foundation.finetuned_adapter(_checkpoint("abc"))
    assert len(calls) == 1


@pytest.mark.usefixtures("releases")
def test_too_many_requests_is_retried_a_bounded_number_of_times(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fail, calls = _http_error(429)
    monkeypatch.setattr(foundation.urllib.request, "urlopen", fail)
    with pytest.raises(ConnectionError, match="GH_TOKEN"):
        foundation.finetuned_adapter(_checkpoint("abc"))
    assert len(calls) == len(foundation.RETRY_SECONDS) + 1
