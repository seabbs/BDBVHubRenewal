from __future__ import annotations

import datetime as dt
import json
import os
from collections.abc import Iterator
from pathlib import Path

import pandas as pd
import pytest

from bdbvhubrenewal.cli import main
from bdbvhubrenewal.reporting import plots

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "tests" / "reporting" / "fixtures"


@pytest.fixture
def in_root(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    # The vintages used here are in the past, and a forecast date cannot
    # follow its reference date.
    monkeypatch.setattr(
        "bdbvhubrenewal.submission.hub.run_date",
        lambda *_: dt.date(2026, 8, 1),
    )
    old = Path.cwd()
    os.chdir(ROOT)
    yield
    os.chdir(old)


def _written(out: Path) -> pd.DataFrame:
    files = list(out.rglob("*.csv"))
    assert len(files) == 1
    return pd.read_csv(files[0])


@pytest.mark.usefixtures("in_root")
def test_cli_nowcast_writes_coherent_zones(tmp_path: Path) -> None:
    args = ["nowcast", "--out", str(tmp_path / "sub")]
    assert main([*args, "--forecast-date", "2026-07-31"]) == 0
    table = _written(tmp_path / "sub")
    assert set(table["forecast_date"]) == {"2026-07-31"}
    level = table["location"].str.count("-")
    zones = table.loc[level == 2]
    provs = table.loc[level == 1]
    assert zones["location"].nunique() > 150
    side = json.loads((tmp_path / "sub" / "zones.json").read_text())
    assert side["zones"] is True
    assert side["n_zones"] == zones["location"].nunique()
    assert side["note"] == ""
    keys = ["output_type_id", "target_end_date"]
    summed = (
        zones.assign(location=zones["location"].str.slice(0, 5))
        .groupby(["location", *keys])["value"]
        .sum()
    )
    want = provs.set_index(["location", *keys])["value"]
    assert (summed.reindex(want.index) == want).all()
    nat = table.loc[level == 0].set_index(keys)["value"]
    assert (nat == provs.groupby(keys)["value"].sum().reindex(nat.index)).all()


@pytest.mark.usefixtures("in_root")
def test_cli_nowcast_without_zone_data_skips_zones(tmp_path: Path) -> None:
    data = tmp_path / "data"
    data.mkdir()
    for f in (ROOT / "data").iterdir():
        if f.name != "daily_zones.csv.gz":
            (data / f.name).symlink_to(f)
    out = tmp_path / "sub"
    assert main(["nowcast", "--data", str(data), "--out", str(out)]) == 0
    assert (_written(out)["location"].str.count("-") < 2).all()
    assert "zone" in (out / "handled.json").read_text()
    side = json.loads((out / "zones.json").read_text())
    assert side["zones"] is False
    assert side["n_zones"] == 0
    assert "zone" in side["note"]


def test_forecast_draws_trajectories_ribbons_and_median() -> None:
    import matplotlib.pyplot as plt

    weeks = ["2026-10-03", "2026-10-10", "2026-10-17"]
    samples = pd.DataFrame(
        {
            "target_end_date": weeks * 100,
            "sample": [s for s in range(1, 101) for _ in weeks],
            "value": [float(s + i) for s in range(100) for i in range(3)],
        }
    )
    fig, ax = plt.subplots()
    plots._draw_forecast(ax, samples, "C0", "", "randomwalk")
    # One faint line per sample, plus the median.
    faint = [ln for ln in ax.lines if (ln.get_alpha() or 1) < 0.5]
    assert len(faint) == 100
    assert sum(ln.get_label() == "randomwalk" for ln in ax.lines) == 1
    # 50% and 90% ribbons.
    assert len(ax.collections) == 2
    plt.close(fig)


def test_plot_scores_by_truth() -> None:
    scores = pd.read_csv(FIXTURES / "nowcast-scores.csv")
    scores = scores.rename(columns={"origin": "vintage"})
    fig = plots.plot_scores_by_truth(scores, models=["randomwalk"])
    assert len(fig.axes) == 1
    own = scores.loc[scores["model"] == "randomwalk"]
    assert len(fig.axes[0].lines) == own["vintage"].nunique()


@pytest.mark.usefixtures("in_root", "flat_model")
def test_cli_lopo_writes_scores_for_mask_aware_models(tmp_path: Path) -> None:
    from evaluation.test_backtest import _MaskAware

    from bdbvhubrenewal.forecasters.base import _REGISTRY, register_model

    register_model("mask_aware", _MaskAware, "Test model.", replace=True)
    try:
        args = [
            "backtest",
            "--lopo",
            "--results",
            str(tmp_path / "res"),
            "--output",
            str(tmp_path / "out"),
        ]
        assert main([*args, "--models", "randomwalk", "mask_aware"]) == 0
    finally:
        _REGISTRY.pop("mask_aware")
    scores = pd.read_csv(tmp_path / "res" / "lopo-scores.csv")
    assert set(scores["model"]) == {"mask_aware"}
    assert (tmp_path / "out" / "lopo-forecasts.csv.gz").exists()
    # The development backtest is not rerun.
    assert not (tmp_path / "res" / "nowcast-scores.csv").exists()


def test_host_devices_cover_the_chains(monkeypatch: pytest.MonkeyPatch) -> None:
    from bdbvhubrenewal.cli import host_devices

    monkeypatch.setenv("XLA_FLAGS", "--xla_foo=1")
    host_devices(4)
    flags = os.environ["XLA_FLAGS"]
    assert "--xla_foo=1" in flags
    assert "--xla_force_host_platform_device_count=4" in flags
    # A larger count already set is kept.
    monkeypatch.setenv("XLA_FLAGS", "--xla_force_host_platform_device_count=8")
    host_devices(4)
    assert os.environ["XLA_FLAGS"] == "--xla_force_host_platform_device_count=8"


@pytest.mark.usefixtures("in_root")
def test_cli_fit_task_carries_the_delay_files_for_the_lag() -> None:
    # A live fit sees every earlier delay file, so it uses the empirical
    # publication lag, not the prior.
    import argparse

    from bdbvhubrenewal.cli import _task
    from bdbvhubrenewal.forecasters.renewal import vintage_inputs
    from bdbvhubrenewal.model import MAIN

    args = argparse.Namespace(
        vintage="latest",
        reference_date=None,
        data="data",
        backtest_config="config/backtest.toml",
    )
    task, _ = _task(args)
    assert task.delay_history is not None
    assert task.delay_history["vintage"].nunique() >= 2
    assert task.target_history is not None
    assert task.target_history["vintage"].nunique() >= 3
    assert vintage_inputs(task, MAIN).release_pairs is not None


@pytest.mark.usefixtures("in_root")
def test_stored_samples_keep_numeric_values_without_our_forecasts(
    tmp_path: Path,
) -> None:
    # Guards the dtypes of a backtest with no development forecasts.
    import argparse

    import numpy as np

    from bdbvhubrenewal.cli import _stored
    from bdbvhubrenewal.data import load_target
    from bdbvhubrenewal.evaluation.backtest import run_forecasts
    from bdbvhubrenewal.evaluation.scoring import energy_scores
    from bdbvhubrenewal.forecasters.renewal import DELAY_SUMMARY_FILE

    args = argparse.Namespace(data="data", results=str(tmp_path))
    # With no fits loaded there are no stored delays.
    _, none = _stored(args, load_target("data"), run_forecasts([], []))
    assert none.empty
    learnt = pd.DataFrame(
        {"model": ["renewal", "randomwalknowcast"], "vintage": "2026-10-24"}
    )
    learnt.to_csv(tmp_path / DELAY_SUMMARY_FILE, index=False)
    stored, delays = _stored(args, load_target("data"), run_forecasts([], []))
    # Only the fitted models' delays are stored ones.
    assert set(delays["model"]) == {"renewal"}
    assert stored["value"].dtype == np.float64
    one = stored.loc[stored["vintage"] == stored["vintage"].max()]
    truth = one[["location", "target_end_date"]].drop_duplicates()
    energy = energy_scores(
        one.assign(origin="o"), truth.assign(truth=1.0), unit=["origin"]
    )
    assert np.isfinite(energy["log_energy"]).all()


@pytest.mark.usefixtures("in_root")
def test_backtest_of_fitted_models_only_points_to_the_fit_store(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    args = ["backtest", "--models", "pprandomwalknowcast"]
    out = ["--results", str(tmp_path), "--output", str(tmp_path)]
    assert main([*args, *out]) == 1
    assert "task fit-store" in capsys.readouterr().err


@pytest.mark.usefixtures("in_root")
def test_cli_fit_store_plans_fits_and_loads(
    flat_model: str, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from bdbvhubrenewal.forecasters import base

    factory = base._REGISTRY[flat_model].factory

    class Broken:
        name = "broken"

        def forecast(self, task: object) -> pd.DataFrame:
            del task
            raise RuntimeError("no fit")

    monkeypatch.setattr(
        base,
        "_REGISTRY",
        {
            "flat": base.ModelSpec("flat", factory, "Fitted.", fitted=True),
            "broken": base.ModelSpec("broken", Broken, "Fails.", fitted=True),
        },
    )
    store, matrix = tmp_path / "store", tmp_path / "matrix.json"
    plan = ["fit-store", "plan", "--store", str(store), "--out", str(matrix)]
    assert main(plan) == 0
    missing = json.loads(matrix.read_text())["include"]
    assert {m["model"] for m in missing} == {"flat", "broken"}
    fit = ["fit-store", "fit", "--store", str(store), "--origin", "2026-10-24"]
    assert main([*fit, "--model", "flat"]) == 0
    assert main([*fit, "--model", "broken"]) == 1
    assert main([*fit, "--model", "renewal"]) == 1
    assert main(plan) == 0
    assert len(json.loads(matrix.read_text())["include"]) == len(missing) - 2
    res = tmp_path / "res"
    load = ["fit-store", "load", "--store", str(store), "--results", str(res)]
    assert main(load) == 0
    # 2026-10-24 is a development origin.
    dev = pd.read_csv(res / "forecasts.csv.gz")
    assert set(dev["vintage"]) == {"2026-10-24"}
    status = json.loads((res / "fit-store.json").read_text())
    assert [s["failed"] for s in status if s["model"] == "broken"].count(
        True
    ) == 1


@pytest.mark.usefixtures("in_root")
def test_cli_nowcast_forecasts_a_later_round_from_an_older_vintage(
    tmp_path: Path,
) -> None:
    out = tmp_path / "sub"
    args = ["nowcast", "--vintage", "2026-10-24", "--out", str(out)]
    assert main([*args, "--reference-date", "2026-11-07"]) == 0
    table = _written(out)
    assert set(table["reference_date"]) == {"2026-11-07"}
    assert table["target_end_date"].max() == "2026-11-28"
    handled = json.loads((out / "handled.json").read_text())
    assert "round 2026-11-07 forecast from the 2026-10-24 release" in handled


@pytest.mark.usefixtures("in_root")
def test_backtest_runs_only_the_named_scenarios(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    seen: dict[str, object] = {}

    def run(*args: object, **kwargs: object) -> tuple[pd.DataFrame, ...]:
        del args
        seen.update(kwargs)
        table = pd.DataFrame({"model": ["randomwalk"], "scenario": ["flat"]})
        return table, table, table, table

    monkeypatch.setattr("bdbvhubrenewal.evaluation.backtest.run_scenarios", run)
    out = ["--results", str(tmp_path), "--output", str(tmp_path)]
    args = ["backtest", "--scenarios-only", "--models", "randomwalk"]
    assert main([*args, "--scenarios", "flat", *out]) == 0
    assert seen["names"] == ["flat"]
    with pytest.raises(SystemExit):
        main([*args, "--scenarios", "sideways", *out])


def test_concat_results_keeps_every_partial(tmp_path: Path) -> None:
    from bdbvhubrenewal.data.tidy import write_csv

    parts = [tmp_path / "falling", tmp_path / "flat"]
    for part in parts:
        part.mkdir()
        write_csv(
            pd.DataFrame({"model": ["randomwalk", "randomwalknowcast"]}).assign(
                scenario=part.name
            ),
            part / "scenario-scores.csv.gz",
        )
    write_csv(pd.DataFrame({"truth": [1]}), parts[0] / "only.csv")
    out = tmp_path / "results"
    assert main(["concat-results", "--out", str(out), *map(str, parts)]) == 0
    scores = pd.read_csv(out / "scenario-scores.csv.gz")
    assert list(scores["scenario"]) == ["falling"] * 2 + ["flat"] * 2
    assert len(pd.read_csv(out / "only.csv")) == 1


def test_digest_prints_the_backtest_digest(
    capsys: pytest.CaptureFixture[str],
) -> None:
    from bdbvhubrenewal.evaluation.digest import backtest_digest

    args = ["digest", "--root", str(ROOT), "--text", "cell"]
    assert main([*args, "--with", "results"]) == 0
    key = capsys.readouterr().out.strip()
    assert key == backtest_digest(ROOT, extra=["results"], text="cell")


def test_chunk_splits_in_order_into_contiguous_blocks() -> None:
    from bdbvhubrenewal.cli import chunk

    items = list("abcdefg")
    parts = [chunk(items, f"{k}/3") for k in range(3)]
    assert parts == [list("abc"), list("de"), list("fg")]
    assert chunk(items, "0/1") == items
    assert chunk(items[:2], "2/3") == []
    for bad in ("3/3", "x", "1/0"):
        with pytest.raises(ValueError, match="chunk"):
            chunk(items, bad)


@pytest.mark.usefixtures("in_root")
def test_backtest_runs_recovery_apart_by_chunk(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from bdbvhubrenewal.evaluation import backtest as bt

    dates: list[tuple[str, ...]] = []

    def recovery(
        models: object, config: bt.BacktestConfig, **kw: object
    ) -> tuple[pd.DataFrame, ...]:
        del models, kw
        dates.append(config.simulated_dates)
        table = pd.DataFrame({"model": ["randomwalk"]})
        return table, table, table

    def development(*args: object, **kwargs: object) -> pd.DataFrame:
        del args, kwargs
        raise AssertionError("ran the development backtest")

    monkeypatch.setattr(bt, "run_recovery", recovery)
    monkeypatch.setattr(bt, "run_development", development)
    out = ["--results", str(tmp_path), "--output", str(tmp_path)]
    args = ["backtest", "--recovery-only", "--models", "randomwalk", *out]
    assert main([*args, "--chunk", "1/2"]) == 0
    everything = bt.load_config("config/backtest.toml").simulated_dates
    assert dates == [tuple(chunk_of(everything, 1, 2))]
    assert (tmp_path / "recovery-scores.csv").exists()


def chunk_of(items: tuple[str, ...], k: int, n: int) -> list[str]:
    from bdbvhubrenewal.cli import chunk

    return chunk(items, f"{k}/{n}")


@pytest.mark.usefixtures("in_root")
def test_backtest_without_recovery_never_runs_it(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from bdbvhubrenewal import cli
    from bdbvhubrenewal.evaluation import backtest as bt

    def recovery(*args: object, **kwargs: object) -> None:
        del args, kwargs
        raise AssertionError("ran the recovery check")

    monkeypatch.setattr(bt, "run_recovery", recovery)

    def development(*args: object, **kwargs: object) -> pd.DataFrame:
        del args, kwargs
        return pd.DataFrame({"model": []})

    def release(*args: object, **kwargs: object) -> None:
        del args, kwargs

    monkeypatch.setattr(bt, "run_development", development)
    monkeypatch.setattr(cli, "_release_backtest", release)
    out = ["--results", str(tmp_path), "--output", str(tmp_path)]
    args = ["backtest", "--no-recovery", "--models", "randomwalk", *out]
    assert main(args) == 0
