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
    assert main(["nowcast", "--out", str(tmp_path / "sub")]) == 0
    table = _written(tmp_path / "sub")
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
        data="data",
        backtest_config="config/backtest.toml",
    )
    task, _ = _task(args)
    assert task.delay_history is not None
    assert task.delay_history["vintage"].nunique() >= 2
    assert task.target_history is not None
    assert task.target_history["vintage"].nunique() >= 3
    assert vintage_inputs(task, MAIN).publication_lag_pmf is not None


@pytest.mark.usefixtures("in_root")
def test_stored_samples_keep_numeric_values_without_our_forecasts() -> None:
    # A backtest of fitted models only has no development forecasts of its
    # own, so the stored samples are the hub teams' alone.
    import argparse

    import numpy as np

    from bdbvhubrenewal.cli import _stored_samples
    from bdbvhubrenewal.data import load_target
    from bdbvhubrenewal.evaluation.backtest import FORECAST_COLUMNS
    from bdbvhubrenewal.evaluation.scoring import energy_scores

    args = argparse.Namespace(data="data")
    stored = _stored_samples(
        args, load_target("data"), pd.DataFrame(columns=FORECAST_COLUMNS)
    )
    assert stored["value"].dtype == np.float64
    one = stored.loc[stored["vintage"] == stored["vintage"].max()]
    truth = one[["location", "target_end_date"]].drop_duplicates()
    energy = energy_scores(
        one.assign(origin="o"), truth.assign(truth=1.0), unit=["origin"]
    )
    assert np.isfinite(energy["log_energy"]).all()
