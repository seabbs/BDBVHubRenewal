from __future__ import annotations

import json
import os
from collections.abc import Iterator
from pathlib import Path

import pandas as pd
import pytest
from matplotlib.figure import Figure

from bdbvhubrenewal.cli import main
from bdbvhubrenewal.reporting import plots

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def in_root() -> Iterator[None]:
    old = Path.cwd()
    os.chdir(ROOT)
    yield
    os.chdir(old)


@pytest.mark.usefixtures("in_root")
def test_cli_nowcast_and_metadata(tmp_path: Path) -> None:
    assert main(["nowcast", "--out", str(tmp_path / "sub")]) == 0
    files = list((tmp_path / "sub").rglob("*.csv"))
    assert len(files) == 1
    table = pd.read_csv(files[0])
    assert set(table["output_type"]) == {"sample"}
    assert main(["metadata", "--out", str(tmp_path / "meta")]) == 0
    assert (tmp_path / "meta" / "epiforecasts-persistence.yml").exists()


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
    assert main(["nowcast", "--no-zones", "--out", str(tmp_path / "p")]) == 0
    assert (_written(tmp_path / "p")["location"].str.count("-") < 2).all()


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


@pytest.mark.usefixtures("in_root")
def test_cli_zone_backtest(tmp_path: Path) -> None:
    out = tmp_path / "zone-scores.csv"
    assert main(["zone-backtest", "--out", str(out)]) == 0
    scores = pd.read_csv(out)
    assert {"persistence", "zonepersistence"} <= set(scores["model"])


@pytest.mark.usefixtures("in_root")
def test_cli_refuses_heldout_without_choice(tmp_path: Path) -> None:
    cfg = (
        (ROOT / "config" / "backtest.toml")
        .read_text()
        .replace(
            'model_choice = "results/model-choice.yml"',
            f'model_choice = "{tmp_path / "missing.yml"}"',
        )
    )
    path = tmp_path / "backtest.toml"
    path.write_text(cfg)
    code = main(
        [
            "backtest",
            "--evaluate-holdout",
            "--config",
            str(path),
            "--results",
            str(tmp_path / "res"),
        ]
    )
    assert code == 1
    assert not (tmp_path / "res" / "heldout-scores.csv").exists()


def test_cli_heldout_with_no_choices_scores_nothing(tmp_path: Path) -> None:
    record = tmp_path / "choices.yml"
    record.write_text("choices: []\n")
    cfg = (
        (ROOT / "config" / "backtest.toml")
        .read_text()
        .replace(
            'model_choice = "results/model-choice.yml"',
            f'model_choice = "{record}"',
        )
    )
    path = tmp_path / "backtest.toml"
    path.write_text(cfg)
    code = main(
        [
            "backtest",
            "--evaluate-holdout",
            "--config",
            str(path),
            "--results",
            str(tmp_path / "res"),
        ]
    )
    # With no choice yet, every vintage is development: nothing is held out.
    assert code == 0
    assert not (tmp_path / "res" / "heldout-scores.csv").exists()


@pytest.mark.usefixtures("in_root")
def test_plot_nowcast_returns_figure(
    target: pd.DataFrame, delays: pd.DataFrame
) -> None:
    with plots.theme("dark"):
        fig = plots.plot_nowcast(
            "CD-IT", target=target, delays=delays, mode="dark"
        )
    assert isinstance(fig, Figure)


def test_save_light_dark(tmp_path: Path, target: pd.DataFrame) -> None:
    paths = plots.save_light_dark(
        lambda mode: plots.plot_revisions(target, mode=mode),
        tmp_path / "rev.svg",
    )
    assert [p.name for p in paths] == ["rev.svg", "rev-dark.svg"]
    assert all(p.stat().st_size > 0 for p in paths)


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
    plots._draw_forecast(ax, samples, "C0", "persistence")
    # One faint line per sample, plus the median.
    faint = [ln for ln in ax.lines if (ln.get_alpha() or 1) < 0.5]
    assert len(faint) == 100
    assert sum(ln.get_label() == "persistence" for ln in ax.lines) == 1
    # 50% and 90% ribbons.
    assert len(ax.collections) == 2
    plt.close(fig)


def test_plot_scores_by_truth() -> None:
    scores = pd.read_csv(ROOT / "results" / "backtest-scores-by-truth.csv")
    fig = plots.plot_scores_by_truth(scores, models=["persistence"])
    assert len(fig.axes) == 1
    assert len(fig.axes[0].lines) == scores["vintage"].nunique()


@pytest.mark.usefixtures("in_root")
def test_cli_lopo_writes_scores_for_mask_aware_models(tmp_path: Path) -> None:
    from evaluation.test_backtest import _MaskAware

    from bdbvhubrenewal.forecasters.base import _REGISTRY, register_model

    register_model("mask_aware", _MaskAware, "Test model.", replace=True)
    try:
        args = [
            "backtest",
            "--lopo",
            "--no-reconstructed",
            "--results",
            str(tmp_path / "res"),
            "--output",
            str(tmp_path / "out"),
        ]
        assert main([*args, "--models", "persistence", "mask_aware"]) == 0
    finally:
        _REGISTRY.pop("mask_aware")
    scores = pd.read_csv(tmp_path / "res" / "lopo-scores.csv")
    assert set(scores["model"]) == {"mask_aware"}
    assert (tmp_path / "out" / "lopo-forecasts.csv.gz").exists()
    # The development backtest is not rerun.
    assert not (tmp_path / "res" / "backtest-scores.csv").exists()


@pytest.mark.usefixtures("in_root")
def test_cli_backtest_keeps_other_models_results(tmp_path: Path) -> None:
    res, out = tmp_path / "res", tmp_path / "out"
    args = ["backtest", "--no-reconstructed", "--results", str(res),
            "--output", str(out), "--models", "persistence"]  # fmt: skip
    assert main(args) == 0
    names = ("backtest-scores.csv", "backtest-scores-by-truth.csv",
             "recovery-scores.csv")  # fmt: skip
    # Another model's results, as a fit on archie would leave them.
    for name in names:
        table = pd.read_csv(res / name)
        pd.concat([table, table.assign(model="other")]).to_csv(
            res / name, index=False
        )
    fc = pd.read_csv(out / "forecasts.csv.gz")
    pd.concat([fc, fc.assign(model="other")]).to_csv(
        out / "forecasts.csv.gz", index=False
    )
    assert main(args) == 0
    for name in names:
        table = pd.read_csv(res / name)
        assert set(table["model"]) == {"persistence", "other"}
        assert not table.duplicated().any()
    fc = pd.read_csv(out / "forecasts.csv.gz")
    assert set(fc["model"]) == {"persistence", "other"}
    assert len(fc) == 2 * (fc["model"] == "other").sum()


@pytest.mark.usefixtures("in_root")
def test_cli_backtest_no_fit_skips_fitted_models(tmp_path: Path) -> None:
    from bdbvhubrenewal.forecasters import list_models

    assert any(s.fitted for s in list_models())
    args = [
        "backtest",
        "--no-fit",
        "--no-reconstructed",
        "--results",
        str(tmp_path / "res"),
        "--output",
        str(tmp_path / "out"),
    ]
    assert main(args) == 0  # would take minutes if it fitted anything
    scores = pd.read_csv(tmp_path / "res" / "backtest-scores.csv")
    unfitted = {s.name for s in list_models() if not s.fitted}
    assert set(scores["model"]) == unfitted
    assert "persistence" in unfitted


@pytest.mark.usefixtures("in_root")
def test_cli_backtest_shards_into_a_cache(tmp_path: Path) -> None:
    res, out, cache = tmp_path / "res", tmp_path / "out", tmp_path / "cache"
    base = ["backtest", "--no-reconstructed", "--no-recovery",
            "--results", str(res), "--output", str(out),
            "--models", "persistence", "--cache", str(cache)]  # fmt: skip
    # A shard only fills the cache.
    assert main([*base, "--shard", "0/2"]) == 0
    assert main([*base, "--shard", "1/2"]) == 0
    assert list(cache.glob("*.csv.gz"))
    assert not (res / "backtest-scores.csv").exists()
    # Without a shard every task comes from the cache and is scored.
    assert main(base) == 0
    scores = pd.read_csv(res / "backtest-scores.csv")
    assert set(scores["model"]) == {"persistence"}
    assert not (res / "recovery-scores.csv").exists()


@pytest.mark.usefixtures("in_root")
def test_cli_recovery_only_uses_the_cache(tmp_path: Path) -> None:
    res, out, cache = tmp_path / "res", tmp_path / "out", tmp_path / "cache"
    base = ["backtest", "--recovery-only", "--results", str(res),
            "--output", str(out), "--models", "persistence",
            "--cache", str(cache)]  # fmt: skip
    assert main([*base, "--shard", "0/2"]) == 0
    assert main([*base, "--shard", "1/2"]) == 0
    assert list(cache.glob("persistence_simulated_*.csv.gz"))
    assert not (res / "recovery-scores.csv").exists()
    assert main(base) == 0
    scores = pd.read_csv(res / "recovery-scores.csv")
    assert set(scores["kind"]) == {"simulated"}
    # The development results are left alone.
    assert not (res / "backtest-scores.csv").exists()


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
        min_horizon=-6,
    )
    task, _ = _task(args)
    assert task.delay_history is not None
    assert task.delay_history["vintage"].nunique() >= 2
    assert task.target_history is not None
    assert task.target_history["vintage"].nunique() >= 3
    assert vintage_inputs(task, MAIN).publication_lag_pmf is not None


@pytest.mark.usefixtures("in_root")
def test_cli_backtest_scenarios(tmp_path: Path) -> None:
    args = [
        "backtest",
        "--scenarios-only",
        "--models",
        "persistence",
        "--scenario-dates",
        "2026-10-17",
        "--results",
        str(tmp_path / "res"),
        "--output",
        str(tmp_path / "out"),
    ]
    assert main(args) == 0
    scores = pd.read_csv(tmp_path / "res" / "scenario-scores.csv")
    assert set(scores["scenario"]) == {
        "calibrated_falling",
        "calibrated_flat",
        "calibrated_rising",
    }
    assert (tmp_path / "out" / "scenario-forecasts.csv.gz").exists()
