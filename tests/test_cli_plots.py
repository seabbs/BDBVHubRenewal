from __future__ import annotations

import os
from collections.abc import Iterator
from pathlib import Path

import pandas as pd
import pytest
from matplotlib.figure import Figure

from bdbv_hub_renewal import plots
from bdbv_hub_renewal.cli import main

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
