from __future__ import annotations

import os
from collections.abc import Iterator
from pathlib import Path

import pandas as pd
import pytest
from matplotlib.figure import Figure

from bdbv_forecasts import plots
from bdbv_forecasts.cli import main

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
