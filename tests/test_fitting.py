from __future__ import annotations

import json
import os
from collections.abc import Iterator
from pathlib import Path

import pandas as pd
import pytest

from bdbv_hub_renewal.cli import main
from bdbv_hub_renewal.data import coherence_summary, national_coherence
from bdbv_hub_renewal.fitting import (
    get_config,
    list_configs,
    persistence_config,
    register_config,
)

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def in_root() -> Iterator[None]:
    old = Path.cwd()
    os.chdir(ROOT)
    yield
    os.chdir(old)


@pytest.fixture
def stand_in() -> str:
    register_config(
        persistence_config("standin", stand_in="a test stand-in"), replace=True
    )
    return "standin"


def test_main_and_fallback_run_the_renewal_model() -> None:
    names = {c.name for c in list_configs()}
    assert {"main", "fallback", "persistence"} <= names
    for name in ("main", "fallback", "persistence"):
        assert get_config(name).stand_in is None


def test_register_config_needs_replace() -> None:
    with pytest.raises(ValueError, match="already registered"):
        register_config(persistence_config("main"))
    old = get_config("main")
    try:
        new = register_config(persistence_config("main"), replace=True)
        assert get_config("main") is new
    finally:
        register_config(old, replace=True)


def test_version(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["--version"]) == 0
    assert capsys.readouterr().out.strip() == "0.1.0"


@pytest.mark.usefixtures("in_root")
def test_fit_then_nowcast_from_the_fit(tmp_path: Path, stand_in: str) -> None:
    fit = tmp_path / "fit"
    sampler = '{"num_chains": 4, "seed": 1}'
    args = ["--model", "persistence", "--config", stand_in, "--vintage"]
    assert main(["fit", *args, "2026-10-17", "--sampler-json", sampler,
                 "--out", str(fit)]) == 0  # fmt: skip
    diag = json.loads((fit / "diagnostics.json").read_text())
    assert diag["sampler"] == "none"
    assert diag["status"] == "not-applicable"
    assert diag["config"] == stand_in
    assert diag["stand_in"] == "a test stand-in"
    assert json.loads((fit / "ppc.json").read_text())["available"] is False
    assert json.loads((fit / "fit.json").read_text())["sampler"]["seed"] == 1
    assert (fit / "progress.log").read_text()

    out = tmp_path / "sub"
    assert main(["nowcast", *args, "2026-10-17", "--fit", str(fit),
                 "--out", str(out)]) == 0  # fmt: skip
    (csv,) = out.rglob("*.csv")
    assert csv.parent.name == "epiforecasts-persistence"
    table = pd.read_csv(csv)
    # Seven weeks up to the reference date and three after.
    assert table["target_end_date"].nunique() == 10
    assert set(table["output_type_id"]) == set(range(1, 101))
    wide = table.pivot_table(
        index=["output_type_id", "target_end_date"],
        columns="location",
        values="value",
    )
    provinces = [c for c in wide.columns if c != "CD"]
    assert (wide["CD"] == wide[provinces].sum(axis=1)).all()
    side = json.loads((out / "national.json").read_text())
    assert sorted(side["modelled_provinces"]) == sorted(provinces)
    assert side["unassigned"] == []

    # The plain persistence nowcast gives the same samples.
    plain = tmp_path / "plain"
    assert main(["nowcast", "--vintage", "2026-10-17", "--out",
                 str(plain)]) == 0  # fmt: skip
    (csv2,) = plain.rglob("*.csv")
    pd.testing.assert_frame_equal(table, pd.read_csv(csv2))


@pytest.mark.usefixtures("in_root")
def test_nowcast_refuses_a_fit_of_another_config(tmp_path: Path) -> None:
    fit = tmp_path / "fit"
    assert main(["fit", "--config", "persistence", "--out", str(fit)]) == 0
    code = main(
        ["nowcast", "--config", "main", "--fit", str(fit), "--out",
         str(tmp_path / "s")]
    )  # fmt: skip
    assert code == 1


@pytest.mark.usefixtures("in_root")
def test_fit_fails_on_unknown_config_or_vintage(tmp_path: Path) -> None:
    out = str(tmp_path / "fit")
    assert main(["fit", "--config", "nope", "--out", out]) == 1
    bad = ["fit", "--config", "main", "--vintage", "1999-01-02"]
    assert main([*bad, "--out", out]) == 1


@pytest.mark.usefixtures("in_root")
def test_plot_submission_and_validate(tmp_path: Path) -> None:
    out = tmp_path / "sub"
    assert main(["nowcast", "--out", str(out)]) == 0
    (csv,) = out.rglob("*.csv")
    png = tmp_path / "plot.png"
    assert main(["plot-submission", "--submission", str(csv), "--out",
                 str(png)]) == 0  # fmt: skip
    assert 0 < png.stat().st_size < 1_000_000
    snapshot = ROOT / "tests/hub_validation/fixtures/hub-snapshot"
    code = main(["validate", "--hub", str(snapshot), "--skip-window",
                 "--metadata-dir", "hub/model-metadata", str(csv)])  # fmt: skip
    assert code == 0


def _with_gap(target: pd.DataFrame, vintage: str) -> pd.DataFrame:
    gap = target.copy()
    mask = (
        (gap["vintage"] == vintage)
        & (gap["location"] == "CD")
        & (gap["target_end_date"] == "2026-10-03")
    )
    gap.loc[mask, ["observed", "total"]] += 3
    return gap


def test_national_coherence(target: pd.DataFrame) -> None:
    table = national_coherence(target)
    assert set(table["stream"]) == {"observed", "imputed", "total"}
    assert (table["gap"] == 0).all()
    summary = coherence_summary(target, "2026-10-24")
    assert summary["coherent"] is True
    assert summary["rows"] == []


def test_national_coherence_reports_a_gap(target: pd.DataFrame) -> None:
    summary = coherence_summary(_with_gap(target, "2026-10-24"), "2026-10-24")
    assert summary["coherent"] is False
    assert summary["streams"]["observed"]["weeks_with_gap"] == 1
    assert summary["streams"]["imputed"]["weeks_with_gap"] == 0
    assert summary["streams"]["total"]["max_abs_gap"] == 3
    assert {r["stream"] for r in summary["rows"]} == {"observed", "total"}
    assert "unassigned" in summary["handling"]


@pytest.mark.usefixtures("in_root")
def test_cli_coherence(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    out = tmp_path / "coherence.json"
    assert main(["coherence", "--out", str(out)]) == 0
    assert json.loads(out.read_text())["coherent"] is True
    assert "equal the sum" in capsys.readouterr().out


def test_cli_plot_ppc(tmp_path: Path) -> None:
    import numpy as np

    from bdbv_hub_renewal.evaluation.ppc import (
        not_available,
        summarise_draws,
        write_ppc,
    )

    weeks = pd.date_range("2026-08-01", periods=10, freq="7D").strftime(
        "%Y-%m-%d"
    )
    obs = pd.DataFrame(
        {
            "location": "CD-IT",
            "stream": "total",
            "target_end_date": weeks,
            "observed": np.arange(10) + 5,
        }
    )
    draws = np.random.default_rng(1).poisson(obs["observed"], (200, 10))
    ppc = write_ppc(
        summarise_draws(obs, draws, as_of="2026-10-24"), tmp_path / "p.json"
    )
    png = tmp_path / "ppc.png"
    assert main(["plot-ppc", "--ppc", str(ppc), "--out", str(png)]) == 0
    assert 0 < png.stat().st_size < 1_000_000
    na = write_ppc(
        not_available("no posterior", model="persistence", config="main"),
        tmp_path / "na.json",
    )
    assert main(["plot-ppc", "--ppc", str(na), "--out", str(png)]) == 1


def test_delays_for_vintage_uses_the_latest_available(
    delays: pd.DataFrame,
) -> None:
    from bdbv_hub_renewal.data import delays_for_vintage

    have = str(delays["vintage"].max())
    same, note = delays_for_vintage(delays, have)
    assert note == ""
    assert same is delays
    later, note = delays_for_vintage(delays, "2099-01-03")
    assert f"so used the {have} delays" in note
    n = (delays["vintage"] == have).sum()
    assert (later["vintage"] == "2099-01-03").sum() == n
    _, note = delays_for_vintage(delays, "2000-01-01")
    assert "default completeness lag" in note


@pytest.mark.usefixtures("in_root")
def test_nowcast_notes_what_it_handled(tmp_path: Path) -> None:
    out = tmp_path / "sub"
    assert main(["nowcast", "--vintage", "2026-10-17", "--out", str(out)]) == 0
    assert json.loads((out / "handled.json").read_text()) == []
    old = tmp_path / "old"
    assert main(["nowcast", "--vintage", "2026-08-18", "--out", str(old)]) == 0
    (note,) = json.loads((old / "handled.json").read_text())
    assert note.startswith("delays: none published by 2026-08-18")


@pytest.mark.usefixtures("in_root")
def test_submission_id_is_one_config_value(tmp_path: Path) -> None:
    from bdbv_hub_renewal.hub import submission_model_abbr

    assert (
        submission_model_abbr(ROOT / "config" / "hub.toml") == "seabbsrenewal"
    )
    cfg = tmp_path / "hub.toml"
    cfg.write_text('[submission]\nmodel_abbr = "other"\n')
    assert submission_model_abbr(cfg) == "other"


@pytest.mark.usefixtures("in_root")
def test_cli_hub_id_and_chosen_model(
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert main(["hub-id"]) == 0
    assert capsys.readouterr().out.strip() == "epiforecasts-seabbsrenewal"
    assert main(["chosen-model"]) == 0
    # The latest entry of results/model-choice.yml.
    assert capsys.readouterr().out.strip() == "persistence"


@pytest.mark.usefixtures("in_root")
def test_renewal_metadata_uses_the_submission_abbr(tmp_path: Path) -> None:
    assert main(["metadata", "--model", "renewal", "--out",
                 str(tmp_path)]) == 0  # fmt: skip
    text = (tmp_path / "epiforecasts-seabbsrenewal.yml").read_text()
    assert "model_abbr: seabbsrenewal" in text
    # Any model can be written under the submission id explicitly.
    out = str(tmp_path / "p")
    args = ["--model", "persistence", "--abbr", "seabbsrenewal"]
    assert main(["metadata", *args, "--out", out]) == 0
    assert (tmp_path / "p" / "epiforecasts-seabbsrenewal.yml").exists()
