from __future__ import annotations

import datetime as dt
import json
import os
from collections.abc import Iterator
from pathlib import Path

import pandas as pd
import pytest

from bdbvhubrenewal.cli import main
from bdbvhubrenewal.data import coherence_summary, national_coherence
from bdbvhubrenewal.reporting.plots import plot_submission
from bdbvhubrenewal.submission.fitting import (
    list_configs,
    randomwalk_config,
    register_config,
)

ROOT = Path(__file__).resolve().parents[2]


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


@pytest.fixture(scope="module")
def plain(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """The default nowcast at 2026-10-17, run once for the module."""
    out = tmp_path_factory.mktemp("plain")
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(
            "bdbvhubrenewal.submission.hub.run_date",
            lambda *_: dt.date(2026, 8, 1),
        )
        mp.chdir(ROOT)
        args = ["nowcast", "--vintage", "2026-10-17", "--out", str(out)]
        assert main(args) == 0
    return out


def test_main_runs_the_renewal_model() -> None:
    names = {c.name for c in list_configs()}
    assert {"main", "renewalsp", "randomwalk"} <= names
    assert "fallback" not in names


def test_register_config_refuses_a_taken_name() -> None:
    with pytest.raises(ValueError, match="already registered"):
        register_config(randomwalk_config("main"))


@pytest.mark.usefixtures("in_root")
def test_fit_then_nowcast_from_the_fit(tmp_path: Path, plain: Path) -> None:
    fit = tmp_path / "fit"
    sampler = '{"num_chains": 4, "seed": 1}'
    args = ["--model", "randomwalk", "--config", "randomwalk", "--vintage"]
    assert main(["fit", *args, "2026-10-17", "--sampler-json", sampler,
                 "--out", str(fit)]) == 0  # fmt: skip
    diag = json.loads((fit / "diagnostics.json").read_text())
    assert diag["sampler"] == "none"
    assert diag["status"] == "not-applicable"
    assert diag["config"] == "randomwalk"
    assert json.loads((fit / "ppc.json").read_text())["available"] is False
    assert json.loads((fit / "fit.json").read_text())["sampler"]["seed"] == 1
    assert (fit / "progress.log").read_text()

    out = tmp_path / "sub"
    assert main(["nowcast", *args, "2026-10-17", "--fit", str(fit),
                 "--out", str(out)]) == 0  # fmt: skip
    (csv,) = out.rglob("*.csv")
    assert csv.parent.name == "seabbs-randomwalk"
    table = pd.read_csv(csv)
    # Seven weeks up to the reference date and three after.
    assert table["target_end_date"].nunique() == 10
    assert set(table["output_type_id"]) == set(range(1, 101))
    wide = table.pivot_table(
        index=["output_type_id", "target_end_date"],
        columns="location",
        values="value",
    )
    # Provinces are `CD-XX`; health zones (`CD-XX-NN`) are not in the sum.
    provinces = [c for c in wide.columns if c.count("-") == 1]
    assert (wide["CD"] == wide[provinces].sum(axis=1)).all()
    side = json.loads((out / "national.json").read_text())
    assert sorted(side["modelled_provinces"]) == sorted(provinces)
    assert side["unassigned"] == []

    # The plain random walk nowcast gives the same samples.
    (csv2,) = plain.rglob("*.csv")
    pd.testing.assert_frame_equal(table, pd.read_csv(csv2))


@pytest.mark.usefixtures("in_root")
def test_nowcast_refuses_a_fit_of_another_config(tmp_path: Path) -> None:
    fit = tmp_path / "fit"
    assert main(["fit", "--config", "randomwalk", "--out", str(fit)]) == 0
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
    # baseline: the committed metadata covers the hub models only.
    args = ["--model", "randomwalknowcast", "--out", str(out)]
    assert main(["nowcast", *args]) == 0
    (csv,) = out.rglob("*.csv")
    png = tmp_path / "plot.png"
    assert main(["plot-submission", "--submission", str(csv), "--out",
                 str(png)]) == 0  # fmt: skip
    assert 0 < png.stat().st_size < 1_000_000
    # Zones have no data bars, so only national and provinces are shown.
    fig = plot_submission(csv, "2026-10-24")
    shown = [ax.get_title().split("(")[1].split(")")[0] for ax in fig.axes]
    assert all(loc.count("-") <= 1 for loc in shown)
    # The committed hub config follows the hub, so it lists the round.
    code = main(["validate", "--hub", str(ROOT / "data"), "--skip-window",
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


def test_cli_plot_ppc(tmp_path: Path) -> None:
    import numpy as np

    from bdbvhubrenewal.evaluation.ppc import (
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
        not_available("no posterior", model="randomwalk", config="main"),
        tmp_path / "na.json",
    )
    assert main(["plot-ppc", "--ppc", str(na), "--out", str(png)]) == 1


def test_delays_for_vintage_uses_the_latest_available(
    delays: pd.DataFrame,
) -> None:
    from bdbvhubrenewal.data import delays_for_vintage

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
def test_nowcast_notes_what_it_handled(tmp_path: Path, plain: Path) -> None:
    assert json.loads((plain / "handled.json").read_text()) == []
    # The first release has no pair of releases, so the model raises.
    old = tmp_path / "old"
    assert main(["nowcast", "--vintage", "2026-08-18", "--out", str(old)]) == 1
    assert not old.exists()


@pytest.mark.usefixtures("in_root")
def test_submission_id_is_one_config_value(tmp_path: Path) -> None:
    from bdbvhubrenewal.submission.hub import model_abbr_for

    hub_toml = ROOT / "config" / "hub.toml"
    assert model_abbr_for("renewalbvd", hub_toml) == "renewal"
    cfg = tmp_path / "hub.toml"
    cfg.write_text('[submission]\nmodel_abbr = "other"\n')
    assert model_abbr_for("renewalsp", cfg) == "other"


@pytest.mark.usefixtures("in_root")
def test_cli_hub_id(
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert main(["hub-id"]) == 0
    assert capsys.readouterr().out.strip() == "seabbs-renewal"


@pytest.mark.usefixtures("in_root")
def test_renewal_metadata_uses_the_submission_abbr(tmp_path: Path) -> None:
    assert main(["metadata", "--model", "renewal", "--out",
                 str(tmp_path)]) == 0  # fmt: skip
    text = (tmp_path / "seabbs-renewal.yml").read_text()
    assert "model_abbr: renewal" in text
    # Any model can be written under the submission id explicitly.
    out = str(tmp_path / "p")
    args = ["--model", "randomwalk", "--abbr", "renewal"]
    assert main(["metadata", *args, "--out", out]) == 0
    assert (tmp_path / "p" / "seabbs-renewal.yml").exists()


@pytest.mark.usefixtures("in_root")
def test_randomwalknowcast_has_its_own_hub_id(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    # A second hub model, submitted under its own id every round. The
    # hub's abbreviations allow no hyphen and fewer than 16 characters.
    from bdbvhubrenewal.submission.hub import model_abbr_for

    abbr = model_abbr_for("randomwalknowcast", ROOT / "config" / "hub.toml")
    assert abbr == "rwnowcast"
    assert abbr.isalnum()
    assert len(abbr) < 16
    # Models without their own id submit under the renewal id.
    assert model_abbr_for("renewalsp", ROOT / "config" / "hub.toml") == (
        "renewal"
    )
    assert main(["hub-id", "--model", "randomwalknowcast"]) == 0
    assert capsys.readouterr().out.strip() == "seabbs-rwnowcast"
    assert main(["metadata", "--model", "randomwalknowcast", "--out",
                 str(tmp_path)]) == 0  # fmt: skip
    text = (tmp_path / "seabbs-rwnowcast.yml").read_text()
    assert "model_abbr: rwnowcast" in text
    assert "https://samabbott.co.uk/BDBVHubRenewal/methods/baselines/" in text
    assert "10.12688/wellcomeopenres.25027.2" in text
    assert "language model" not in text


@pytest.mark.usefixtures("in_root")
def test_randomwalknowcast_config_runs_the_nowcast_model(
    tmp_path: Path, plain: Path
) -> None:
    fit = tmp_path / "fit"
    args = ["--config", "randomwalknowcast", "--vintage", "2026-10-17"]
    assert main(["fit", *args, "--out", str(fit)]) == 0
    diag = json.loads((fit / "diagnostics.json").read_text())
    assert diag["model"] == "randomwalknowcast"
    assert diag["status"] == "not-applicable"
    out = tmp_path / "sub"
    assert main(["nowcast", *args, "--fit", str(fit), "--out", str(out)]) == 0
    (csv,) = out.rglob("*.csv")
    (csv2,) = plain.rglob("*.csv")
    now, base = pd.read_csv(csv), pd.read_csv(csv2)
    assert len(now) == len(base)
    # The nowcast lifts a recent week above its published total (314), where
    # random walk carries an older week forward.
    week = now["target_end_date"] == "2026-10-10"
    nat = (now["location"] == "CD") & week
    assert now.loc[nat, "value"].median() > 314
    assert now.loc[nat, "value"].std() > 0


def test_committed_metadata_is_only_for_hub_models() -> None:
    # One file per hub model in config/hub.toml, written as the CLI does.
    names = sorted(p.stem for p in (ROOT / "hub" / "model-metadata").glob("*"))
    assert names == [
        "seabbs-renewal",
        "seabbs-rwnowcast",
    ]
    for name in names:
        text = (ROOT / "hub" / "model-metadata" / f"{name}.yml").read_text()
        assert "language model" not in text
        assert "NIHR207404" in text
        # Team seabbs, by name and abbreviation (Sam, 2026-10-28).
        assert "team_name: seabbs\n" in text
        assert "team_abbr: seabbs\n" in text


@pytest.mark.usefixtures("in_root")
def test_committed_renewal_metadata_is_one_of_the_variants(
    tmp_path: Path,
) -> None:
    # seabbs-renewal.yml is the metadata of the variant chosen for
    # submission. After a choice, regenerate it (see
    # bdbvhubrenewal.submission.metadata).
    path = ROOT / "hub" / "model-metadata" / "seabbs-renewal.yml"
    rendered = {}
    for variant in ("renewal", "renewalsp", "renewalbvd"):
        out = tmp_path / variant
        assert main(["metadata", "--model", variant, "--out", str(out)]) == 0
        rendered[variant] = (out / "seabbs-renewal.yml").read_text()
    assert path.read_text() in rendered.values()
    assert len(set(rendered.values())) == 3
