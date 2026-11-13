from __future__ import annotations

import datetime as dt
import json
import os
import subprocess
import sys
from collections.abc import Iterator
from pathlib import Path

import pandas as pd
import pytest

from bdbvhubrenewal.cli import main
from bdbvhubrenewal.evaluation.germany import load_germany_config
from bdbvhubrenewal.reporting import plots

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "tests" / "reporting" / "fixtures"
DATA = ROOT / "tests" / "fixtures" / "data"


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
    for f in DATA.iterdir():
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
            "--data",
            str(DATA),
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


def test_host_devices_takes_effect_after_importing_the_package() -> None:
    # Importing the package must not start JAX, or the device count set
    # by `host_devices` before a fit comes too late and chains run one
    # after another.
    code = (
        "from bdbvhubrenewal.cli import host_devices\n"
        "host_devices(3)\n"
        "import jax\n"
        "print(jax.local_device_count())\n"
    )
    env = {k: v for k, v in os.environ.items() if k != "XLA_FLAGS"}
    out = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        check=True,
        env=env,
    )
    assert out.stdout.strip().splitlines()[-1] == "3"


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
        data=str(DATA),
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

    args = argparse.Namespace(data=str(DATA), results=str(tmp_path))
    # With no fits loaded and no hub files there are no stored samples or
    # delays, and the empty samples are still numeric.
    empty, none = _stored(args, load_target(DATA), run_forecasts([], []))
    assert none.empty
    assert empty.empty
    assert empty["value"].dtype == np.float64
    # Hub files in the data directory: our 2026-11-07 submission.
    data = tmp_path / "data"
    data.mkdir()
    for f in DATA.iterdir():
        (data / f.name).symlink_to(f)
    sub = ROOT / "data" / "our-submissions" / "model-output" / "seabbs-renewal"
    hub = data / "hub-submissions" / "model-output" / "seabbs-renewal"
    hub.mkdir(parents=True)
    for f in sub.iterdir():
        (hub / f.name).symlink_to(f)
    args.data = str(data)
    learnt = pd.DataFrame(
        {"model": ["renewal", "randomwalknowcast"], "vintage": "2026-10-24"}
    )
    learnt.to_csv(tmp_path / DELAY_SUMMARY_FILE, index=False)
    stored, delays = _stored(args, load_target(DATA), run_forecasts([], []))
    # Only the fitted models' delays are stored ones.
    assert set(delays["model"]) == {"renewal"}
    assert not stored.empty
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
    args = ["backtest", "--data", str(DATA), "--models", "pprandomwalknowcast"]
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
    args = [
        "backtest",
        "--data",
        str(DATA),
        "--scenarios-only",
        "--models",
        "randomwalk",
    ]
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
    args = [
        "backtest",
        "--data",
        str(DATA),
        "--recovery-only",
        "--models",
        "randomwalk",
        *out,
    ]
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
    args = [
        "backtest",
        "--data",
        str(DATA),
        "--no-recovery",
        "--models",
        "randomwalk",
        *out,
    ]
    assert main(args) == 0


def test_concat_results_keeps_one_copy_of_shared_tables(
    tmp_path: Path,
) -> None:
    from bdbvhubrenewal.data.tidy import write_csv

    parts = [tmp_path / "a", tmp_path / "b"]
    for part in parts:
        part.mkdir()
        write_csv(pd.DataFrame({"truth": [1, 2]}), part / "truth.csv")
        write_csv(pd.DataFrame({"x": [1]}), part / "scores.csv")
    out = tmp_path / "results"
    args = ["concat-results", "--out", str(out), "--dedupe", "truth.csv"]
    assert main([*args, "--", *map(str, parts)]) == 0
    assert len(pd.read_csv(out / "truth.csv")) == 2
    assert len(pd.read_csv(out / "scores.csv")) == 2


@pytest.mark.usefixtures("in_root")
def test_zone_backtest_of_a_skipped_block_writes_empty_files(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    def nothing(*args: object, **kwargs: object) -> tuple[pd.DataFrame, ...]:
        del args, kwargs
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    monkeypatch.setattr(
        "bdbvhubrenewal.evaluation.zones.zone_backtest", nothing
    )
    names = ["zone-scores.csv", "zone-energy.csv", "zone-quantiles.csv"]
    outs = [
        "--out", str(tmp_path / names[0]),
        "--energy-out", str(tmp_path / names[1]),
        "--quantiles-out", str(tmp_path / names[2]),
        "--draws-out", str(tmp_path / "zone-draws.csv.gz"),
    ]  # fmt: skip
    assert main(["zone-backtest", "--chunk", "0/3", *outs]) == 0
    assert all((tmp_path / n).exists() for n in names)


def test_concat_results_skips_empty_parts(tmp_path: Path) -> None:
    from bdbvhubrenewal.data.tidy import write_csv

    parts = [tmp_path / "a", tmp_path / "b"]
    for part in parts:
        part.mkdir()
    write_csv(pd.DataFrame(), parts[0] / "zone-scores.csv")
    write_csv(pd.DataFrame({"x": [1]}), parts[1] / "zone-scores.csv")
    out = tmp_path / "results"
    assert main(["concat-results", "--out", str(out), *map(str, parts)]) == 0
    assert len(pd.read_csv(out / "zone-scores.csv")) == 1


def test_digest_of_one_model(capsys: pytest.CaptureFixture[str]) -> None:
    from bdbvhubrenewal.evaluation.digest import model_digest

    assert main(["digest", "--root", str(ROOT), "--model", "randomwalk"]) == 0
    key = capsys.readouterr().out.strip()
    assert key == model_digest("randomwalk", ROOT)
    assert key != model_digest("pprandomwalknowcast", ROOT)


def test_fit_store_plan_takes_origins(tmp_path: Path) -> None:
    from bdbvhubrenewal.cli import main

    out, keys = tmp_path / "matrix.json", tmp_path / "keys.json"
    rc = main(
        [
            "fit-store",
            "plan",
            "--store",
            str(tmp_path / "store"),
            "--models",
            "renewal",
            "--origins",
            "2026-11-08",
            "--out",
            str(out),
            "--keys",
            str(keys),
        ]
    )
    assert rc == 0
    planned = json.loads(out.read_text())["include"]
    assert [(p["model"], p["origin"]) for p in planned] == [
        ("renewal", "2026-11-08")
    ]
    assert [k.rsplit("/", 1)[0] for k in json.loads(keys.read_text())] == [
        "renewal/2026-11-08"
    ]


def test_fit_store_unpack_says_when_the_store_has_no_fit(
    tmp_path: Path,
) -> None:
    from bdbvhubrenewal.cli import main

    args = ["fit-store", "unpack", "--store", str(tmp_path / "store")]
    args += ["--model", "renewal", "--origin", "2026-11-08"]
    assert main([*args, "--out", str(tmp_path / "fit")]) == 3
    assert not (tmp_path / "fit").exists()


def _germany_data(tmp_path: Path) -> tuple[Path, Path]:
    """The fixture as the hub's triangle, and a config for it."""
    long = pd.read_csv(
        ROOT / "tests" / "fixtures" / "germany" / "germany_covid19_hosp.csv.gz"
    )
    keys = ["location", "age_group", "reference_date"]
    long = long.sort_values([*keys, "report_date"])
    long["change"] = (
        long.groupby(keys)["confirm"].diff().fillna(long["confirm"])
    )
    long["delay"] = (
        pd.to_datetime(long["report_date"])
        - pd.to_datetime(long["reference_date"])
    ).dt.days
    wide = long.pivot_table(
        index=keys, columns="delay", values="change"
    ).rename(columns=lambda d: f"value_{d}d")
    data = tmp_path / "germany"
    data.mkdir()
    wide.reset_index().rename(columns={"reference_date": "date"}).to_csv(
        data / "COVID-19_hospitalizations.csv", index=False
    )
    text = (ROOT / "config" / "germany.toml").read_text()
    text = text.replace('"2021-11-22"', '"2021-10-04"')
    text = text.replace('"2022-04-29"', '"2021-10-05"')
    text = text.replace("delay = 40", "delay = 10")
    text = text.replace('"2022-02-10"', '"2021-10-06"')
    config = tmp_path / "germany.toml"
    config.write_text(text)
    return data, config


def test_cli_germany_backtest_scores_a_block_of_dates(tmp_path: Path) -> None:
    data, config = _germany_data(tmp_path)
    out = tmp_path / "results"
    assert main(["germany-backtest", "--config", str(config), "--data",
                 str(data), "--models", "randomwalk", "--chunk", "1/2",
                 "--out", str(out)]) == 0  # fmt: skip
    scores = pd.read_csv(out / "scores.csv.gz")
    assert set(scores["origin"]) == {"2021-10-05"}
    assert set(scores["model"]) == {"randomwalk"}
    forecasts = pd.read_csv(out / "forecasts.csv.gz")
    assert set(forecasts["origin"]) == {"2021-10-05"}


def test_cli_germany_teams_scores_the_hub_teams(tmp_path: Path) -> None:
    data, config = _germany_data(tmp_path)
    rows = [
        {
            "location": "DE",
            "age_group": "00+",
            "forecast_date": "2021-10-04",
            "target_end_date": "2021-10-04",
            "target": "0 day ahead inc hosp",
            "type": "quantile",
            "quantile": q,
            "value": 100.0 * (i + 1),
            "pathogen": "COVID-19",
        }
        for i, q in enumerate((0.025, 0.1, 0.25, 0.5, 0.75, 0.9, 0.975))
    ]
    names = load_germany_config(config).submissions
    models = ("KIT-simple_nowcast", "KIT-frozen_baseline")
    for name, model in zip(names, models, strict=True):
        pd.DataFrame(rows).assign(model=model).to_csv(
            data / Path(name).name, index=False
        )
    out = tmp_path / "results"
    assert main(["germany-teams", "--config", str(config), "--data",
                 str(data), "--out", str(out)]) == 0  # fmt: skip
    teams = pd.read_csv(out / "teams.csv.gz")
    assert set(teams["model"]) == {"KIT", "KIT-frozen_baseline"}
    assert len(teams) == 2
    assert (out / "teams-forecasts.csv.gz").exists()
    assert main(["germany-paper", "--config", str(config), "--results",
                 str(out)]) == 0  # fmt: skip
    paper = pd.read_csv(out / "paper-forecasts.csv.gz")
    assert set(paper["model"]) == {"KIT", "KIT-frozen_baseline"}
    meta = json.loads((out / "paper-forecasts.json").read_text())
    assert meta["split_date"] == "2021-10-06"
    assert meta["baseline"] == "KIT-frozen_baseline"


def _outside_repo(tmp_path: Path) -> tuple[Path, dict[str, str]]:
    """Return a working directory with the config and no `data/`."""
    import shutil

    shutil.copytree(ROOT / "config", tmp_path / "config")
    env = {**os.environ, "BDBV_DATA_DIR": str(ROOT / "tests/fixtures/data")}
    return tmp_path, env


def test_loaders_follow_bdbv_data_dir_after_the_cli_starts(
    tmp_path: Path,
) -> None:
    # A new process: the conftest sets this one's loader default itself.
    cwd, env = _outside_repo(tmp_path)
    code = (
        "from bdbvhubrenewal.cli import main\n"
        "from bdbvhubrenewal.data import load_locations\n"
        "main(['--version'])\n"
        "print(len(load_locations()))\n"
    )
    out = subprocess.run(
        [sys.executable, "-c", code],
        cwd=cwd,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert out.returncode == 0, out.stderr
    assert int(out.stdout.split()[-1]) > 0


@pytest.mark.slow
def test_the_smoke_fit_runs_with_no_data_dir(tmp_path: Path) -> None:
    cwd, env = _outside_repo(tmp_path)
    sampler = '{"num_chains": 1, "num_warmup": 5, "num_samples": 5}'
    out = subprocess.run(
        [
            sys.executable, "-m", "bdbvhubrenewal.cli", "fit",
            "--model", "renewal", "--config", "main",
            "--sampler-json", sampler, "--out", str(tmp_path / "fit"),
        ],
        cwd=cwd,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )  # fmt: skip
    assert out.returncode == 0, out.stderr[-2000:]
    assert (tmp_path / "fit").is_dir()


@pytest.mark.parametrize(
    ("argv", "attr", "path"),
    [
        (["backtest"], "data", ""),
        (["compare-hub"], "data", ""),
        (["compare-hub"], "hub_dir", "/hub-submissions"),
    ],
)
def test_data_defaults_follow_bdbv_data_dir(
    monkeypatch: pytest.MonkeyPatch, argv: list[str], attr: str, path: str
) -> None:
    from bdbvhubrenewal.cli import build_parser

    monkeypatch.setenv("BDBV_DATA_DIR", "elsewhere")
    args = build_parser().parse_args(argv)
    assert getattr(args, attr) == f"elsewhere{path}"
