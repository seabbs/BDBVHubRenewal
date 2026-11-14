from __future__ import annotations

import ast
import dataclasses
import shutil
from pathlib import Path

import pandas as pd
import pytest

from bdbvhubrenewal.evaluation.backtest import BacktestConfig
from bdbvhubrenewal.evaluation.digest import REPORT_ONLY, backtest_digest

ROOT = Path(__file__).resolve().parents[2]


def _digest_tree(root: Path) -> dict[str, Path]:
    for name in ("pyproject.toml", "uv.lock"):
        shutil.copy(ROOT / name, root / name)
    files = {
        "code": root / "src/bdbvhubrenewal/evaluation/a.py",
        "report": root / "src/bdbvhubrenewal/reporting/report.py",
        "data": root / "data/target.csv",
        "sources": root / "data/sources.json",
        "raw": root / "data/raw/x.csv",
        "results": root / "results/forecasts.csv.gz",
    }
    for path in files.values():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("a = 1\n")
    return files


def test_backtest_digest_follows_what_a_backtest_reads(
    tmp_path: Path,
) -> None:
    files = _digest_tree(tmp_path)
    before = backtest_digest(tmp_path)
    # Report-only code, the download record and raw downloads are not read.
    for name in ("report", "sources", "raw", "results"):
        files[name].write_text("a = 2\n")
        assert backtest_digest(tmp_path) == before, name
    assert backtest_digest(tmp_path, text="other") != before
    assert backtest_digest(tmp_path, extra=["results"]) != before
    for name in ("code", "data"):
        files[name].write_text("a = 3\n")
        after = backtest_digest(tmp_path)
        assert after != before, name
        before = after


def _reporting_imports(path: Path) -> set[str]:
    found = set()
    for node in ast.walk(ast.parse(path.read_text())):
        names = []
        if isinstance(node, ast.ImportFrom) and node.module:
            names = [node.module]
        elif isinstance(node, ast.Import):
            names = [a.name for a in node.names]
        found |= {
            n.rsplit(".", 1)[1]
            for n in names
            if n.startswith("bdbvhubrenewal.reporting.")
        }
    return found


def test_backtest_code_never_imports_report_only_modules() -> None:
    src = ROOT / "src" / "bdbvhubrenewal"
    report_only = {Path(p).stem for p in REPORT_ONLY}
    assert report_only
    for path in sorted(src.rglob("*.py")):
        rel = path.relative_to(ROOT).as_posix()
        if rel in REPORT_ONLY:
            continue
        used = _reporting_imports(path) & report_only
        # The `report` command, which no backtest runs.
        if rel == "src/bdbvhubrenewal/cli.py":
            used -= {"report"}
        assert not used, f"{rel} imports {used}"


def _ran(run: object) -> set[str]:
    """Run a call and return the package files whose functions it ran."""
    import sys

    src = (ROOT / "src" / "bdbvhubrenewal").as_posix()
    seen: set[str] = set()

    def profile(frame, event, arg):  # type: ignore[no-untyped-def]
        del arg
        if event == "call" and frame.f_code.co_filename.startswith(src):
            seen.add(frame.f_code.co_filename)

    sys.setprofile(profile)
    try:
        run()  # type: ignore[operator]
    finally:
        sys.setprofile(None)
    return {Path(f).relative_to(ROOT).as_posix() for f in seen}


@pytest.mark.parametrize(
    "model", ["randomwalk", "randomwalknowcast", "pprandomwalknowcast"]
)
def test_model_code_covers_what_its_forecasts_run(
    model: str,
    target: pd.DataFrame,
    delays: pd.DataFrame,
    daily: pd.DataFrame,
    config: BacktestConfig,
) -> None:
    from bdbvhubrenewal.data import load_daily_zones, load_health_zones
    from bdbvhubrenewal.data.zones import hub_zones
    from bdbvhubrenewal.evaluation import backtest as bt
    from bdbvhubrenewal.evaluation.digest import model_code
    from bdbvhubrenewal.evaluation.nowcast import forecast_releases
    from bdbvhubrenewal.evaluation.zones import zone_forecasts
    from bdbvhubrenewal.forecasters import get_model

    data = ROOT / "tests" / "fixtures" / "data"
    real = target.loc[target["kind"] == "real"]
    tasks = bt.make_tasks(real, delays.assign(kind="real"), config, daily=daily)
    one = [
        dataclasses.replace(config, simulated_dates=config.simulated_dates[:1])
    ]
    calls = [
        lambda: forecast_releases(tasks, target, [model]),
        lambda: bt.recovery_forecasts([get_model(model)], one[0]),
        lambda: zone_forecasts(
            target, delays, daily, load_daily_zones(data),
            hub_zones(data / "hub-config" / "tasks.json"),
            models=[model], config=config,
            origins=bt.real_vintages(real)[-2:-1],
        ),
        lambda: bt.scenario_forecasts(
            [get_model(model)], config, target, delays,
            load_health_zones(data), "2026-10-24", ["2026-10-17"],
            names=["flat"],
        ),
    ]  # fmt: skip
    ran: set[str] = set()
    for call in calls:
        ran |= _ran(call)
    missing = ran - set(model_code(model, ROOT))
    assert not missing, f"{model} runs code its key leaves out: {missing}"


def test_model_code_leaves_out_other_models() -> None:
    from bdbvhubrenewal.evaluation.digest import model_code

    code = set(model_code("randomwalk", ROOT))
    assert "src/bdbvhubrenewal/forecasters/renewal.py" not in code
    assert "src/bdbvhubrenewal/forecasters/pprandomwalk.py" not in code
    assert "src/bdbvhubrenewal/forecasters/randomwalk.py" in code
    pp = set(model_code("pprandomwalknowcast", ROOT))
    assert "src/bdbvhubrenewal/forecasters/renewal.py" not in pp
    # It builds on the random walk module, so that is in its key.
    assert "src/bdbvhubrenewal/forecasters/randomwalk.py" in pp


@pytest.mark.slow
@pytest.mark.parametrize("model", ["renewal", "renewalsp", "renewalbvd"])
def test_fit_code_covers_what_a_stored_fit_runs(
    model: str,
    target: pd.DataFrame,
    delays: pd.DataFrame,
    daily: pd.DataFrame,
    config: BacktestConfig,
) -> None:
    from bdbvhubrenewal.evaluation import store

    tasks = store.origin_tasks(target, delays, config, daily)
    task = tasks[sorted(tasks)[-1]]
    if model == "renewalsp":
        task = dataclasses.replace(task, locations=task.locations[:2])
    tiny = {
        model: {
            "num_chains": 1,
            "num_warmup": 4,
            "num_samples": 4,
            "target_accept": 0.8,
            "max_tree_depth": 2,
            "seed": 1,
        }
    }
    out: list[store.StoredFit] = []
    ran = _ran(lambda: out.append(store.fit(model, task, tiny, "c")))
    assert out[0].meta["status"] == "ok", out[0].meta.get("error")
    missing = ran - set(store.fit_code(model, ROOT))
    assert not missing, f"{model} runs code its key leaves out: {missing}"
