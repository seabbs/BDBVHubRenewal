from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from bdbvhubrenewal.data import load_daily, load_delays, load_target, tidy
from bdbvhubrenewal.evaluation.backtest import BacktestConfig
from bdbvhubrenewal.forecasters import ForecastTask

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "tests" / "fixtures" / "data"
"""A frozen copy of the tidy hub data. It is never refreshed: the
workflows fetch the live data, and the tests read only this copy."""

FIXTURE_VINTAGES = (
    "2026-08-18",
    "2026-10-09",
    "2026-10-17",
    "2026-10-24",
    "2026-11-01",
    "2026-11-08",
)
"""The real vintages in the fixture, oldest first."""

# The loaders and the CLI read `data/` by default. Point both at the
# fixture before any test builds a parser or loads a table.
os.environ["BDBV_DATA_DIR"] = str(DATA)
tidy._DEFAULT_DATA_DIR = DATA


FAKE_CHECKPOINT = "fakefake0001"
"""Settings hash of the fake latest fine-tuned checkpoint release."""


def _write_checkpoint_release(
    root: Path, tag_hash: str, adapter: dict[str, bytes], settings_hash: str
) -> None:
    """Write a fake release `chronos-ft-<tag_hash>` under `root`.

    Its assets are the adapter files, `settings.json` and their
    `sha256sums.txt`, as `scripts/finetune_chronos.py train` writes them.
    """
    import hashlib
    import json

    assets = root / f"chronos-ft-{tag_hash}"
    assets.mkdir(parents=True)
    files = {**adapter, "settings.json": json.dumps({"hash": settings_hash})}
    for name, data in files.items():
        raw = data.encode() if isinstance(data, str) else data
        (assets / name).write_bytes(raw)
    (assets / "sha256sums.txt").write_text(
        "".join(
            f"{hashlib.sha256((assets / n).read_bytes()).hexdigest()}  {n}\n"
            for n in files
        )
    )


@pytest.fixture(autouse=True)
def _fake_checkpoint_releases(
    tmp_path_factory: pytest.TempPathFactory, monkeypatch: pytest.MonkeyPatch
) -> Path:
    """Resolve the fine-tuned checkpoint from a fake release list.

    No test reaches GitHub. The latest `chronos-ft-*` release is
    `FAKE_CHECKPOINT`, with its `sha256sums.txt` only.
    """
    from bdbvhubrenewal.forecasters import foundation

    root = tmp_path_factory.getbasetemp() / "checkpoint-releases"
    if not root.exists():
        _write_checkpoint_release(
            root, FAKE_CHECKPOINT, {"adapter_model.safetensors": b"w",
                                    "adapter_config.json": b"{}"},
            FAKE_CHECKPOINT,
        )  # fmt: skip
        for name in ("adapter_model.safetensors", "adapter_config.json"):
            (root / f"chronos-ft-{FAKE_CHECKPOINT}" / name).unlink()
    monkeypatch.setattr(foundation, "FINETUNED_URL", root.as_uri())
    monkeypatch.setattr(
        foundation,
        "finetuned_releases",
        lambda: [
            {
                "tag_name": f"chronos-ft-{FAKE_CHECKPOINT}",
                "published_at": "2026-11-13T12:00:00Z",
                "draft": False,
            }
        ],
    )
    foundation.resolve_checkpoint.cache_clear()
    yield root
    foundation.resolve_checkpoint.cache_clear()


@pytest.fixture
def write_release():  # type: ignore[no-untyped-def]
    """Write a fake checkpoint release (`_write_checkpoint_release`)."""
    return _write_checkpoint_release


@pytest.fixture
def fake_checkpoint() -> str:
    """Settings hash of the fake latest checkpoint release."""
    return FAKE_CHECKPOINT


@pytest.fixture(autouse=True)
def _one_province_at_a_time(monkeypatch: pytest.MonkeyPatch) -> None:
    """Fit per-province variants in this process, where stand-ins reach.

    Tests that check the worker processes set the variable themselves.
    """
    monkeypatch.setenv("BDBV_PROVINCE_WORKERS", "1")


@pytest.fixture(scope="session")
def target() -> pd.DataFrame:
    return load_target(DATA)


@pytest.fixture(scope="session")
def vintages() -> list[str]:
    """`FIXTURE_VINTAGES`, for tests that pin expectations to them."""
    return list(FIXTURE_VINTAGES)


@pytest.fixture(scope="session")
def delays() -> pd.DataFrame:
    return load_delays(DATA)


@pytest.fixture(scope="session")
def daily() -> pd.DataFrame:
    return load_daily(DATA)


@pytest.fixture(scope="session")
def tasks_json() -> Path:
    return DATA / "hub-config" / "tasks.json"


@pytest.fixture(scope="session")
def schema_json() -> Path:
    return DATA / "hub-config" / "model-metadata-schema.json"


@pytest.fixture
def config() -> BacktestConfig:
    return BacktestConfig(
        n_samples=100,
        seed=1,
        simulated_dates=("2026-08-01", "2026-09-05"),
    )


@pytest.fixture(scope="session")
def daily_zones() -> pd.DataFrame:
    from bdbvhubrenewal.data import load_daily_zones

    return load_daily_zones(DATA)


class FlatModel:
    """Latest reported week carried forward, a stand-in for harness tests.

    The random walk models need release snapshots and cannot forecast
    most origins, so tests of the backtest, scoring and report machinery
    run this model instead.
    """

    name = "flat"
    supports_mask = False

    def forecast(self, task: ForecastTask) -> pd.DataFrame:
        rng = task.rng()
        weeks = list(task.target_end_dates)
        frames = []
        for loc in task.locations:
            series = (
                task.target.loc[task.target["location"] == loc]
                .set_index("target_end_date")["total"]
                .sort_index()
            )
            level = float(series.iloc[-1]) if len(series) else 0.0
            values = np.column_stack(
                [
                    np.full(task.n_samples, float(series[w]))
                    if w in series.index
                    else rng.poisson(level, task.n_samples).astype(float)
                    for w in weeks
                ]
            )
            frames.append(
                pd.DataFrame(
                    {
                        "location": loc,
                        "target_end_date": np.tile(weeks, task.n_samples),
                        "sample": np.repeat(
                            np.arange(1, task.n_samples + 1), len(weeks)
                        ),
                        "value": values.reshape(-1),
                    }
                )
            )
        final = pd.concat(frames, ignore_index=True)
        if not task.report_by:
            return final
        # Counts reported by a later date are the final counts to then.
        later = [
            final.loc[final["target_end_date"] <= date].assign(as_of=date)
            for date in task.report_by
        ]
        return pd.concat(
            [final.assign(as_of=np.nan), *later], ignore_index=True
        )


@pytest.fixture
def flat_model(monkeypatch: pytest.MonkeyPatch) -> str:
    """Register `FlatModel` as `flat` for one test and return its name."""
    from bdbvhubrenewal.forecasters import base

    monkeypatch.setitem(
        base._REGISTRY,
        "flat",
        base.ModelSpec("flat", FlatModel, "Test stand-in."),
    )
    return "flat"
