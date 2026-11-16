"""The public entry point to the fine-tuned Chronos-2 (`chronos_ft`)."""

from __future__ import annotations

import hashlib
import runpy
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from bdbvhubrenewal import chronos_ft
from bdbvhubrenewal.forecasters import foundation

_ADAPTER = {
    "adapter_config.json": b"{}",
    "adapter_model.safetensors": b"weights",
}

EXAMPLE = Path(__file__).parents[1] / "docs" / "examples" / "chronos_ft.py"


@pytest.fixture
def releases(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, write_release
) -> Path:
    """Fake releases `chronos-ft-old` and `chronos-ft-new`, the newest."""
    root = tmp_path / "releases"
    root.mkdir()
    for tag_hash in ("old", "new"):
        write_release(root, tag_hash, _ADAPTER, tag_hash)
    monkeypatch.setattr(foundation, "FINETUNED_URL", root.as_uri())
    monkeypatch.setattr(foundation, "FINETUNED_CACHE", str(tmp_path / "cache"))
    listing = [
        {"tag_name": t, "published_at": at, "draft": False}
        for t, at in (
            ("chronos-ft-old", "2026-11-01T00:00:00Z"),
            ("germany-ft-later", "2026-11-20T00:00:00Z"),
            ("chronos-ft-new", "2026-11-12T00:00:00Z"),
        )
    ]
    monkeypatch.setattr(foundation, "finetuned_releases", lambda: listing)
    foundation.resolve_checkpoint.cache_clear()
    yield root
    foundation.resolve_checkpoint.cache_clear()


@pytest.mark.usefixtures("releases")
def test_load_takes_the_newest_chronos_ft_release(tmp_path: Path) -> None:
    model = chronos_ft.load()
    assert model.checkpoint.hash == "new"
    assert model.adapter == tmp_path / "cache" / "chronos-ft-new"
    sums = dict(model.checkpoint.sha256)
    for name, data in _ADAPTER.items():
        assert (model.adapter / name).read_bytes() == data
        assert sums[name] == hashlib.sha256(data).hexdigest()


@pytest.mark.usefixtures("releases")
def test_load_takes_a_pinned_hash() -> None:
    assert chronos_ft.load("old").checkpoint.hash == "old"


def test_load_refuses_an_adapter_that_fails_its_sha256(
    releases: Path, tmp_path: Path
) -> None:
    (releases / "chronos-ft-new" / "adapter_model.safetensors").write_bytes(
        b"tampered"
    )
    with pytest.raises(ValueError, match="SHA-256"):
        chronos_ft.load("new")
    assert not (tmp_path / "cache" / "chronos-ft-new").exists()


class _FakePipeline:
    """Records what it is given and returns fixed quantiles."""

    quantiles = (0.1, 0.5, 0.9)

    def __init__(self, value: float = 0.0) -> None:
        self.value = value
        self.calls: list[dict] = []

    def predict_quantiles(self, inputs, prediction_length, quantile_levels):
        self.calls.append(
            {
                "inputs": inputs,
                "prediction_length": prediction_length,
                "quantile_levels": list(quantile_levels),
            }
        )
        out = [
            np.full(
                (len(g["target"]), prediction_length, len(quantile_levels)),
                self.value,
            )
            + np.arange(len(quantile_levels))
            for g in inputs
        ]
        return out, None


@pytest.fixture
def fake(monkeypatch: pytest.MonkeyPatch) -> _FakePipeline:
    pipe = _FakePipeline()
    monkeypatch.setattr(chronos_ft, "_pipeline", lambda _checkpoint: pipe)
    return pipe


def _triangle(n_rows: int = 6, n_delays: int = 3) -> pd.DataFrame:
    """A reporting triangle: row t is known up to delay n_rows - 1 - t."""
    counts = np.arange(1, n_rows * n_delays + 1, dtype=float).reshape(
        n_rows, n_delays
    )
    t, d = np.indices(counts.shape)
    counts[t + d > n_rows - 1] = np.nan
    return pd.DataFrame(
        counts,
        index=pd.date_range("2026-01-03", periods=n_rows, freq="7D"),
        columns=range(n_delays),
    )


def test_the_triangle_is_laid_out_as_the_ragged_block(
    fake: _FakePipeline,
) -> None:
    tri = _triangle()
    chronos_ft.forecast_triangle(tri, horizon=2, model=None)
    (call,) = fake.calls
    (group,) = call["inputs"]
    counts = tri.to_numpy()
    # The rows reported at every delay are the context.
    np.testing.assert_array_equal(group["target"], [counts[:4].sum(axis=1)])
    # The rest are nowcast (delays - 1 rows) then forecast (horizon rows).
    assert call["prediction_length"] == 2 + 2
    lag1 = group["past_covariates"]["series_lag_1"]
    np.testing.assert_array_equal(lag1, counts[:4, :2].sum(axis=1))
    # Row 4 is known at delay 1, row 5 is not; future rows never are.
    future = group["future_covariates"]["series_lag_1"]
    assert future[0] == counts[4, :2].sum()
    assert np.isnan(future[1:]).all()
    assert set(group["past_covariates"]) == {
        f"series_lag_{d}" for d in range(3)
    }


@pytest.mark.usefixtures("fake")
def test_forecasts_are_quantiles_of_the_final_count_at_least_reported() -> None:
    tri = _triangle()
    out = chronos_ft.forecast_triangle(
        tri, horizon=2, model=None, quantile_levels=(0.1, 0.5, 0.9)
    )
    assert list(out.columns) == [
        "place",
        "reference",
        "step",
        "quantile_level",
        "value",
    ]
    assert len(out) == 4 * 3
    assert sorted(set(out["step"])) == [-1, 0, 1, 2]
    assert out["reference"].max() == tri.index[-1] + 2 * pd.Timedelta("7D")
    reported = np.nansum(tri.to_numpy(), axis=1)
    nowcast = out.loc[out["step"] <= 0]
    for _, row in nowcast.iterrows():
        at = tri.index.get_loc(row["reference"])
        assert row["value"] >= reported[at]
    # The fake gives 0, 1, 2 by level, and nothing is reported ahead.
    future = out.loc[out["step"] == 2].sort_values("quantile_level")
    np.testing.assert_array_equal(future["value"], [0.0, 1.0, 2.0])


def test_places_share_one_block(fake: _FakePipeline) -> None:
    tri = _triangle()
    out = chronos_ft.forecast_triangle(
        {"north": tri, "south": tri * 2}, horizon=1, model=None
    )
    (group,) = fake.calls[0]["inputs"]
    assert group["target"].shape == (2, 4)
    assert "south_lag_2" in group["past_covariates"]
    assert set(out["place"]) == {"north", "south"}


@pytest.mark.usefixtures("fake")
def test_a_gap_inside_the_reported_part_fails() -> None:
    tri = _triangle()
    tri.iloc[1, 1] = np.nan
    with pytest.raises(ValueError, match="missing"):
        chronos_ft.forecast_triangle(tri, model=None)


@pytest.mark.usefixtures("fake")
def test_too_short_a_triangle_fails() -> None:
    with pytest.raises(ValueError, match="two rows"):
        chronos_ft.forecast_triangle(_triangle(3, 3), model=None)


@pytest.mark.usefixtures("releases", "fake")
def test_the_docs_example_runs() -> None:
    ns = runpy.run_path(str(EXAMPLE), run_name="example")
    out = ns["forecast"]
    assert {"place", "reference", "step", "quantile_level", "value"} <= set(
        out.columns
    )
