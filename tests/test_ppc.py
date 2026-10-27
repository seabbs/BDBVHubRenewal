from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
from matplotlib.figure import Figure

from bdbv_hub_renewal.ppc import (
    not_available,
    plot_ppc,
    read_ppc,
    summarise_draws,
    write_ppc,
)

AS_OF = "2026-10-31"


def observed_table(weeks: int = 12) -> pd.DataFrame:
    dates = pd.date_range(end=AS_OF, periods=weeks, freq="7D")
    rows = [
        {
            "location": loc,
            "stream": stream,
            "target_end_date": d.date().isoformat(),
            "observed": 0,
        }
        for loc in ("CD-IT", "CD-NK", "CD-HU")
        for stream in ("observed", "imputed", "total")
        for d in dates
    ]
    return pd.DataFrame(rows)


def calibrated(
    n_draws: int = 400, seed: int = 3
) -> tuple[pd.DataFrame, np.ndarray[Any, Any]]:
    rng = np.random.default_rng(seed)
    obs = observed_table()
    rate = rng.uniform(2, 60, size=len(obs))
    obs["observed"] = rng.poisson(rate)
    return obs, rng.poisson(rate, size=(n_draws, len(obs)))


def test_rows_have_the_schema() -> None:
    obs, draws = calibrated()
    ppc = summarise_draws(obs, draws, as_of=AS_OF, config="fallback")
    assert ppc["schema_version"] == 1
    assert ppc["available"] is True
    assert ppc["model"] == "renewal"
    assert ppc["config"] == "fallback"
    assert ppc["intervals"] == [0.5, 0.9]
    assert len(ppc["rows"]) == len(obs)
    row = ppc["rows"][0]
    assert set(row) == {
        "location",
        "stream",
        "target_end_date",
        "observed",
        "median",
        "lower_50",
        "upper_50",
        "lower_90",
        "upper_90",
        "pit",
        "complete",
    }
    for r in ppc["rows"]:
        assert r["lower_90"] <= r["lower_50"] <= r["median"]
        assert r["median"] <= r["upper_50"] <= r["upper_90"]
        assert 0 <= r["pit"] <= 1
        assert isinstance(r["observed"], int)
    json.dumps(ppc)  # serialisable as is


def test_calibrated_draws_cover_nominally() -> None:
    obs, draws = calibrated()
    rows = pd.DataFrame(summarise_draws(obs, draws, as_of=AS_OF)["rows"])
    inside90 = rows["observed"].between(rows["lower_90"], rows["upper_90"])
    inside50 = rows["observed"].between(rows["lower_50"], rows["upper_50"])
    # Discrete counts make the intervals a little conservative.
    assert 0.85 <= inside90.mean() <= 1.0
    assert 0.45 <= inside50.mean() <= 0.75
    pit = np.sort(rows["pit"].to_numpy())
    ks = np.max(np.abs(pit - (np.arange(len(pit)) + 0.5) / len(pit)))
    assert ks < 0.12


def test_completeness_follows_the_lag() -> None:
    obs, draws = calibrated()
    rows = summarise_draws(obs, draws, as_of=AS_OF, complete_lag_weeks=4)[
        "rows"
    ]
    done = {r["target_end_date"]: r["complete"] for r in rows}
    assert done["2026-10-03"] is True
    assert done["2026-10-10"] is False
    assert done[AS_OF] is False


def test_randomised_pit_is_seeded_and_correct_at_the_edges() -> None:
    obs = observed_table(weeks=1).iloc[:3].assign(observed=[5, 5, 5])
    draws = np.column_stack([np.full(50, 9), np.full(50, 1), np.full(50, 5)])
    a = summarise_draws(obs, draws, as_of=AS_OF, seed=7)["rows"]
    b = summarise_draws(obs, draws, as_of=AS_OF, seed=7)["rows"]
    assert [r["pit"] for r in a] == [r["pit"] for r in b]
    assert a[0]["pit"] == 0.0  # every draw above the observation
    assert a[1]["pit"] == 1.0  # every draw below it
    assert 0 < a[2]["pit"] < 1  # a point mass at it: uniform


def test_other_interval_levels() -> None:
    obs, draws = calibrated()
    ppc = summarise_draws(obs, draws, as_of=AS_OF, intervals=(0.8,))
    assert "lower_80" in ppc["rows"][0]
    assert "lower_90" not in ppc["rows"][0]


@pytest.mark.parametrize(
    ("change", "match"),
    [
        (lambda o, d: (o.drop(columns="stream"), d), "lacks columns"),
        (lambda o, d: (o.assign(stream="x"), d), "unknown streams"),
        (lambda o, d: (o, d[:, :-1]), "shape"),
    ],
)
def test_bad_inputs_raise(change: Any, match: str) -> None:
    obs, draws = calibrated()
    o, d = change(obs, draws)
    with pytest.raises(ValueError, match=match):
        summarise_draws(o, d, as_of=AS_OF)


def test_not_available_and_round_trip(tmp_path: Path) -> None:
    na = not_available("persistence has no posterior", "persistence")
    assert na == {
        "schema_version": 1,
        "available": False,
        "reason": "persistence has no posterior",
        "model": "persistence",
    }
    assert read_ppc(write_ppc(na, tmp_path / "ppc.json")) == na
    obs, draws = calibrated()
    ppc = summarise_draws(obs, draws, as_of=AS_OF)
    assert read_ppc(write_ppc(ppc, tmp_path / "f" / "ppc.json")) == ppc


@pytest.mark.parametrize(
    "payload",
    [[1], {"available": True, "rows": []}, {"schema_version": 1}],
)
def test_read_ppc_rejects_bad_files(tmp_path: Path, payload: Any) -> None:
    path = tmp_path / "ppc.json"
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match=r"ppc\.json"):
        read_ppc(path)


def test_plot_ppc_small_multiples(tmp_path: Path) -> None:
    obs, draws = calibrated()
    fig = plot_ppc(summarise_draws(obs, draws, as_of=AS_OF))
    assert isinstance(fig, Figure)
    visible = [ax for ax in fig.axes if ax.get_visible()]
    assert len(visible) == 3
    assert visible[0].get_title().startswith("CD-")
    assert "90%" in (visible[0].get_title())
    out = tmp_path / "ppc.png"
    fig.savefig(out, dpi=150)
    assert 0 < out.stat().st_size < 1_000_000


def test_plot_ppc_refuses_unavailable() -> None:
    with pytest.raises(ValueError, match="no posterior"):
        plot_ppc(not_available("no posterior here"))
