"""Tests of scripts/bdbv_ft_store.py's pure parts."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "bdbv_ft_store.py"


@pytest.fixture(scope="module")
def store():
    spec = importlib.util.spec_from_file_location("bdbv_ft_store", SCRIPT)
    assert spec is not None
    assert spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules["bdbv_ft_store"] = mod
    spec.loader.exec_module(mod)
    return mod


def test_draws_become_hub_quantiles_with_truth(store) -> None:
    draws = pd.DataFrame(
        {
            "model": "m",
            "origin": "2026-10-17",
            "location": "CD",
            "target_end_date": "2026-10-10",
            "sample": np.arange(1, 1001),
            "value": np.arange(1000, dtype=float),
        }
    )
    truth = pd.DataFrame(
        {
            "location": ["CD"],
            "target_end_date": ["2026-10-10"],
            "truth": [500.0],
            "truth_vintage": ["2026-11-07"],
        }
    )
    out = store.bdbv_quantiles(draws, truth, (0.05, 0.5, 0.95))
    assert out["quantile"].tolist() == [0.05, 0.5, 0.95]
    assert out["value"].round(1).tolist() == [49.9, 499.5, 949.0]
    assert (out["truth"] == 500.0).all()
    assert set(out.columns) >= {
        "model",
        "origin",
        "location",
        "target_end_date",
        "quantile",
        "value",
        "truth",
        "truth_vintage",
    }


def test_weeks_with_no_truth_yet_keep_an_empty_truth(store) -> None:
    draws = pd.DataFrame(
        {
            "model": "m",
            "origin": "2026-11-15",
            "location": "CD",
            "target_end_date": "2026-11-21",
            "sample": [1, 2],
            "value": [1.0, 3.0],
        }
    )
    truth = pd.DataFrame(
        columns=["location", "target_end_date", "truth", "truth_vintage"]
    )
    out = store.bdbv_quantiles(draws, truth, (0.5,))
    assert out["value"].tolist() == [2.0]
    assert out["truth"].isna().all()
