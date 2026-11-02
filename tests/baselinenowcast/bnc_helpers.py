"""Helpers for the baselinenowcast tests: R fixtures and cases."""

from __future__ import annotations

from pathlib import Path
from typing import NamedTuple

import numpy as np
import pandas as pd

FIXTURES = Path(__file__).parent / "fixtures"
CASES = ["toy", "example", "neg", "struct2", "hub"]
SMALL = ["toy", "example", "neg", "struct2"]


def read(case: str, name: str) -> np.ndarray:
    """Read a fixture matrix, with NaN for NA."""
    path = FIXTURES / case / f"{name}.csv"
    return pd.read_csv(path, header=None, na_values="NA").to_numpy(float)


def read_vec(case: str, name: str) -> np.ndarray:
    return read(case, name).ravel()


def read_ints(case: str, name: str) -> list[int]:
    path = FIXTURES / case / f"{name}.txt"
    return [int(float(x)) for x in path.read_text().split()]


class Case(NamedTuple):
    """A fixture triangle with its settings."""

    name: str
    values: np.ndarray
    mask: np.ndarray
    n_hist: int
    n_retro: int
    k: int


def load_case(name: str) -> Case:
    tri = read(name, "tri")
    mask = ~np.isnan(tri)
    n_hist, n_retro, k = read_ints(name, "settings")
    return Case(name, np.nan_to_num(tri), mask, n_hist, n_retro, k)


def from_na(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Split an R matrix with NA into values (0 under NA) and mask."""
    m = ~np.isnan(x)
    return np.nan_to_num(x), m
