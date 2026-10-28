"""Check the port against the R package baselinenowcast.

The fixtures in tests/data/baselinenowcast are made by
scripts/baselinenowcast_fixtures.R with the version pinned in VERSION.
The baselinenowcast fixtures workflow regenerates them and fails on a
diff. These tests need no R.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from bdbvhubrenewal.data.triangle import (
    apply_delay,
    delay_pmf,
    fit_nb_size,
    retrospective_errors,
)

FIXTURES = Path(__file__).parent / "baselinenowcast"
SETTINGS = pd.read_csv(FIXTURES / "settings.csv")


def _triangle(name: str) -> np.ndarray:
    rows = pd.read_csv(FIXTURES / "triangles.csv")
    rows = rows.loc[rows["triangle"] == name]
    return rows.pivot(index="row", columns="delay", values="count").to_numpy()


def _expected(file: str, name: str, column: str) -> np.ndarray:
    rows = pd.read_csv(FIXTURES / file)
    return rows.loc[rows["triangle"] == name, column].to_numpy(dtype=float)


@pytest.mark.parametrize("name", SETTINGS["triangle"])
def test_delay_and_point_nowcast_match_baselinenowcast(name: str) -> None:
    n = int(SETTINGS.set_index("triangle").loc[name, "n_history_delay"])
    tri = _triangle(name)
    pmf = delay_pmf(tri[-n:])
    assert pmf == pytest.approx(_expected("pmf.csv", name, "pmf"), rel=1e-9)
    point = apply_delay(tri, pmf).sum(axis=1)
    want = _expected("point.csv", name, "point_final")
    assert point == pytest.approx(want, rel=1e-9)


@pytest.mark.parametrize("name", SETTINGS["triangle"])
def test_uncertainty_sizes_match_baselinenowcast(name: str) -> None:
    row = SETTINGS.set_index("triangle").loc[name]
    tri = _triangle(name)
    obs, pred = retrospective_errors(
        tri, int(row["n_retrospective_nowcasts"]), int(row["n_history_delay"])
    )
    sizes = fit_nb_size(obs, pred)
    # Two optimisers on a flat likelihood: agree to 0.1%.
    assert sizes == pytest.approx(_expected("nb_size.csv", name, "nb_size"),
                                  rel=1e-3)  # fmt: skip
