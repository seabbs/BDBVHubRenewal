"""A stored fit's own counts as reported by a later release."""

from __future__ import annotations

import datetime as dt
from typing import Any

import numpy as np
import pytest

from bdbvhubrenewal.forecasters.reporting import (
    as_reported_samples,
    decode,
    draw_reported,
    encode,
)


def curves(
    n: int = 40, n_days: int = 30, n_w: int = 3, n_p: int = 2, seed: int = 1
) -> dict[str, Any]:
    """Reporting curves of a delay that reports evenly over 20 days."""
    rng = np.random.default_rng(seed)
    final = rng.gamma(5.0, 4.0, size=(n, n_w, n_p))
    share = np.minimum(np.arange(1, n_days + 1) / 20, 1.0)
    rest = np.concatenate(
        [share[None, :, None, None] * final[:, None], final[:, None]], axis=1
    )
    out: dict[str, Any] = {"cutoff": 100}
    for s in ("known", "imputed"):
        out |= {
            s: rest,
            f"vintage_{s}": np.where(
                np.arange(n_w)[:, None] < n_w - 1, 10.0, np.nan
            )
            * np.ones((n_w, n_p)),
            f"draw_{s}": rng.poisson(3.0, size=(n, n_w, n_p)).astype(float),
            f"phi_{s}": np.full(n, 1e6),
        }
    return out


def test_encoding_keeps_the_curves_to_1_255_of_the_final_count() -> None:
    c = curves()
    back = decode(encode(c))
    for s in ("known", "imputed"):
        final = c[s][:, -1:]
        assert np.all(np.abs(back[s] - c[s]) <= final / 255 / 2 + 1e-4)
        np.testing.assert_allclose(back[s][:, -1], c[s][:, -1], rtol=1e-6)
        np.testing.assert_array_equal(back[f"draw_{s}"], c[f"draw_{s}"])


def test_a_cut_off_past_the_curves_reports_everything() -> None:
    c = curves(n=400)
    rng = np.random.default_rng(2)
    late = draw_reported(c, 1000, rng)
    at_end = draw_reported(c, 30, rng)
    seen = np.where(np.isnan(c["vintage_known"]), 0.0, 20.0)
    both = c["draw_known"] + c["draw_imputed"]
    drawn = np.where(np.isnan(c["vintage_known"]), both, 0.0)
    expected = seen + drawn + 2 * c["known"][:, -1]
    # Concentrations of 1e6: the draws are Poisson about the mean.
    for got in (late, at_end):
        assert np.abs(got - expected).mean() < 0.15 * expected.mean()
    # The day of the cut-off reports nothing new. Unseen weeks keep each
    # trajectory's draw; imputed counts are redrawn about the vintage's.
    now = draw_reported(c, 0, rng)
    unseen = np.isnan(c["vintage_known"])
    np.testing.assert_array_equal(now[:, unseen], drawn[:, unseen])
    assert now[:, ~unseen].mean() == pytest.approx(20.0, rel=0.05)


def test_samples_keep_each_trajectory_s_place_and_week() -> None:
    c = curves(n=5)
    block = {
        "start_date": "2026-06-07",
        "places": ["CD-IT", "CD-NK"],
        "weeks": ["2026-10-10", "2026-10-17", "2026-10-24"],
        "curves": encode(c),
    }
    cutoff = dt.date(2026, 6, 7) + dt.timedelta(days=100)
    got = as_reported_samples(block, cutoff, np.random.default_rng(1))
    assert list(got.columns) == [
        "location",
        "target_end_date",
        "sample",
        "value",
    ]
    assert len(got) == 5 * 3 * 2
    # At the fit's own cut-off, the unseen week is each trajectory's
    # draw of it, known and imputed.
    row = got.set_index(["sample", "target_end_date", "location"])["value"]
    w = block["weeks"][-1]
    for k in range(1, 6):
        for p, place in enumerate(block["places"]):
            want = (
                c["draw_known"][k - 1, -1, p] + c["draw_imputed"][k - 1, -1, p]
            )
            assert row[(k, w, place)] == want
