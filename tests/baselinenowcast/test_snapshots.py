"""Snapshot extension and hub weeks."""

from __future__ import annotations

import numpy as np
import pandas as pd

from bdbvhubrenewal.baselinenowcast import hub_week_matrix, hub_weeks


def test_hub_week_matrix_uses_sunday_to_saturday_weeks() -> None:
    first = "2026-07-05"  # a Sunday
    w, sats = hub_week_matrix(first, 17)
    assert w.shape == (3, 17)
    assert [str(s) for s in sats] == ["2026-07-11", "2026-07-18", "2026-07-25"]
    np.testing.assert_array_equal(w.sum(1), [7, 7, 3])
    w2, sats2 = hub_week_matrix("2026-07-08", 17)
    assert str(sats2[0]) == "2026-07-18"
    assert w2[0, :4].sum() == 0
    assert w2[0].sum() == 7


def test_hub_weeks_give_12_horizons_at_a_saturday_and_13_otherwise() -> None:
    d = 84
    n_ref = 250
    for cutoff, expected in (("2026-10-24", 12), ("2026-10-16", 13)):
        first = pd.Timestamp(cutoff) - pd.Timedelta(days=n_ref - 1)
        t = np.arange(n_ref)[:, None]
        dl = np.arange(d + 1)[None, :]
        mask = (t + dl) <= n_ref - 1
        agg = hub_weeks(first.date())
        _, m = agg(np.where(mask, 1.0, 0.0), mask)
        assert int((~m.all(axis=1)).sum()) == expected
        # The cut-off week holds only days up to the cut-off.
        w, _ = hub_week_matrix(first.date(), n_ref)
        assert w[-1].sum() == {"2026-10-24": 7, "2026-10-16": 6}[cutoff]


def test_hub_weeks_sum_days_and_propagate_unobserved() -> None:
    first = "2026-07-05"
    n = 16
    v = np.ones((n, 2))
    m = np.ones((n, 2), bool)
    m[-1, 1] = False
    out, ok = hub_weeks(first)(v, m)
    np.testing.assert_allclose(np.asarray(out)[:, 0], [7, 7, 2])
    np.testing.assert_array_equal(ok[:, 1], [True, True, False])
    assert ok[:, 0].all()
