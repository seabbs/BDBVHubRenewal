"""Introductions into places with no recent cases."""

from __future__ import annotations

import numpy as np
import pytest

from bdbvhubrenewal.forecasters.introductions import introduction_rates

# Country 0, provinces 1 and 2, zones 3 and 4 in province 1 and zones 5
# and 6 in province 2.
PARENT = np.array([-1, 0, 0, 1, 1, 2, 2])


def _reported() -> np.ndarray:
    zones = np.zeros((4, 12))
    zones[0, 1] = 1  # zone 3: seen early, silent since
    zones[1, 10] = 1  # zone 4: first case in the window
    # zone 5: never a case
    zones[3, [2, 10]] = 1  # zone 6: seen early, back in the window
    provinces = np.stack([zones[:2].sum(0), zones[2:].sum(0)])
    return np.vstack([provinces.sum(0), provinces, zones])


def test_first_cases_and_reintroductions_have_their_own_rates() -> None:
    # k = 3, and every place starts at week 11, so weeks 9 to 11 count.
    q = introduction_rates(_reported(), np.full(7, 11), PARENT, 3)
    # A first case: zone 4 in weeks 9 and 10, with a case in 10, and
    # zone 5 in weeks 9 to 11. Province 1's zones have 1 in 2 weeks.
    # Province 2's have none, so they take all zones': 1 in 5 weeks.
    # Zone 6's first case in week 2 is before the window.
    assert q[5] == pytest.approx(1 / 5)
    # A case after 3 silent weeks in a place seen before: zone 3 in weeks
    # 9 to 11 and zone 6 in weeks 9 and 10, with a case in 10. Province
    # 1's zones have none, so they take all zones': 1 in 5 weeks.
    assert q[3] == pytest.approx(1 / 5)
    assert q[4] == pytest.approx(1 / 5)
    assert q[6] == pytest.approx(1 / 2)
    # Both provinces come back in week 10 after weeks 9 and 10 at risk,
    # and the country is a group of its own.
    np.testing.assert_allclose(q[:3], 1 / 2)


def test_only_the_walk_weeks_to_the_start_count() -> None:
    # Starting at week 8, weeks 6 to 8 have no case, so no place has an
    # estimate of its own or from a parent.
    q = introduction_rates(_reported(), np.full(7, 8), PARENT, 3)
    np.testing.assert_allclose(q, 0.0)
    # A wider window takes in zone 6's first case in week 2 too.
    wide = introduction_rates(_reported(), np.full(7, 11), PARENT, 10)
    assert wide[5] > 0
