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
    # A case after 3 silent weeks in a place seen before. Province 2's
    # zones: zone 6 in weeks 9 and 10, with a case in 10. Over the whole
    # history it would be 1 in 5 weeks (6 to 10), so the window counts.
    assert q[6] == pytest.approx(1 / 2)
    # Province 1's zones have no re-introduction in the window or before.
    np.testing.assert_allclose(q[3:5], 0.0)
    # Both provinces come back in week 10 after weeks 9 and 10 at risk,
    # and the country is a group of its own.
    np.testing.assert_allclose(q[:3], 1 / 2)


def test_without_an_introduction_in_the_window_the_whole_history_counts() -> (
    None
):
    q = introduction_rates(_reported(), np.full(7, 11), PARENT, 3)
    # Province 2's zones have no first case in weeks 9 to 11, so the weeks
    # to the start week stand in: zone 5 in weeks 0 to 11 and zone 6 in
    # weeks 0 to 2, with its first case in week 2.
    assert q[5] == pytest.approx(1 / 15)
    # Starting at week 8 no place has a case in weeks 6 to 8, so all fall
    # back, and province 2's zones have 1 first case in 9 + 3 weeks.
    early = introduction_rates(_reported(), np.full(7, 8), PARENT, 3)
    assert early[5] == pytest.approx(1 / 12)
