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
    # zones: zone 6 in weeks 9 and 10, with a case in 10. Province 1's
    # zones have none at risk with a case, so the global term adds 0.
    assert q[6] == pytest.approx(1 / 2)
    # Both provinces come back in week 10 after weeks 9 and 10 at risk.
    # Provinces have no level outside their sibling group, nor has the
    # country, so their rate is the local one.
    np.testing.assert_allclose(q[:3], 1 / 2)


def test_a_zone_with_no_introduction_among_its_siblings_takes_the_rest():
    q = introduction_rates(_reported(), np.full(7, 11), PARENT, 3)
    # Province 1's zones have no re-introduction in weeks 9 to 11. The
    # zones of province 2 have 1 in 2 weeks at risk, so the global term
    # gives them 1 / 2 rather than 0.
    np.testing.assert_allclose(q[3:5], 1 / 2)
    # Zone 5 has no first case among its siblings in the window, and the
    # whole history no longer stands in. Zone 4's first case in week 10,
    # after weeks 9 and 10 with none, gives the global 1 / 2.
    assert q[5] == pytest.approx(1 / 2)


def test_local_and_global_pressures_combine_as_independent_sources() -> None:
    reported = _reported()
    # Zone 3 comes back in week 11 after weeks 9 to 11 at risk: 1 in 3
    # for province 1's zones, where province 2's zones have 1 in 2.
    reported[3, 11] = 1
    reported[1, 11] += 1
    reported[0, 11] += 1
    q = introduction_rates(reported, np.full(7, 11), PARENT, 3)
    both = 1 - (1 - 1 / 3) * (1 - 1 / 2)
    assert q[3] == pytest.approx(both)
    assert q[6] == pytest.approx(both)


def test_the_rate_is_0_only_when_the_whole_level_has_no_introduction() -> None:
    # Starting at week 8 no place has a case in weeks 6 to 8, and the
    # whole history does not stand in, so every rate is 0.
    early = introduction_rates(_reported(), np.full(7, 8), PARENT, 3)
    np.testing.assert_allclose(early, 0.0)
