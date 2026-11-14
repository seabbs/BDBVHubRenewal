"""``baselinenowcast`` against R baselinenowcast 0.2.2."""

from __future__ import annotations

import jax
import numpy as np
import pytest
from bnc_helpers import Case, read_vec

from bdbvhubrenewal.baselinenowcast import baselinenowcast

pytestmark = pytest.mark.filterwarnings(
    "ignore:.*reference times available.*:UserWarning"
)


def test_point_nowcast_matches_r(case: Case) -> None:
    got = baselinenowcast(case.values, case.mask, output_type="point")
    np.testing.assert_allclose(
        got, read_vec(case.name, "point"), rtol=1e-10, atol=1e-8
    )


def test_samples_need_a_key(small: Case) -> None:
    with pytest.raises(ValueError):
        baselinenowcast(small.values, small.mask)


def test_output_type_is_checked(small: Case) -> None:
    with pytest.raises(ValueError):
        baselinenowcast(
            small.values,
            small.mask,
            jax.random.key(0),
            output_type="mean",  # type: ignore[arg-type]
        )


def test_point_does_not_need_two_retrospective_nowcasts() -> None:
    v = np.array(
        [[10.0, 5.0, 2.0], [12.0, 6.0, 3.0], [11.0, 5.0, 0.0], [13.0, 0.0, 0.0]]
    )
    m = np.array([[1, 1, 1], [1, 1, 1], [1, 1, 0], [1, 0, 0]], bool)
    out = baselinenowcast(v, m, output_type="point")
    assert out.shape == (4,)
    assert float(out[-1]) > 13.0
