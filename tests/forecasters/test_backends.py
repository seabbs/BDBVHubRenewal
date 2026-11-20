"""The Chronos-2 backend module (nowcastfoundation.backends.chronos2)."""

from nowcastfoundation import forecaster
from nowcastfoundation.backends import Series, chronos2


def test_the_forecaster_predicts_through_the_chronos2_backend():
    assert forecaster._predict is chronos2._predict
    assert (
        forecaster.CHRONOS_MODEL == chronos2.CHRONOS_MODEL == "amazon/chronos-2"
    )


def test_the_group_type_is_the_backends():
    assert forecaster.Series is Series
