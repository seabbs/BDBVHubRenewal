"""The coupling fit from a model's own matured forecasts (trajectories)."""

import numpy as np
import pytest
from scipy.special import ndtri

from nowcastutils.trajectories import couple, fit_coupling, pit_scores

LEVELS = np.array([0.025, 0.1, 0.25, 0.5, 0.75, 0.9, 0.975])
SPREAD = np.array([-2.0, -1.3, -0.7, 0.0, 0.7, 1.3, 2.0])


def _scores(rho, s, origins=400, series=4, steps=6, seed=3):
    """Normal scores `[origins, series, steps]` from the gaussian copula."""
    q = np.zeros((series, steps, len(LEVELS))) + np.linspace(0, 1, 7)
    u = couple(
        q,
        np.linspace(0, 1, 7),
        "gaussian",
        {"rho": rho, "s": s},
        n_samples=origins,
        seed=seed,
    )
    return ndtri(np.clip(u, 1e-9, 1 - 1e-9))


def test_a_value_at_the_median_scores_zero_and_one_below_all_is_low():
    q = (10.0 + SPREAD)[None, None, :]
    rng = np.random.default_rng(1)
    assert pit_scores(q, LEVELS, np.array([[10.0]]), rng)[0, 0] == (
        pytest.approx(0.0)
    )
    # Halfway between the 0.25 and 0.5 quantiles: level 0.375.
    mid = np.array([[10.0 - 0.35]])
    got = pit_scores(q, LEVELS, mid, rng)[0, 0]
    assert got == pytest.approx(ndtri(0.375))
    assert pit_scores(q, LEVELS, np.array([[0.0]]), rng)[0, 0] < -3


def test_a_value_on_a_flat_quantile_function_draws_its_level():
    # Every quantile 0 and 0 reported: the level is uniform on (0, 1),
    # not a fixed score, so zero counts carry no false correlation.
    q = np.zeros((1, 2000, len(LEVELS)))
    z = pit_scores(q, LEVELS, np.zeros((1, 2000)), np.random.default_rng(2))
    assert np.isfinite(z).all()
    assert z.mean() == pytest.approx(0.0, abs=0.1)
    assert z.std() == pytest.approx(1.0, abs=0.1)


def test_missing_cells_have_no_score():
    q = (10.0 + SPREAD)[None, None, :].repeat(2, axis=1)
    z = pit_scores(
        q, LEVELS, np.array([[10.0, np.nan]]), np.random.default_rng(1)
    )
    assert np.isnan(z[0, 1])
    assert np.isfinite(z[0, 0])


def test_the_gaussian_fit_recovers_the_correlations_it_was_drawn_with():
    fit = fit_coupling(_scores(0.8, 0.4), "gaussian")
    assert fit["method"] == "gaussian"
    assert fit["rho"] == pytest.approx(0.8, abs=0.05)
    assert fit["s"] == pytest.approx(0.4, abs=0.05)
    assert fit["n_origins"] == 400
    # Many origins: little shrinkage.
    assert fit["shrinkage_rho"] < 0.05


def test_scores_with_no_dependence_shrink_to_independence():
    z = np.random.default_rng(5).standard_normal((30, 3, 5))
    fit = fit_coupling(z, "gaussian")
    assert abs(fit["rho"]) < 0.1
    assert 0.0 <= fit["s"] < 0.1
    assert fit["shrinkage_rho"] > 0.2


def test_one_origin_gives_no_evidence_so_independence():
    fit = fit_coupling(_scores(0.8, 0.4, origins=1), "gaussian")
    assert fit["rho"] == 0.0
    assert fit["s"] == 0.0
    assert fit["shrinkage_rho"] == 1.0
    assert fit["shrinkage_s"] == 1.0


def test_missing_scores_are_left_out_and_the_fit_couples():
    z = _scores(0.8, 0.4)
    z[::3, 1, 2:] = np.nan
    fit = fit_coupling(z, "gaussian")
    assert np.isfinite([fit["rho"], fit["s"]]).all()
    q = (10.0 + SPREAD)[None, None, :].repeat(3, axis=1)
    assert couple(q, LEVELS, "gaussian", fit, 10, 1).shape == (10, 1, 3)


@pytest.mark.parametrize("method", ["independent", "comonotone"])
def test_methods_without_parameters_fit_to_their_name(method):
    assert fit_coupling(_scores(0.8, 0.4), method) == {"method": method}


def test_an_unknown_method_is_an_error():
    with pytest.raises(ValueError, match="unknown method"):
        fit_coupling(_scores(0.8, 0.4), "schaake")


def test_one_series_has_no_series_pairs_so_s_is_zero():
    fit = fit_coupling(_scores(0.6, 0.0, series=1), "gaussian")
    assert fit["n_series_pairs"] == 0
    assert fit["s"] == 0.0
    assert fit["shrinkage_s"] == 1.0
    assert fit["rho"] > 0.3


def test_s_stays_below_one_as_rho_does():
    # Identical series correlate fully; the fit keeps s below 1, as rho.
    one = _scores(0.0, 0.0, series=1)
    fit = fit_coupling(np.repeat(one, 3, axis=1), "gaussian")
    assert 0.99 < fit["s"] < 1.0
