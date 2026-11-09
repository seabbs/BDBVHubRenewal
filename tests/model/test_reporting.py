import numpy as np
import pytest
from jax import random
from numpyro.infer import Predictive

from bdbvhubrenewal.model import build_model, model_args
from bdbvhubrenewal.model.reporting import expected_reported_by


@pytest.fixture(scope="module")
def fitted(sim_vintage):
    """The model and prior draws standing in for a posterior."""
    model = build_model(sim_vintage)
    return model, _prior(model, sim_vintage)


def _prior(model, data, n=4):
    prior = Predictive(model.sample, num_samples=n)(
        random.key(2), **model_args(model, data)
    )
    return {
        k: v
        for k, v in prior.items()
        if not k.endswith("_obs") and k != "onsets_delays"
    }


def _weekly(daily):
    n = daily.shape[-2] // 7 * 7
    w = daily[..., :n, :].reshape(*daily.shape[:-2], n // 7, 7, -1)
    return w.sum(-2)


@pytest.mark.parametrize("later", [0, 3, 11, 40, None])
def test_expected_reported_by_matches_the_model(sim_vintage, fitted, later):
    model, prior = fitted
    n_init = model.latent.n_initialization_points
    as_of = None if later is None else sim_vintage.cutoff + later
    pred = Predictive(model.sample, posterior_samples=prior)(
        random.key(3),
        **model_args(model, sim_vintage, forecast=True, as_of=as_of),
    )
    onsets = np.asarray(pred["onsets_expected_daily"], dtype=float)
    days = np.arange(onsets.shape[1])
    cuts = [n_init + sim_vintage.cutoff]
    cuts.append(np.inf if as_of is None else n_init + as_of)
    for k in range(onsets.shape[0]):
        out = expected_reported_by(
            onsets[k],
            np.asarray(pred["onsets_delay_pmf"][k], dtype=float),
            np.asarray(pred["missing_fraction"][k], dtype=float),
            np.asarray(pred["onsets_publication_lag"][k], dtype=float),
            np.asarray(sim_vintage.hub_delay_pmf, dtype=float),
            np.asarray(cuts),
            days,
        )
        for stream in ("known", "imputed"):
            by_cut = _weekly(out[stream])
            np.testing.assert_allclose(
                by_cut[0],
                pred[f"onsets_expected_{stream}"][k],
                rtol=1e-4,
                atol=1e-4,
            )
            np.testing.assert_allclose(
                by_cut[1],
                pred[f"onsets_expected_{stream}_final"][k],
                rtol=1e-4,
                atol=1e-4,
            )
