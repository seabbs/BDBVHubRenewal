import jax.numpy as jnp
import numpyro
import pytest

from bdbvhubrenewal.model.rt_observation import RtPseudoObservation

ARGS = {
    "day": 7,
    "groups": [[True, True], [True, False]],
    "values": [0.0, 0.1],
    "scales": [0.1, 0.2],
    "kinds": [0, 1],
}


def sites(alpha):
    obs = RtPseudoObservation("bvd", alpha, **ARGS)
    rt = jnp.ones((21, 2))
    infections = jnp.ones((21, 2))
    with numpyro.handlers.trace() as tr, numpyro.handlers.seed(rng_seed=0):
        obs(rt=rt, infections=infections, n_init=0)
    return tr


def test_sites_are_named_from_the_name():
    tr = sites(0.5)
    assert set(tr) == {"bvd_log_rt_model", "bvd_z", "bvd_rt_obs"}
    assert tr["bvd_rt_obs"]["is_observed"]


def test_no_tempering_adds_no_likelihood():
    assert "bvd_rt_obs" not in sites(0.0)


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"alpha": 1.5}, "alpha"),
        ({"day": -1}, "day"),
        ({"groups": [True, False]}, "groups"),
        ({"values": [0.0]}, "values"),
        ({"scales": [0.1, 0.0]}, "positive"),
        ({"kinds": [0, 2]}, "kinds"),
    ],
)
def test_bad_inputs_raise(change, message):
    args = {"alpha": 0.5, **ARGS, **change}
    with pytest.raises(ValueError, match=message):
        RtPseudoObservation("bvd", **args)


def test_repr_names_the_component():
    obs = RtPseudoObservation("bvd", 0.5, **ARGS)
    assert repr(obs) == (
        "RtPseudoObservation(name='bvd', alpha=0.5, day=7, n_groups=2)"
    )
