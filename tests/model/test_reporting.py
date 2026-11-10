import datetime as dt

import numpy as np
import pytest
from jax import random
from numpyro.infer import Predictive

from bdbvhubrenewal.model import (
    build_model,
    forecast_trajectories,
    model_args,
)
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


@pytest.mark.parametrize("later", [4, 17])
def test_post_hoc_draws_match_the_model_as_reported(sim_vintage, fitted, later):
    from bdbvhubrenewal.forecasters.reporting import (
        NEGLIGIBLE,
        as_reported_samples,
        draw_reported,
        encode,
    )

    model, prior = fitted
    n, n_weeks = 2000, sim_vintage.n_weeks
    weeks = list(range(n_weeks - 4, n_weeks + 2))
    cut = sim_vintage.cutoff + later
    draws = forecast_trajectories(
        model,
        sim_vintage,
        n,
        posterior=prior,
        as_of=(cut,),
        report_weeks=weeks,
        report_days=30,
    )
    curves = draws["reporting"]
    model_view = draws["as_reported"][cut]["province"][:, weeks].sum(-1)
    post_hoc = draw_reported(curves, later, np.random.default_rng(1)).sum(-1)
    assert post_hoc.shape == model_view.shape
    # The data weeks share their expectations across the two passes, so
    # only the reporting noise differs; the forecast weeks also redraw
    # the latent process, so they are compared in distribution only.
    data_weeks = weeks.index(n_weeks - 1) + 1
    for j in range(len(weeks)):
        a, b = post_hoc[:, j], model_view[:, j]
        se = np.sqrt((a.var() + b.var()) / n)
        tol = 4 * se if j < data_weeks else 0.15 * max(b.mean(), 1.0)
        assert abs(a.mean() - b.mean()) <= tol + 1e-6, (j, a.mean(), b.mean())
        np.testing.assert_allclose(
            np.quantile(a, [0.25, 0.5, 0.75]),
            np.quantile(b, [0.25, 0.5, 0.75]),
            rtol=0.15,
            atol=2,
        )
    # The stored path (encoded, then drawn for a release date) differs
    # from the curves only by their rounding: shares to 1/255 of the
    # eventual count, and cells under `NEGLIGIBLE` reported at once.
    block = {
        "start_date": sim_vintage.start_date.isoformat(),
        "places": list(sim_vintage.names),
        "weeks": [f"w{w:03d}" for w in weeks],
        "curves": encode(curves),
    }
    release = sim_vintage.start_date + dt.timedelta(days=cut)
    rows = as_reported_samples(block, release, np.random.default_rng(1))
    stored = (
        rows.groupby(["sample", "target_end_date"])["value"]
        .sum()
        .unstack()
        .to_numpy()
    )
    final = curves["known"][:, -1] + curves["imputed"][:, -1]
    bound = (final / 255 / 2 + 2 * NEGLIGIBLE + 0.01).sum(-1).mean(0)
    for j in range(len(weeks)):
        a, b = stored[:, j], post_hoc[:, j]
        se = np.sqrt((a.var() + b.var()) / n)
        assert abs(a.mean() - b.mean()) <= 4 * se + bound[j], j
    # Reported by the cut-off: never more than eventually, in expectation.
    eventual = draws["province"][:, weeks].sum(-1).mean(0)
    assert np.all(post_hoc.mean(0) <= eventual * 1.1 + 2)
