"""The BVDOutbreakSize-informed variant (tempered Rt pseudo-observation)."""

from dataclasses import replace
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import numpyro
import pandas as pd
import pytest
from jax.scipy.stats import norm

from bdbvhubrenewal.model import BVD, MAIN, build_model, model_args
from bdbvhubrenewal.model.inputs import (
    bvd_inputs,
    bvd_rows,
    vintage_from_tables,
)
from bdbvhubrenewal.model.observation import RtPseudoObservation

from .conftest import SIM_LOCATIONS, VINTAGE_DAYS, vintage_date
from .test_model import trace

DATA = Path(__file__).parent / "data"


@pytest.fixture(scope="module")
def release():
    return bvd_rows(
        pd.read_csv(DATA / "bvd_3151_province_estimates.csv"),
        pd.read_csv(DATA / "bvd_3151_province_timeseries.csv"),
    )


def test_bvd_rows(release):
    assert len(release) == 9
    ituri = release.set_index("group").loc["Ituri"]
    assert ituri["members"] == "CD-IT"
    assert ituri["log_sd"] == pytest.approx(
        np.log(1.5194 / 0.7129) / 3.29, 1e-3
    )
    est = pd.read_csv(DATA / "bvd_3151_province_estimates.csv")
    est.loc[0, "province"] = "atlantis"
    with pytest.raises(ValueError, match="Unknown BVD patch"):
        bvd_rows(est, pd.read_csv(DATA / "bvd_3151_province_timeseries.csv"))


def test_bvd_inputs_map_and_mask(release):
    import datetime as dt

    names = ["CD-IT", "CD-NK", "CD-HU"]
    start = dt.date(2026, 5, 3)
    out = bvd_inputs(release, names, start)
    assert out["groups"].shape == (4, 3)  # national plus three patches
    assert out["kinds"].tolist() == [0, 1, 1, 1]
    assert out["day"] == (dt.date(2026, 10, 20) - start).days + 6
    assert np.all(out["scales"] >= 0.05)
    masked = bvd_inputs(release, names, start, masked=("CD-NK",))
    assert masked["kinds"].tolist() == [1, 1]


def test_pseudo_observation_density_is_analytic():
    rt = jnp.full((70, 2), 1.2).at[:, 1].set(0.8)
    inf = jnp.ones((70, 2)).at[:, 1].set(3.0)
    obs = RtPseudoObservation(
        0.5,
        14,
        [[True, True], [False, True]],
        [0.1, -0.3],
        [0.2, 0.25],
        [0, 1],
    )
    with numpyro.handlers.trace() as tr:
        obs.sample(rt, inf, 14)
    national = jnp.log((1.2 + 3 * 0.8) / 4)
    model = jnp.array([national, jnp.log(0.8) - national])
    expect = norm.logpdf(
        jnp.array([0.1, -0.3]), model, jnp.array([0.2, 0.25]) / jnp.sqrt(0.5)
    ).sum()
    got = tr["bvd_rt_obs"]["fn"].log_prob(tr["bvd_rt_obs"]["value"]).sum()
    assert float(got) == pytest.approx(float(expect), rel=1e-5)


def sim_with_bvd(outbreak, sim_tables, table, release, config):
    day = VINTAGE_DAYS[-1]
    target, delays = sim_tables[day]
    rows = release.copy()
    rows["as_of"] = vintage_date(outbreak, day - 4)
    return vintage_from_tables(
        target,
        delays,
        vintage_date(outbreak, day),
        SIM_LOCATIONS,
        table,
        config,
        bvd=rows,
    )


def test_alpha_zero_reproduces_main(outbreak, sim_tables, table, release):
    data = sim_with_bvd(
        outbreak, sim_tables, table, release, replace(BVD, bvd_alpha=0.0)
    )
    assert data.bvd is not None
    plain = build_model(data, MAIN)
    zero = build_model(data, replace(BVD, bvd_alpha=0.0))
    tr = trace(plain, data)
    params = {
        k: v["value"]
        for k, v in tr.items()
        if v["type"] == "sample" and not v["is_observed"]
    }
    ld = [
        float(
            numpyro.infer.util.log_density(
                m.sample, (), model_args(m, data), params
            )[0]
        )
        for m in (plain, zero)
    ]
    assert ld[0] == ld[1]
    with_bvd = build_model(data, BVD)
    tr_bvd = trace(with_bvd, data)
    assert "bvd_rt_obs" in tr_bvd and "bvd_z" in tr_bvd


def test_group_rt_weights_by_counts_not_per_resident():
    # Province 0 is ten times larger, so with equal infections per
    # resident it has ten times the onsets and dominates national Rt.
    rt = jnp.ones((70, 2)).at[:, 1].set(2.0)
    per_resident = jnp.ones((70, 2))
    obs = RtPseudoObservation(1.0, 14, [[True, True]], [0.0], [0.1], [0])
    group, national = obs.log_group_rt(
        rt, per_resident, 14, jnp.array([10 / 11, 1 / 11])
    )
    expect = jnp.log((10 * 1.0 + 1 * 2.0) / 11)
    assert float(group[0]) == pytest.approx(float(expect), rel=1e-6)
    assert float(national) == pytest.approx(float(expect), rel=1e-6)


def test_fit_axis_reaches_the_bvd_day(outbreak, sim_tables, table, release):
    from bdbvhubrenewal.model.model import fit_weeks

    day = VINTAGE_DAYS[-1]
    target, delays = sim_tables[day]
    rows = release.copy()
    # BVD's as_of a few days after ours puts s_b in a later week.
    rows["as_of"] = vintage_date(outbreak, day + 3)
    data = vintage_from_tables(
        target, delays, vintage_date(outbreak, day), SIM_LOCATIONS, table,
        BVD, bvd=rows,
    )  # fmt: skip
    assert data.bvd["day"] // 7 + 1 > data.n_weeks
    assert fit_weeks(data, BVD) == data.bvd["day"] // 7 + 1
    model = build_model(data, BVD)
    tr = trace(model, data)
    assert np.isfinite(np.asarray(tr["bvd_log_rt_model"]["value"])).all()
    ld = numpyro.infer.util.log_density(
        model.sample,
        (),
        model_args(model, data),
        {
            k: v["value"]
            for k, v in tr.items()
            if v["type"] == "sample" and not v["is_observed"]
        },
    )[0]
    assert np.isfinite(float(ld))
