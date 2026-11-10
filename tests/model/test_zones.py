"""The renewal model with health zones as its places."""

import numpy as np
import numpyro
import pandas as pd
import pytest
from numpyro.infer import init_to_median

from bdbvhubrenewal.model import MAIN, build_model, model_args
from bdbvhubrenewal.model.inputs import vintage_from_tables
from bdbvhubrenewal.model.zones import (
    build_zone_model,
    nested_basis,
    zone_table,
    zone_vintage_from_tables,
)

from .conftest import SIM_LOCATIONS, VINTAGE_DAYS, vintage_date

# Two real zone codes per simulated province, so the zone table has them.
ZONES = {
    "CD-IT": ("CD-IT-10", "CD-IT-35"),
    "CD-NK": ("CD-NK-03", "CD-NK-14"),
    "CD-HU": ("CD-HU-12", "CD-HU-02"),
    "CD-TO": ("CD-TO-06", "CD-TO-16"),
    "CD-BU": ("CD-BU-05", "CD-BU-01"),
    "CD-SK": ("CD-SK-01", "CD-SK-02"),
}


def _params(model, data):
    median = numpyro.handlers.substitute(
        numpyro.handlers.seed(model.sample, rng_seed=0),
        substitute_fn=init_to_median(num_samples=15),
    )
    with numpyro.handlers.trace() as tr:
        median(**model_args(model, data))
    return tr, {
        k: v["value"]
        for k, v in tr.items()
        if v["type"] == "sample" and not v["is_observed"]
    }


def _density(model, data, params):
    return float(
        numpyro.infer.util.log_density(
            model.sample, (), model_args(model, data), params
        )[0]
    )


def _split(target):
    """Split each province's weekly counts over its two zones."""
    rows = []
    prov = target.loc[target["location"].isin(ZONES)]
    for _, r in prov.iterrows():
        a, b = ZONES[r["location"]]
        for zone, share in ((a, 0.7), (b, 0.3)):
            obs = np.floor(r["observed"] * share)
            imp = np.floor(r["imputed"] * share)
            if zone == b:
                obs, imp = (
                    r["observed"] - np.floor(r["observed"] * 0.7),
                    r["imputed"] - np.floor(r["imputed"] * 0.7),
                )
            rows.append(
                {
                    **r.to_dict(),
                    "location": zone,
                    "observed": obs,
                    "imputed": imp,
                    "total": obs + imp,
                }
            )
    return pd.DataFrame(rows)


def test_nested_basis_pools_zones_within_provinces():
    parents = np.array([0, 0, 0, 1, 1, 2])
    active = np.ones(6, dtype=bool)
    between, within = nested_basis(parents, active)
    # Province deviations: equal within a province, sum to zero over the
    # provinces.
    assert between.shape == (6, 2)
    for col in between.T:
        assert col[0] == col[1] == col[2] and col[3] == col[4]
        assert np.unique(col[parents], return_index=True)[0].size <= 3
    by_province = between[[0, 3, 5]]
    np.testing.assert_allclose(by_province.sum(axis=0), 0.0, atol=1e-12)
    np.testing.assert_allclose(
        by_province.T @ by_province, np.eye(2), atol=1e-12
    )
    # Zone deviations sum to zero within each province; a province with
    # one zone has none.
    assert within.shape == (6, 2 + 1)
    for p in range(3):
        np.testing.assert_allclose(
            within[parents == p].sum(axis=0), 0.0, atol=1e-12
        )
    np.testing.assert_allclose(within[5], 0.0)
    np.testing.assert_allclose(between.T @ within, 0.0, atol=1e-12)
    np.testing.assert_allclose(within.T @ within, np.eye(3), atol=1e-12)


def test_nested_basis_leaves_out_inactive_places():
    parents = np.array([0, 0, 1, 1, 2])
    active = np.array([True, False, True, True, False])
    between, within = nested_basis(parents, active)
    # Province 2 has no active zone, so it follows the pooled trend.
    assert between.shape == (5, 1)
    np.testing.assert_allclose(between[4], 0.0)
    # An inactive zone takes its province's deviation, with no own.
    np.testing.assert_allclose(between[1], between[0])
    np.testing.assert_allclose(within[[0, 1, 4]], 0.0)
    assert within.shape == (5, 1)


def test_one_zone_per_province_is_the_province_model(
    outbreak, sim_tables, table
):
    day = VINTAGE_DAYS[-1]
    target, delays = sim_tables[day]
    vintage = vintage_date(outbreak, day)
    province = vintage_from_tables(
        target, delays, vintage, SIM_LOCATIONS, table
    )
    zoned = zone_vintage_from_tables(
        target,
        delays,
        vintage,
        SIM_LOCATIONS,
        table,
        parents={p: p for p in SIM_LOCATIONS},
    )
    assert zoned.parent_names == list(SIM_LOCATIONS)
    p_model, z_model = build_model(province), build_zone_model(zoned)
    _, params = _params(p_model, province)
    assert _density(z_model, zoned, params) == pytest.approx(
        _density(p_model, province, params), rel=1e-5
    )


@pytest.fixture(scope="module")
def zone_vintage(outbreak, sim_tables):
    day = VINTAGE_DAYS[-1]
    target, delays = sim_tables[day]
    zones = tuple(z for p in SIM_LOCATIONS for z in ZONES[p])
    return zone_vintage_from_tables(
        _split(target),
        delays,
        vintage_date(outbreak, day),
        zones,
        zone_table(),
        parents={z: z[:5] for z in zones},
    )


def test_zone_vintage_scores_delays_by_province(zone_vintage):
    v = zone_vintage
    assert v.n_provinces == 12
    assert v.parent_names == list(SIM_LOCATIONS)
    assert v.parents.tolist() == [i // 2 for i in range(12)]
    assert v.delay_counts.shape == (v.n_weeks, 6, 61)
    assert v.kernel.shape == (12, 12)
    np.testing.assert_allclose(v.kernel.sum(axis=1), 1.0)
    pops = zone_table().set_index("location")["population"]
    np.testing.assert_allclose(v.populations, pops[v.names].to_numpy())


def test_zone_model_shares_province_parts(zone_vintage):
    v = zone_vintage
    model = build_zone_model(v)
    tr, params = _params(model, v)
    # Delay offsets and missing-onset deviations are by province.
    seeded_provinces = np.unique(v.parents[v.seeded]).size
    assert params["delay_offset_z"].shape == (seeded_provinces - 1,)
    assert params["logit_pi0_z"].shape[-1] == seeded_provinces - 1
    assert tr["delay_shift"]["value"].shape[-1] == 6
    # Rt has province deviations and zone deviations within provinces.
    _, within = nested_basis(v.parents, v.seeded)
    assert params["log_rt_deviations_z"].shape[-1] == seeded_provinces - 1
    assert params["log_rt_zone_deviations_z"].shape[-1] == within.shape[1]
    assert tr["onsets_delays"]["value"].shape[-2:] == (6, 61)
    assert np.isfinite(_density(model, v, params))


def test_zone_bvd_groups_map_by_province(outbreak, sim_tables):
    from bdbvhubrenewal.model.inputs import bvd_inputs

    rows = pd.DataFrame(
        {
            "group": ["national", "p1"],
            "members": ["", "CD-IT"],
            "as_of": ["2026-09-01"] * 2,
            "log_mean": [0.0, 0.1],
            "log_sd": [0.1, 0.2],
            "ref_log_mean": [0.0, 0.0],
            "ref_log_sd": [0.1, 0.1],
        }
    )
    names = ["CD-IT-10", "CD-IT-35", "CD-NK-03"]
    # Zones enter by their province, so a patch holds its zones.
    out = bvd_inputs(
        rows,
        [z[:5] for z in names],
        pd.Timestamp("2026-08-01").date(),
    )
    assert out["groups"].tolist() == [[True, True, True], [True, True, False]]
    assert MAIN.bvd is False
