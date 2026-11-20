"""The simulator's latent process against the R engine, run by run.

Each `latent-*.json` fixture (from `scripts/latent_fixtures.R`) holds a
drawn design and values and every random draw R made for it, in order:
standard normals and the path kinds of independent places. The run is
replayed by substituting those draws at the model's sample sites, in the
order the model makes them, and must use all of them.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
import pytest
from numpyro import handlers

from nowcastrenewal.latent.pmfs import gt_pmf
from nowcastrenewal.latent.rt import (
    KINDS,
    PlaceRt,
    RtPath,
    WeeklyNoise,
    daily_walk,
)
from nowcastrenewal.latent.simulate import (
    COLUMNS,
    simulate_latent,
    simulate_latent_batch,
)
from nowcastrenewal.latent.start import InitializeStrata

FIXTURES = Path(__file__).parent / "fixtures" / "latent"
LATENT = sorted(FIXTURES.glob("latent-*.json"))


def load(name: str) -> dict[str, Any]:
    return json.loads((FIXTURES / f"{name}.json").read_text())


def matrix(x: Any) -> np.ndarray:
    return np.array(x, dtype=float)


def replay(
    model: Any, normal: Any, choice: Any, *args: Any, **kwargs: Any
) -> Any:
    """Run `model` with R's draws at its sample sites, in order."""
    z = np.atleast_1d(np.asarray(normal, dtype=float))
    choices = [choice] if isinstance(choice, str) else list(choice or [])
    with jax.enable_x64(True):
        trace = handlers.trace(handlers.seed(model, 0)).get_trace(
            *args, **kwargs
        )
        data: dict[str, Any] = {}
        i = j = 0
        for name, site in trace.items():
            if site["type"] != "sample":
                continue
            value = jnp.asarray(site["value"])
            if jnp.issubdtype(value.dtype, jnp.integer):
                data[name] = jnp.asarray(KINDS.index(choices[j]))
                j += 1
                continue
            size = int(value.size)
            data[name] = jnp.asarray(z[i : i + size]).reshape(value.shape)
            i += size
        assert i == len(z), "R's normal draws were not all used"
        assert j == len(choices), "R's path draws were not all used"
        return handlers.substitute(handlers.seed(model, 0), data=data)(
            *args, **kwargs
        )


def test_paths_match_the_r_engine() -> None:
    f = load("paths")
    for kind, path in f["paths"].items():
        np.testing.assert_allclose(
            RtPath.from_values(kind, f["values"]).log_r(f["n"]),
            path,
            rtol=1e-12,
            atol=1e-15,
            err_msg=kind,
        )
    w = f["daily_walk"]
    np.testing.assert_allclose(
        daily_walk(np.array(w["days"]), np.array(w["walk"])),
        w["out"],
        rtol=1e-12,
    )


def test_weekly_noise_is_an_ar1_held_for_a_week_from_zero() -> None:
    z = np.array([0.5, -1.0, 2.0, 0.3])
    out = np.asarray(
        replay(
            lambda: WeeklyNoise("w")(n=30, sd=0.2, noise_name="e"),
            z,
            [],
        )
    )
    e = np.zeros(5)
    for w in range(1, 5):
        e[w] = 0.8 * e[w - 1] + 0.2 * z[w - 1]
    np.testing.assert_allclose(out, np.repeat(e, 7)[:30], rtol=1e-15)


def test_a_shared_relation_draws_only_the_strata_noise() -> None:
    path = RtPath("growth", amplitude=0.2)
    rt = PlaceRt("rt", path, "shared", noise_scale=5.0, coupling=1.0)
    with jax.enable_x64(True):
        trace = handlers.trace(handlers.seed(rt, 0)).get_trace(
            n=21, place=np.array([0, 0, 1])
        )
    sites = [k for k, s in trace.items() if s["type"] == "sample"]
    assert sites == [f"rt_stratum_{j}_decentered" for j in range(3)]


@pytest.mark.parametrize("early", [True, False])
def test_the_start_is_a_seed_on_the_last_day_or_a_level(early: bool) -> None:
    with jax.enable_x64(True):
        out = np.asarray(InitializeStrata(4, early)(np.array([0.4, 3.0, 0.0])))
    if early:
        want = np.zeros((4, 3))
        want[-1] = [1.0, 3.0, 0.0]
    else:
        want = np.tile([0.4, 3.0, 0.0], (4, 1))
    np.testing.assert_array_equal(out, want)


@pytest.mark.parametrize("path", LATENT, ids=lambda p: p.stem)
def test_latent_runs_replay_the_r_engine(path: Path) -> None:
    case = json.loads(path.read_text())
    design, values = case["design"], case["values"]
    gt = gt_pmf(values["gt_mean"], design["gt_family"], values["gt_cv"])
    strata = pd.DataFrame(case["strata"])
    n = case["n_days"] - len(gt)
    args = (n, gt, strata, design, values, KINDS, 0.5)
    tape = case["tape"]
    if "error" in case:
        with pytest.raises(ValueError, match=case["error"]):
            replay(simulate_latent, tape["normal"], tape["choice"], *args)
        return
    run = replay(simulate_latent, tape["normal"], tape["choice"], *args)
    assert run.n_gt == case["n_gt"]
    np.testing.assert_allclose(
        run.infections, matrix(case["infections"]), rtol=1e-9, atol=1e-9
    )
    np.testing.assert_allclose(
        run.onsets, matrix(case["onsets"]), rtol=1e-9, atol=1e-9
    )
    np.testing.assert_allclose(
        run.stratum_population, case["stratum_population"], rtol=1e-12
    )
    assert run.attack == pytest.approx(case["attack"], rel=1e-9)
    want = pd.DataFrame(case["latent"]).astype(float)
    for col in ("infections", "onsets", "r", "reff"):
        np.testing.assert_allclose(
            run.path[col].to_numpy(),
            want[col].to_numpy(),
            rtol=1e-9,
            atol=1e-9,
            err_msg=col,
        )


def test_the_latent_cases_cover_every_design_level() -> None:
    ran = [
        c["design"]
        for c in (json.loads(p.read_text()) for p in LATENT)
        if "error" not in c
    ]
    for factor, levels in {
        "mixing": {"none", "symmetric", "asymmetric", "distance"}
        | {"time_varying"},
        "place_rt": {"shared", "independent", "correlated", "lagged"},
        "window": {"early", "established", "endemic"},
        "imports": {"none", "constant", "pulses"},
        "seeding": {"start", "mid"},
    }.items():
        assert {d[factor] for d in ran} == levels, factor


def _simulations() -> list[dict[str, Any]]:
    case = json.loads(LATENT[-1].read_text())
    sims = [
        {
            "id": f"s{i}",
            "key": i,
            "n_days": case["n_days"],
            "design": case["design"],
            "values": case["values"],
            "strata": case["strata"],
        }
        for i in range(2)
    ]
    return [*sims, {**sims[0], "id": "short", "n_days": 3}]


def test_a_batch_runs_each_simulation_and_records_its_errors() -> None:
    sims = _simulations()
    long, values = simulate_latent_batch(sims, KINDS)
    assert list(long.columns) == list(COLUMNS)
    assert set(long["id"]) == {"s0", "s1"}
    k = len(sims[0]["strata"])
    s0 = long[long["id"] == "s0"]
    assert set(s0["group"]) == set(range(k + 1))
    assert s0.groupby("group").size().eq(sims[0]["n_days"]).all()
    # Group 0 is the sum of the strata.
    whole = s0[s0["group"] == 0].set_index("day")["onsets"]
    parts = s0[s0["group"] > 0].groupby("day")["onsets"].sum()
    np.testing.assert_allclose(whole, parts, rtol=1e-12)
    assert (
        values["short"]["error"]
        == "the generation time is longer than the series"
    )
    assert values["s0"]["error"] is None
    assert len(values["s0"]["stratum_population"]) == k
    gt = gt_pmf(
        sims[0]["values"]["gt_mean"],
        sims[0]["design"]["gt_family"],
        sims[0]["values"]["gt_cv"],
    )
    np.testing.assert_allclose(values["s0"]["gt"], gt)
    assert values["s0"]["n_gt"] == len(gt)


def test_a_key_gives_the_same_run_and_another_key_another() -> None:
    sims = _simulations()[:2]
    sims[1]["key"] = 0
    long, _ = simulate_latent_batch(sims, KINDS)
    a = long[long["id"] == "s0"]["onsets"].to_numpy()
    b = long[long["id"] == "s1"]["onsets"].to_numpy()
    np.testing.assert_array_equal(a, b)
    sims[1]["key"] = 5
    long, _ = simulate_latent_batch(sims, KINDS)
    b = long[long["id"] == "s1"]["onsets"].to_numpy()
    assert not np.allclose(a, b)


def test_any_error_in_one_simulation_is_recorded_and_the_batch_goes_on() -> (
    None
):
    # As the R batch's tryCatch, which redraws a simulation on any error.
    sims = _simulations()[:2]
    broken = sims[1]
    broken["values"] = {
        k: v for k, v in broken["values"].items() if k != "r_drift"
    }
    long, values = simulate_latent_batch(sims, KINDS)
    assert values["s1"]["error"].startswith("KeyError")
    assert values["s0"]["error"] is None
    assert set(long["id"]) == {"s0"}


def test_a_single_draw_of_a_vector_value_may_be_a_scalar() -> None:
    case = json.loads(LATENT[0].read_text())
    values = {
        k: (v[0] if isinstance(v, list) and len(v) == 1 else v)
        for k, v in case["values"].items()
    }
    sim = {**_simulations()[0], "values": values, "design": case["design"]}
    sim["strata"], sim["n_days"] = case["strata"], case["n_days"]
    long, out = simulate_latent_batch([sim], KINDS)
    assert out["s0"]["error"] is None
    assert np.isfinite(long["onsets"]).all()


def test_exponential_depletion_runs_and_stays_within_the_population() -> None:
    sims = _simulations()[:1]
    long, out = simulate_latent_batch(sims, KINDS, depletion="exponential")
    assert out["s0"]["error"] is None
    s = long[(long["id"] == "s0") & (long["group"] > 0)]
    total = s.groupby("group")["infections"].sum().to_numpy()
    assert (total <= np.array(out["s0"]["stratum_population"]) * 2).all()


def test_an_unknown_depletion_is_refused() -> None:
    _, out = simulate_latent_batch(
        _simulations()[:1], KINDS, depletion="quadratic"
    )
    assert "depletion" in out["s0"]["error"]


def test_the_fitted_models_do_not_import_the_latent_process() -> None:
    code = (
        "import sys, nowcastrenewal.forecaster, nowcastrenewal.zones;"
        "print(sorted(m for m in sys.modules"
        " if m.startswith('nowcastrenewal.latent')))"
    )
    out = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        check=True,
    )
    assert out.stdout.strip() == "[]"
