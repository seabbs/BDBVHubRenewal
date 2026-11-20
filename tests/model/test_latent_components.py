"""The simulator's renewal-step components against the R engine.

The fixtures come from `scripts/latent_fixtures.R`, which runs the R
functions in `packages/nowcastfoundation/r/R/infections.R` and
`observation.R`.
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
from pyrenew.deterministic import DeterministicPMF, DeterministicVariable
from pyrenew.latent.infection_functions import (
    logistic_susceptibility_adjustment,
)

from nowcastrenewal.latent.imports import Imports
from nowcastrenewal.latent.mixing import PlaceMixing, count_mixing
from nowcastrenewal.latent.pmfs import gt_pmf, incubation_pmf
from nowcastrenewal.latent.reference import ReferenceEvents
from nowcastrenewal.latent.strata import (
    ExponentialDepletion,
    LinearDepletion,
    Shifts,
    Variants,
    strata_renewal,
)

FIXTURES = Path(__file__).parent / "fixtures" / "latent"


def load(name: str) -> dict[str, Any]:
    return json.loads((FIXTURES / f"{name}.json").read_text())


def matrix(x: Any) -> np.ndarray:
    return np.array(x, dtype=float)


def run(fn: Any, *args: Any, **kwargs: Any) -> Any:
    """Run a component's sample in float64 under a seed."""
    with jax.enable_x64(True):
        return handlers.seed(fn, 0)(*args, **kwargs)


def test_pmfs_match_the_r_engine() -> None:
    f = load("pmfs")
    for g in f["gt"]:
        np.testing.assert_allclose(
            gt_pmf(g["mean"], g["family"], g["cv"]), g["pmf"], rtol=1e-10
        )
    for g in f["incubation"]:
        np.testing.assert_allclose(
            incubation_pmf(g["mean"], g["cv"]), g["pmf"], rtol=1e-10
        )


@pytest.mark.parametrize("history", [False, True])
def test_reference_events_match_the_r_engine(history: bool) -> None:
    c = load("pmfs")["convolve"]
    pmfs = [np.array(p) for p in c["pmfs"]]
    asc = np.array([0.5, 1.0, 0.2])
    events = ReferenceEvents(
        "ref",
        delay_rvs=[
            DeterministicPMF(f"delay_{i}", p) for i, p in enumerate(pmfs)
        ],
        ascertainment_rv=DeterministicVariable("asc", asc),
    )
    before = np.array(c["before"]) if history else None
    out = run(events.sample, matrix(c["x"]), before=before)
    want = matrix(c["out_before"] if history else c["out"]) * asc
    np.testing.assert_allclose(out, want, rtol=1e-12)


def test_an_established_stratum_opens_at_its_ascertained_level() -> None:
    # A stratum at a constant level before and during the span has
    # constant expected reference events from day 1.
    pmf = incubation_pmf(9.0)
    events = ReferenceEvents(
        "ref",
        delay_rvs=[DeterministicPMF("d", pmf)],
        ascertainment_rv=DeterministicVariable("asc", np.array([0.17])),
    )
    out = run(
        events.sample, np.full((30, 1), 1503.0), before=np.array([1503.0])
    )
    np.testing.assert_allclose(out, 1503.0 * 0.17, rtol=1e-12)


def _mixing(f: dict[str, Any], case: dict[str, Any]) -> PlaceMixing:
    v = f["values"]
    return PlaceMixing(
        "mixing",
        kind=case["mixing"],
        share=v["mixing_eps"],
        order=[i - 1 for i in v["mixing_order"]],
        xy=np.stack([v["mixing_x"], v["mixing_y"]], axis=1),
        distance_range=v["mixing_range"],
        quiet=[v["quiet_places"] - 1],
        switch=case["switch"],
        switch_at=v["mixing_switch_at"],
        switch_days=v["mixing_switch_days"],
    )


def test_place_mixing_matches_the_r_engine() -> None:
    f = load("mixing")
    strata = pd.DataFrame(f["strata"])
    places = sorted(strata["place"].unique())
    place = strata["place"].map(places.index).to_numpy()
    age = strata["age"].to_numpy()
    for case in f["cases"]:
        K, eps = run(_mixing(f, case).sample, f["n"], place=place, age=age)
        np.testing.assert_allclose(
            K, matrix(case["K"]), rtol=1e-12, err_msg=case["mixing"]
        )
        np.testing.assert_allclose(
            eps, case["eps"], rtol=1e-12, err_msg=case["mixing"]
        )


def test_count_mixing_moves_infections_between_strata() -> None:
    # A row's giving share is eps times its row sum, as r7: a row whose
    # kernel row sums to a half gives away half as much.
    K = jnp.array([[0.0, 0.5], [1.0, 0.0]])
    force = jnp.array([10.0, 4.0])
    W = count_mixing(K, jnp.array([0.2]))[0]
    np.testing.assert_allclose(
        W @ force, force + 0.2 * (K @ force - K.sum(axis=1) * force)
    )


def test_imports_match_the_r_engine() -> None:
    f = load("imports")
    strata = pd.DataFrame(f["strata"])
    v = f["values"]
    for case in f["cases"]:
        late = (
            strata["place"].isin([f"P{i:02d}" for i in [v["late_places"]]])
            & (case["seeding"] == "mid")
        ).to_numpy()
        imports = Imports(
            "imports",
            kind=case["imports"],
            rate=v["import_rate"],
            pulse_at=v["import_at"],
            pulse_size=v["import_size"],
            late=late,
            late_at=v["late_at"],
            late_days=v["late_days"],
        )
        np.testing.assert_allclose(
            run(imports.sample, f["n"], unit=np.array(f["share"])),
            matrix(case["out"]),
            rtol=1e-12,
            err_msg=f"{case['imports']} {case['seeding']}",
        )


def test_the_trickle_reaches_every_stratum_with_cases_but_late_ones() -> None:
    late = np.array([False, True, False])
    imports = Imports("imports", kind="none", late=late, trickle=0.3)
    out = np.asarray(run(imports.sample, 10, unit=np.array([1.0, 1.0, 0.0])))
    np.testing.assert_allclose(out[:, 0], 0.3)
    np.testing.assert_allclose(out[:, 2], 0.0)
    # The late stratum's own import is its unit over the late days.
    assert out[:, 1].sum() == pytest.approx(1.0)


def _renewal(f: dict[str, Any], which: str) -> Any:
    start, log_r = matrix(f["start"]), matrix(f["log_r"])
    kwargs: dict[str, Any] = {}
    if which == "full":
        s, v = f["shifts"], f["variants"]
        kwargs = {
            "mixing": (matrix(f["K"]), np.array(f["eps"])),
            "imports": matrix(f["imports"]),
            "shifts": Shifts(
                at=s["at"],
                size=s["size"],
                days=s["days"],
                reactive=s["reactive"],
                up=s["up"],
            ),
            "variants": Variants(
                at=[v["at"]],
                escape=[v["escape"]],
                multiplier=[v["mult"]],
                days=[v["days"]],
            ),
        }
    if which == "exhaust":
        start, log_r = start * 50, log_r + 1.5
    with jax.enable_x64(True):
        return strata_renewal(
            start,
            np.array(f["gt"]),
            np.exp(log_r),
            np.array(f["population"]),
            **kwargs,
        )


@pytest.mark.parametrize("which", ["plain", "full", "exhaust"])
def test_the_strata_renewal_matches_the_r_engine(which: str) -> None:
    f = load("renewal")
    out = _renewal(f, which)
    want = f["runs"][which]
    for key in ("infections", "r", "reff", "susceptible"):
        np.testing.assert_allclose(
            np.asarray(getattr(out, key)),
            matrix(want[key]),
            rtol=1e-9,
            atol=1e-9,
            err_msg=key,
        )


def test_the_exhausted_renewal_never_infects_more_than_its_population() -> None:
    f = load("renewal")
    out = _renewal(f, "exhaust")
    pop = np.array(f["population"])
    assert (np.asarray(out.infections).sum(axis=0) <= pop + 1e-6).all()
    assert (np.asarray(out.susceptible) >= 0).all()


def test_exponential_depletion_is_pyrenews_logistic_adjustment() -> None:
    # S (1 - e^{-R F / N}) is PyRenew's logistic susceptibility adjustment
    # with the raw incidence R F, written with expm1.
    with jax.enable_x64(True):
        r = jnp.array([1.3, 0.8, 2.0])
        force = jnp.array([50.0, 3.0, 400.0])
        s = jnp.array([9000.0, 4000.0, 15000.0])
        pop = jnp.array([1e4, 5e3, 2e4])
        new = ExponentialDepletion()(r, force, s, pop, jnp.zeros(3))
        want = logistic_susceptibility_adjustment(r * force, s / pop, pop)
    np.testing.assert_allclose(new, want, rtol=1e-9)


def test_exponential_depletion_is_linear_while_few_are_infected() -> None:
    with jax.enable_x64(True):
        r, force = jnp.array([1.5]), jnp.array([2.0])
        s, pop, im = jnp.array([1e6]), jnp.array([1e6]), jnp.array([0.5])
        lin = LinearDepletion()(r, force, s, pop, im)
        exp = ExponentialDepletion()(r, force, s, pop, im)
    np.testing.assert_allclose(exp, lin, rtol=1e-5)


def test_both_depletions_stop_at_the_susceptibles() -> None:
    with jax.enable_x64(True):
        r, force = jnp.array([5.0]), jnp.array([1e5])
        s, pop, im = jnp.array([10.0]), jnp.array([1e3]), jnp.array([20.0])
        for depletion in (LinearDepletion(), ExponentialDepletion()):
            assert float(depletion(r, force, s, pop, im)[0]) == 10.0


def test_a_variant_raises_r_and_returns_immune_people_once() -> None:
    v = Variants(at=[10.0], escape=[0.5], multiplier=[2.0], days=[4.0])
    with jax.enable_x64(True):
        m = np.asarray(v.multiplier_on(jnp.arange(1.0, 31.0)))
    assert m[0] == pytest.approx(1.0, abs=1e-4)
    assert m[-1] == pytest.approx(2.0, abs=1e-4)
    assert m[9] == pytest.approx(1.5)


def test_the_fitted_models_do_not_import_the_simulator_components() -> None:
    # The fit keys hash the forecasters' import closure, so the simulator's
    # components stay out of it and adding them refits nothing.
    code = (
        "import sys, nowcastrenewal.forecaster, nowcastrenewal.zones;"
        "mods = ('imports', 'mixing', 'pmfs', 'reference', 'strata');"
        "print([m for m in mods"
        " if f'nowcastrenewal.latent.{m}' in sys.modules])"
    )
    out = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        check=True,
    )
    assert out.stdout.strip() == "[]"
