"""The JAX latent simulator against the R engine's infection side.

The fixtures come from `scripts/latent_fixtures.R`, which runs the R
functions in `packages/nowcastfoundation/r/R/infections.R`.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import jax
import numpy as np
import pandas as pd
import pytest
import yaml

from bdbvhubrenewal.model import simulate_latent as sl

FIXTURES = Path(__file__).parent / "fixtures" / "latent"
GRID = Path(__file__).resolve().parents[2] / "config/finetune/grids/r7.yaml"
LATENT = sorted(FIXTURES.glob("latent-*.json"))


def load(name: str) -> dict[str, Any]:
    return json.loads((FIXTURES / f"{name}.json").read_text())


def matrix(x: Any) -> np.ndarray:
    return np.array(x, dtype=float)


@pytest.fixture(scope="module")
def grid() -> dict[str, Any]:
    return yaml.safe_load(GRID.read_text())


def test_pmfs_match_the_r_engine() -> None:
    f = load("pmfs")
    for g in f["gt"]:
        np.testing.assert_allclose(
            sl.gt_pmf(g["mean"], g["family"], g["cv"]), g["pmf"], rtol=1e-10
        )
    for g in f["incubation"]:
        np.testing.assert_allclose(
            sl.incubation_pmf(g["mean"], g["cv"]), g["pmf"], rtol=1e-10
        )
    c = f["convolve"]
    pmfs = [np.array(p) for p in c["pmfs"]]
    np.testing.assert_allclose(
        sl.convolve_rows(matrix(c["x"]), pmfs), matrix(c["out"]), rtol=1e-12
    )


def test_paths_match_the_r_engine() -> None:
    f = load("paths")
    for kind, path in f["paths"].items():
        np.testing.assert_allclose(
            sl.path_log_r(f["n"], kind, f["values"]),
            path,
            rtol=1e-12,
            atol=1e-15,
            err_msg=kind,
        )
    w = f["daily_walk"]
    np.testing.assert_allclose(
        sl.daily_walk(np.array(w["days"]), np.array(w["walk"])),
        w["out"],
        rtol=1e-12,
    )


def test_mixing_matches_the_r_engine() -> None:
    f = load("mixing")
    strata = pd.DataFrame(f["strata"])
    for case in f["cases"]:
        values = {**f["values"], "mixing_switch": case["switch"]}
        K, eps = sl.place_mixing(
            f["n"], strata, {"mixing": case["mixing"]}, values
        )
        np.testing.assert_allclose(K, matrix(case["K"]), rtol=1e-12)
        np.testing.assert_allclose(eps, case["eps"], rtol=1e-12)


def test_imports_match_the_r_engine() -> None:
    f = load("imports")
    strata = pd.DataFrame(f["strata"])
    for case in f["cases"]:
        design = {"imports": case["imports"], "seeding": case["seeding"]}
        np.testing.assert_allclose(
            sl.stratum_imports(
                f["n"], np.array(f["share"]), strata, design, f["values"]
            ),
            matrix(case["out"]),
            rtol=1e-12,
        )


@pytest.mark.parametrize("run", ["plain", "full", "exhaust"])
def test_the_renewal_matches_the_r_engine(run: str) -> None:
    f = load("renewal")
    kwargs: dict[str, Any] = {}
    start = matrix(f["start"])
    log_r = matrix(f["log_r"])
    if run == "full":
        kwargs = {
            "mixing": (matrix(f["K"]), np.array(f["eps"])),
            "imports": matrix(f["imports"]),
            "shifts": f["shifts"],
            "variants": f["variants"],
        }
    if run == "exhaust":
        start, log_r = start * 50, log_r + 1.5
    out = sl.renewal_infections(
        start, np.array(f["gt"]), log_r, np.array(f["population"]), **kwargs
    )
    want = f["runs"][run]
    for key in ("infections", "r", "reff", "susceptible"):
        np.testing.assert_allclose(
            out[key], matrix(want[key]), rtol=1e-9, atol=1e-9, err_msg=key
        )


def test_the_exhausted_renewal_never_infects_more_than_its_population() -> None:
    f = load("renewal")
    out = sl.renewal_infections(
        matrix(f["start"]) * 50,
        np.array(f["gt"]),
        matrix(f["log_r"]) + 1.5,
        np.array(f["population"]),
    )
    assert (
        out["infections"].sum(axis=0) <= np.array(f["population"]) + 1e-6
    ).all()
    assert (out["susceptible"] >= 0).all()


@pytest.mark.parametrize("path", LATENT, ids=lambda p: p.stem)
def test_latent_runs_replay_the_r_engine(
    path: Path, grid: dict[str, Any]
) -> None:
    case = json.loads(path.read_text())
    design, values = case["design"], case["values"]
    gt = sl.gt_pmf(values["gt_mean"], design["gt_family"], values["gt_cv"])
    tape = sl.Tape(case["tape"]["normal"], case["tape"]["choice"])
    strata = pd.DataFrame(case["strata"])
    n = case["n_days"] - len(gt)
    if "error" in case:
        with pytest.raises(ValueError, match=case["error"]):
            sl.latent_onsets(grid, n, gt, strata, design, values, tape)
        return
    run = sl.latent_onsets(grid, n, gt, strata, design, values, tape)
    assert tape.finished()
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
    designs = [json.loads(p.read_text())["design"] for p in LATENT]
    ran = [
        d
        for p, d in zip(LATENT, designs, strict=True)
        if "error" not in json.loads(p.read_text())
    ]
    for factor, levels in {
        "mixing": {
            "none",
            "symmetric",
            "asymmetric",
            "distance",
            "time_varying",
        },
        "place_rt": {"shared", "independent", "correlated", "lagged"},
        "window": {"early", "established", "endemic"},
        "imports": {"none", "constant", "pulses"},
        "seeding": {"start", "mid"},
    }.items():
        assert {d[factor] for d in ran} == levels, factor


def test_exponential_depletion_is_linear_while_few_are_infected() -> None:
    f = load("renewal")
    args = (matrix(f["start"]), np.array(f["gt"]), matrix(f["log_r"]) - 0.5)
    pop = np.array(f["population"]) * 1e3
    lin = sl.renewal_infections(*args, pop)
    exp = sl.renewal_infections(*args, pop, depletion="exponential")
    np.testing.assert_allclose(
        exp["infections"], lin["infections"], rtol=1e-3, atol=1e-9
    )


def test_exponential_depletion_never_infects_more_than_its_population() -> None:
    f = load("renewal")
    out = sl.renewal_infections(
        matrix(f["start"]) * 50,
        np.array(f["gt"]),
        matrix(f["log_r"]) + 1.5,
        np.array(f["population"]),
        depletion="exponential",
    )
    assert (out["susceptible"] >= 0).all()


def test_an_unknown_depletion_is_refused() -> None:
    with pytest.raises(ValueError, match="depletion"):
        sl.renewal_infections(
            np.ones((2, 1)),
            np.array([0.5, 0.5]),
            np.zeros((3, 1)),
            np.array([10.0]),
            depletion="quadratic",
        )


def _draws() -> dict[str, Any]:
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
    short = {**sims[0], "id": "short", "n_days": 3}
    return {"seed": 7, "simulations": [*sims, short]}


def test_a_batch_writes_each_simulation_and_its_errors(
    grid: dict[str, Any], tmp_path: Path
) -> None:
    draws = _draws()
    (tmp_path / "draws.json").write_text(json.dumps(draws))
    sl.main(
        [
            "--grid",
            str(GRID),
            "--draws",
            str(tmp_path / "draws.json"),
            "--out",
            str(tmp_path / "out"),
        ]
    )
    long = pd.read_csv(tmp_path / "out" / "latent-7.csv.gz")
    values = json.loads((tmp_path / "out" / "latent-values-7.json").read_text())
    assert list(long.columns) == [
        "id",
        "day",
        "group",
        "infections",
        "onsets",
        "r",
        "reff",
    ]
    assert set(long["id"]) == {"s0", "s1"}
    k = len(draws["simulations"][0]["strata"])
    s0 = long[long["id"] == "s0"]
    assert set(s0["group"]) == set(range(k + 1))
    assert (
        s0.groupby("group").size().eq(draws["simulations"][0]["n_days"]).all()
    )
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


def test_a_key_gives_the_same_run_and_another_key_another(
    grid: dict[str, Any],
) -> None:
    draws = _draws()
    draws["simulations"] = draws["simulations"][:2]
    draws["simulations"][1]["key"] = 0
    long, _ = sl.simulate_batch(grid, draws)
    a = long[long["id"] == "s0"]["onsets"].to_numpy()
    b = long[long["id"] == "s1"]["onsets"].to_numpy()
    np.testing.assert_array_equal(a, b)
    draws["simulations"][1]["key"] = 5
    long, _ = sl.simulate_batch(grid, draws)
    b = long[long["id"] == "s1"]["onsets"].to_numpy()
    assert not np.allclose(a, b)


def test_a_single_draw_of_a_vector_value_may_be_a_scalar(
    grid: dict[str, Any],
) -> None:
    case = json.loads(LATENT[0].read_text())
    values = {
        k: (v[0] if isinstance(v, list) and len(v) == 1 else v)
        for k, v in case["values"].items()
    }
    gt = sl.gt_pmf(
        values["gt_mean"], case["design"]["gt_family"], values["gt_cv"]
    )
    run = sl.latent_onsets(
        grid,
        case["n_days"] - len(gt),
        gt,
        pd.DataFrame(case["strata"]),
        case["design"],
        values,
        sl.KeyStream(jax.random.key(1)),
    )
    assert np.isfinite(run.onsets).all()


def test_the_fitted_models_do_not_import_the_simulator() -> None:
    # Their fit keys hash the code they import, which stays as it was.
    from bdbvhubrenewal.evaluation import store

    root = Path(__file__).resolve().parents[2]
    for model in store.stored_models():
        code = store.fit_code(model, root)
        assert not any("simulate_latent" in f for f in code), model


def test_any_error_in_one_simulation_is_recorded_and_the_batch_goes_on(
    grid: dict[str, Any],
) -> None:
    # As the R batch's tryCatch, which redraws a simulation on any error.
    draws = _draws()
    broken = draws["simulations"][1]
    broken["values"] = {
        k: v for k, v in broken["values"].items() if k != "amplitude"
    }
    broken["design"] = {**broken["design"], "place_rt": "independent"}
    long, values = sl.simulate_batch(grid, draws)
    assert values["s1"]["error"].startswith("KeyError")
    assert values["s0"]["error"] is None
    assert set(long["id"]) == {"s0"}


def test_each_simulation_records_its_generation_time(
    grid: dict[str, Any],
) -> None:
    draws = _draws()
    _, values = sl.simulate_batch(grid, draws)
    sim = draws["simulations"][0]
    gt = sl.gt_pmf(
        sim["values"]["gt_mean"],
        sim["design"]["gt_family"],
        sim["values"]["gt_cv"],
    )
    np.testing.assert_allclose(values["s0"]["gt"], gt)
    assert values["s0"]["n_gt"] == len(gt)
