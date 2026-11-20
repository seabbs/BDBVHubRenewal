"""The simulator's design and value draws (`nowcastfoundation.simulator.draws`).

Checked against the grid's own shares and sampling distributions, and
against the R engine's `draw_design`/`draw_values`, called through Rscript.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from collections import Counter
from pathlib import Path

import numpy as np
import pytest
from scipy import stats

from nowcastfoundation.simulator import draws

ROOT = Path(__file__).parents[2]
GRID_FILE = ROOT / "config" / "finetune" / "grid.yaml"
ENGINE = ROOT / "packages" / "nowcastfoundation" / "r" / "R"
OPTIONS = (21, 28, 42, 56, 84)


@pytest.fixture(scope="module")
def grid():
    return draws.read_grid(GRID_FILE)


def _scipy(d):
    match d["dist"]:
        case "lognormal":
            return stats.lognorm(d["sdlog"], scale=d["median"])
        case "normal":
            return stats.norm(d["mean"], d["sd"])
        case "halfnormal":
            return stats.halfnorm(scale=d["scale"])
        case "gamma":
            shape = (d["mean"] / d["sd"]) ** 2
            return stats.gamma(shape, scale=d["sd"] ** 2 / d["mean"])
        case "beta":
            return stats.beta(
                d["a"], d["b"], loc=d["lower"], scale=d["upper"] - d["lower"]
            )
        case "uniform":
            return stats.uniform(d["lower"], d["upper"] - d["lower"])
    raise AssertionError(d)


def test_factor_levels_follow_their_weights(grid):
    rng = draws.generator(1)
    n = 20_000
    cells = [draws.draw_cell(grid, rng) for _ in range(n)]
    for name, levels in grid["factors"].items():
        w = np.asarray(list(levels.values()), dtype=float)
        p = w / w.sum()
        seen = Counter(c[name] for c in cells)
        assert set(seen) <= set(levels)
        for level, share in zip(levels, p, strict=True):
            se = np.sqrt(share * (1 - share) / n)
            assert abs(seen[level] / n - share) < 5 * se, (name, level)


def test_every_setting_follows_its_sampling_distribution(grid):
    rng = draws.generator(2)
    for name, d in grid["distributions"].items():
        x = draws.draw_setting(grid, name, rng, 5_000)
        assert stats.kstest(x, _scipy(d).cdf).pvalue > 1e-4, name
    assert isinstance(draws.draw_setting(grid, "gt_mean", rng), float)


def test_counts_are_their_base_plus_a_poisson(grid):
    rng = draws.generator(3)
    for name, k in grid["counts"].items():
        x = draws.draw_count(grid, name, rng, 5_000)
        assert x.min() >= k["base"], name
        se = np.sqrt(max(k["mean"], 1e-9) / 5_000)
        assert abs(x.mean() - k["base"] - k["mean"]) < 5 * se + 1e-9, name


def test_unknown_names_are_refused(grid):
    rng = draws.generator(4)
    with pytest.raises(KeyError):
        draws.draw_setting(grid, "nope", rng)
    with pytest.raises(KeyError):
        draws.draw_count(grid, "nope", rng)
    with pytest.raises(KeyError):
        draws.draw_share(grid, "nope", rng)


def test_keyed_generators_repeat_and_differ(grid):
    a = draws.draw_design(grid, draws.generator(7, 1, 2))
    b = draws.draw_design(grid, draws.generator(7, 1, 2))
    assert a == b
    others = [
        draws.draw_design(grid, draws.generator(7, 1, k)) for k in range(20)
    ]
    assert len({json.dumps(o, sort_keys=True) for o in others}) > 1


def test_designs_fit_the_budget_and_report_on_their_days(grid):
    rng = draws.generator(5)
    for _ in range(3_000):
        d = draws.draw_design(grid, rng)
        assert d["weeks"] >= grid["floors"]["weeks"]
        n_p, n_a = d["n_places"], d["n_ages"]
        if d["strata_link"] != "independent":
            assert n_p * n_a <= draws.max_strata(grid, d["weeks"])
            assert (
                draws.system_locations(n_p, n_a, grid["regions"])
                <= grid["max_locations"]
            )
        else:
            m = grid["max_independent"]
            if d["strata"] == "places_ages":
                assert n_p == 2
                assert n_a <= m // 2
            if d["strata"] == "ages":
                assert n_a <= m
        days = d["report_weekdays"]
        match d["report_days"]:
            case "daily":
                assert days == "all"
            case "weekdays":
                assert days == "Monday;Tuesday;Wednesday;Thursday;Friday"
            case "fixed_days":
                got = days.split(";")
                assert len(got) in (2, 3)
                assert got == sorted(got, key=draws.WEEK.index)
            case "weekly":
                assert days in draws.WEEK
    assert draws.draw_design(grid, rng, weeks=9)["weeks"] == 9


def test_values_have_the_engine_s_shapes(grid):
    rng = draws.generator(6)
    for _ in range(1_000):
        d = draws.draw_design(grid, rng)
        v = draws.draw_values(grid, d, rng)
        n_p, n_s = d["n_places"], d["n_places"] * d["n_ages"]
        for key in (
            "ascertainment",
            "incubation_mean",
            "obs_sqrt_phi",
            "obs_family",
            "revise_over",
            "revise_move",
            "revise_keep",
        ):
            assert len(v[key]) == n_s, key
        for key in ("population", "place_lags", "mixing_x", "mixing_y"):
            assert len(v[key]) == n_p, key
        assert sorted(v["mixing_order"]) == list(range(1, n_p + 1))
        assert len(v["walk_r"]) == d["weeks"] + 18
        assert len(v["delay_walk"]) == d["weeks"] + 18
        assert len(v["wave_scale"]) == v["waves"] + 1
        assert len(v["quiet_places"]) < n_p
        cases = set(range(1, n_p + 1)) - set(v["quiet_places"])
        assert set(v["late_places"]) <= cases - {min(cases)}
        assert set(v["shock_places"]) <= cases
        assert (d["shocks"] == "none") <= (v["shock_places"] == [])
        n = len(v["shock_places"])
        assert len(v["shock_at"]) == len(v["shock_mult"]) == n
        assert all(0 < a < 1 for a in v["ascertainment"])
        assert v["max_delay"] in OPTIONS
        assert 0 <= v["offset"] <= 6
        if d["window"] == "early":
            assert v["variant_at"] == []
    one = draws.draw_values(grid, {**d, "offset": 3}, rng)
    assert one["offset"] == 3


def test_the_longest_delay_is_the_first_option_past_the_99th_centile():
    assert draws.max_delay_for(5) == 21
    assert draws.max_delay_for(21) == 21
    assert draws.max_delay_for(21.5) == 28
    assert draws.max_delay_for(57) == 84
    assert draws.max_delay_for(500) == 84
    # R's qgeom counts failures: qgeom(0.99, 0.5) is 6.
    assert draws.delay_q99("none", {"delay_mean": 1.0}) == 6


R_DRAWS = """
for (f in c("grid.R", "design.R", "infections.R", "delays.R")) {
  source(file.path(Sys.getenv("ENGINE"), f))
}
grid <- read_grid(Sys.getenv("GRID_FILE"))
set.seed(11)
n <- as.integer(Sys.getenv("N"))
for (i in seq_len(n)) {
  d <- draw_design(grid)
  v <- draw_values(grid, d)
  cat(jsonlite::toJSON(list(design = d, values = v), digits = NA), "\\n")
}
"""


def _r_draws(n):
    out = subprocess.run(
        ["Rscript", "-e", R_DRAWS],
        env={
            "ENGINE": str(ENGINE),
            "GRID_FILE": str(GRID_FILE),
            "N": str(n),
            "PATH": os.environ["PATH"],
            "HOME": os.environ.get("HOME", ""),
        },
        capture_output=True,
        text=True,
        timeout=900,
        check=True,
    )
    lines = out.stdout.splitlines()
    return [json.loads(line) for line in lines if line.strip()]


def _listed(x):
    return x if isinstance(x, list) else [x]


def _needs_r():
    if shutil.which("Rscript") is None:
        return "needs Rscript"
    probe = subprocess.run(
        [
            "Rscript",
            "-e",
            "quit(status = !all(sapply(c('yaml', 'jsonlite'), "
            "requireNamespace, quietly = TRUE)))",
        ],
        capture_output=True,
    )
    return None if probe.returncode == 0 else "needs R packages yaml, jsonlite"


@pytest.mark.skipif(_needs_r() is not None, reason=_needs_r() or "")
def test_draws_match_the_r_engine_s(grid):
    n = 3_000
    r = _r_draws(n)
    rng = draws.generator(12)
    py = []
    for _ in range(n):
        d = draws.draw_design(grid, rng)
        py.append({"design": d, "values": draws.draw_values(grid, d, rng)})
    for part in ("design", "values"):
        assert set(r[0][part]) == set(py[0][part]), part
        for key in py[0][part]:
            a = [_listed(x[part][key]) for x in r]
            b = [_listed(x[part][key]) for x in py]
            # The same number of elements per draw, on average.
            la = np.array([len(x) for x in a], dtype=float)
            lb = np.array([len(x) for x in b], dtype=float)
            se = np.sqrt((la.var() + lb.var()) / n) + 1e-9
            assert abs(la.mean() - lb.mean()) < 5 * se, (part, key)
            # One number per draw (its first element, and its mean), as a
            # draw's elements are not independent.
            ea = [x[0] for x in a if x]
            eb = [x[0] for x in b if x]
            if not ea or not eb:
                assert not ea, (part, key)
                assert not eb, (part, key)
                continue
            if isinstance(ea[0], str | bool):
                ca, cb = Counter(ea), Counter(eb)
                for level in set(ca) | set(cb):
                    pa, pb = ca[level] / len(ea), cb[level] / len(eb)
                    p = (ca[level] + cb[level]) / (len(ea) + len(eb))
                    se = np.sqrt(p * (1 - p) * (1 / len(ea) + 1 / len(eb)))
                    assert abs(pa - pb) < 5 * se + 1e-9, (part, key, level)
                continue
            ma = [np.mean(x) for x in a if x]
            mb = [np.mean(x) for x in b if x]
            for xa, xb in ((ea, eb), (ma, mb)):
                ks = stats.ks_2samp(
                    np.asarray(xa, float), np.asarray(xb, float)
                )
                assert ks.pvalue > 1e-4, (part, key, ks)
