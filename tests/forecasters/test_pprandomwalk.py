"""The partially pooled generative random walk nowcast."""

from __future__ import annotations

import jax
import numpy as np
import pandas as pd
import pytest
from scipy.stats import lognorm

from bdbvhubrenewal.data.snapshots import SnapshotArrays
from bdbvhubrenewal.evaluation.backtest import BacktestConfig, make_tasks
from bdbvhubrenewal.forecasters import (
    ForecastTask,
    PPRandomWalkNowcast,
    RandomWalkNowcast,
    check_samples,
    get_model,
    list_models,
)
from bdbvhubrenewal.forecasters.chain import Settings
from bdbvhubrenewal.forecasters.pprandomwalk import (
    NotConverged,
    Sampler,
    Tree,
    _live_tree,
    fit_posterior,
    reported_share,
    triangle,
)

SMALL = Sampler(
    num_warmup=40, num_samples=40, num_chains=1, max_rhat=np.inf,
    max_divergent=1.0,
)  # fmt: skip
"""A sampler for shape tests, with the convergence check off."""

D = 42


def _task(
    target: pd.DataFrame,
    delays: pd.DataFrame,
    daily: pd.DataFrame,
    vintage: str,
) -> ForecastTask:
    rows = target.loc[target["vintage"] == vintage]
    return make_tasks(rows, delays, BacktestConfig(seed=3), daily=daily)[0]


@pytest.fixture(scope="module")
def task(
    target: pd.DataFrame, delays: pd.DataFrame, daily: pd.DataFrame
) -> ForecastTask:
    return _task(target, delays, daily, "2026-11-01")


@pytest.fixture(scope="module")
def fitted(task: ForecastTask) -> tuple[PPRandomWalkNowcast, pd.DataFrame]:
    model = PPRandomWalkNowcast(sampler=SMALL)
    return model, model.forecast(task)


def _simulate(
    seed: int,
    mu: float,
    sd: float,
    sigma: float,
    n_leaves: int = 3,
    n_weeks: int = 20,
    level: float = 150.0,
) -> SnapshotArrays:
    """Simulate release snapshots from known delays and a walk.

    Each leaf's weekly expected count is a random walk on the log scale
    with spread `sigma`. Each day's cases are Poisson on a seventh of it,
    and each case is reported after a log-normal delay (`mu`, `sd`), cut
    at `D` days. Releases see the cases reported by their cut-off.
    """
    rng = np.random.default_rng(seed)
    n_days = 7 * n_weeks
    week = np.arange(n_days) // 7
    steps = rng.normal(0.0, sigma, (n_leaves, n_weeks - 1))
    log_lam = np.log(level) + np.concatenate(
        [np.zeros((n_leaves, 1)), np.cumsum(steps, axis=1)], axis=1
    )
    cases = rng.poisson(np.exp(log_lam[:, week]) / 7)
    cut = np.array([n_days - 71, n_days - 36, n_days - 29, n_days - 22,
                    n_days - 15, n_days - 8, n_days - 3])  # fmt: skip
    counts = np.zeros((n_leaves + 1, cut.size, n_days))
    for p in range(n_leaves):
        for t in np.flatnonzero(cases[p]):
            x = lognorm.rvs(
                sd, scale=np.exp(mu), size=cases[p, t], random_state=rng
            )
            report = t + np.minimum(np.floor(x), D)
            counts[p + 1, :, t] = (report[None, :] <= cut[:, None]).sum(axis=1)
    observed = np.arange(n_days)[None, :] <= cut[:, None]
    counts = counts * observed[None]
    counts[0] = counts[1:].sum(axis=0)
    days = pd.date_range("2026-03-01", periods=n_days, freq="D")
    ends = tuple(
        (days[0] + pd.Timedelta(days=7 * w + 6)).date().isoformat()
        for w in range(n_weeks)
    )
    names = ("CD", *(f"CD-P{i}" for i in range(n_leaves)))
    releases = tuple(days[c].date().isoformat() for c in cut)
    parent = np.array([-1] + [0] * n_leaves)
    return SnapshotArrays(
        names, releases, days, counts, observed, cut, week, ends, parent
    )


def test_registered_as_a_fitted_randomwalknowcast() -> None:
    spec = {s.name: s for s in list_models()}["pprandomwalknowcast"]
    assert spec.fitted
    assert not spec.baseline
    assert "generative" in spec.description
    model = get_model("pprandomwalknowcast")
    assert isinstance(model, PPRandomWalkNowcast)
    assert isinstance(model, RandomWalkNowcast)


def test_sampler_defaults_are_two_chains_of_500_and_750() -> None:
    s = Sampler()
    assert (s.num_chains, s.num_warmup, s.num_samples) == (2, 500, 750)
    assert s.max_rhat == 1.05
    assert s.max_divergent == 0.01


def test_tree_levels_paths_and_leaves() -> None:
    # Country, two provinces, and two zones in the first province.
    tree = Tree.of(np.array([-1, 0, 0, 1, 1]))
    assert tree.level.tolist() == [0, 1, 1, 2, 2]
    assert tree.leaf.tolist() == [False, False, True, True, True]
    # A zone's value adds its province's offset and its own.
    assert tree.path[3].tolist() == [0, 1, 0, 1, 0]
    assert tree.path[0].sum() == 0


def test_reported_share_is_a_cdf_that_reaches_one_at_the_maximum_delay() -> (
    None
):
    ages = np.arange(-1, D + 5)
    pi = np.asarray(reported_share(1.8, 0.7, ages, D))
    assert pi[0] == 0.0
    assert (np.diff(pi) >= 0).all()
    assert pi[ages == D][0] == pytest.approx(1.0)
    assert (pi[ages >= D] == pi[ages == D][0]).all()
    # Discretised log-normal: day d is reported by age a when d <= a,
    # so the share at age a is F(a + 1) / F(D + 1).
    want = lognorm.cdf(6.0, 0.7, scale=np.exp(1.8)) / lognorm.cdf(
        D + 1.0, 0.7, scale=np.exp(1.8)
    )
    assert pi[ages == 5][0] == pytest.approx(want, rel=1e-5)


def test_triangle_holds_each_release_increment_of_the_leaves() -> None:
    snap = _simulate(0, 1.5, 0.6, 0.1, n_leaves=2, n_weeks=12)
    settings = Settings()
    o = len(snap.releases) - 1
    tri = triangle(snap, o, settings, fit_weeks=8)
    assert tri.leaves.tolist() == [1, 2]
    assert tri.first == snap.cut_week(o) - 7
    assert tri.n_weeks == 8
    # Every case of the leaves on the fitted days is in one cell, unless
    # it came after the day was D days old at an earlier release.
    day = np.arange(snap.week.size)
    in_fit = snap.week >= tri.first
    total = snap.counts[1:, o][:, in_fit & (day <= snap.cut[o])].sum()
    assert tri.count.sum() == pytest.approx(total)
    assert (tri.count >= 0).all()
    assert (tri.lo < tri.hi).all()
    assert (tri.lo >= -1).all()
    assert (tri.hi <= D).all()
    assert ((tri.week >= 0) & (tri.week < tri.n_weeks)).all()


def test_triangle_drops_reports_after_the_maximum_delay() -> None:
    snap = _simulate(1, 1.5, 0.6, 0.1, n_leaves=1, n_weeks=20)
    # A late report of an old day, after it was D days old at the
    # second release: the model gives it no probability.
    counts = snap.counts.copy()
    t = int(snap.cut[1]) - D - 3
    counts[:, 2:, t] += 5.0
    late = SnapshotArrays(
        snap.locations, snap.releases, snap.days, counts, snap.observed,
        snap.cut, snap.week, snap.week_ends, snap.parent,
    )  # fmt: skip
    o = len(snap.releases) - 1
    base = triangle(snap, o, Settings(), fit_weeks=16)
    tri = triangle(late, o, Settings(), fit_weeks=16)
    assert tri.count.sum() == base.count.sum()


@pytest.mark.slow
def test_recovers_a_known_delay_and_walk_spread() -> None:
    mu, sd, sigma = 1.8, 0.6, 0.25
    snap = _simulate(2, mu, sd, sigma, n_leaves=4, n_weeks=20)
    o = len(snap.releases) - 1
    tri = triangle(snap, o, Settings(), fit_weeks=16)
    post = fit_posterior(
        tri,
        Tree.of(snap.parent),
        Sampler(num_warmup=300, num_samples=300, num_chains=2),
        jax.random.key(1),
        D,
    )
    assert post.info["divergences"] <= 3
    assert post.info["worst_rhat"] < 1.05
    # The share reported by each age is close to the simulated delay's.
    ages = np.array([2, 5, 10, 20])
    truth = lognorm.cdf(ages + 1.0, sd, scale=np.exp(mu))
    share = np.median(post.share[:, tri.leaves][..., ages], axis=0)
    assert np.abs(share - truth[None, :]).max() < 0.08
    spread = np.median(post.sigma[:, 0])
    assert 0.5 * sigma < spread < 2.0 * sigma


def test_a_fit_that_does_not_converge_raises() -> None:
    snap = _simulate(3, 1.5, 0.6, 0.2, n_leaves=2, n_weeks=12)
    o = len(snap.releases) - 1
    tri = triangle(snap, o, Settings(), fit_weeks=8)
    with pytest.raises(NotConverged):
        fit_posterior(
            tri,
            Tree.of(snap.parent),
            Sampler(num_warmup=5, num_samples=20, num_chains=2),
            jax.random.key(0),
            D,
        )


def test_samples_cover_every_place_and_week(
    task: ForecastTask, fitted: tuple[PPRandomWalkNowcast, pd.DataFrame]
) -> None:
    _, out = fitted
    check_samples(out, task)
    assert (out["value"] >= 0).all()
    assert set(out["location"]) == set(task.locations)
    assert set(out["target_end_date"]) == set(task.target_end_dates)
    # More samples than posterior draws: each sample takes a draw.
    assert out["sample"].nunique() == task.n_samples


def test_samples_vary_with_the_posterior(
    fitted: tuple[PPRandomWalkNowcast, pd.DataFrame],
) -> None:
    _, out = fitted
    it = out.loc[out["location"] == "CD-IT"]
    last = it.loc[it["target_end_date"] == it["target_end_date"].max()]
    assert last["value"].std() > 0


def test_delay_summary_is_the_fitted_delay(
    task: ForecastTask, fitted: tuple[PPRandomWalkNowcast, pd.DataFrame]
) -> None:
    model, _ = fitted
    summary = model.delay_summary
    assert summary is not None
    assert set(summary["location"]) >= set(task.locations)
    assert set(summary["quantity"]) == {"reported_share"}
    assert summary["x"].max() == model.settings.max_delay
    for _, grp in summary.groupby("location"):
        share = grp.sort_values("x")["q50"].to_numpy()
        assert (np.diff(share) >= -1e-12).all()
        assert share[-1] == pytest.approx(1.0)
    # It is pp's own posterior, not the chain ladder of
    # randomwalknowcast.
    rw = RandomWalkNowcast()
    rw.forecast(task)
    assert rw.delay_summary is not None
    a = summary.set_index(["location", "x"])["q50"]
    b = rw.delay_summary.set_index(["location", "x"])["q50"]
    assert not np.allclose(a.loc[b.index], b)


def test_diagnostics_hold_the_fit_and_each_place(
    task: ForecastTask, fitted: tuple[PPRandomWalkNowcast, pd.DataFrame]
) -> None:
    model, _ = fitted
    fits = [r for r in model.fit_diagnostics if "fit" in r]
    assert len(fits) == 1
    for key in ("worst_rhat", "min_ess", "divergences", "seconds"):
        assert key in fits[0]
    places = [r for r in model.fit_diagnostics if "location" in r]
    assert [r["location"] for r in places] == list(task.locations)
    assert all("sigma" in r and "delay_median" in r for r in places)


def test_triangle_fits_only_the_leaves_asked_for() -> None:
    snap = _simulate(4, 1.5, 0.6, 0.1, n_leaves=3, n_weeks=12)
    o = len(snap.releases) - 1
    tri = triangle(snap, o, Settings(), fit_weeks=8, leaves=np.array([3]))
    assert tri.leaves.tolist() == [3]
    assert set(tri.place.tolist()) == {0}
    full = triangle(snap, o, Settings(), fit_weeks=8)
    assert tri.count.sum() == full.count[full.place == 2].sum()


def test_silent_places_are_not_fitted_and_draw_introductions(
    fitted: tuple[PPRandomWalkNowcast, pd.DataFrame],
) -> None:
    model, out = fitted
    places = {
        r["location"]: r for r in model.fit_diagnostics if "location" in r
    }
    assert places["CD-KN"]["silent"]
    assert not places["CD-IT"]["silent"]
    kn = out.loc[out["location"] == "CD-KN", "value"]
    assert kn.median() == 0


def test_the_fit_tree_holds_only_live_leaves_and_their_ancestors() -> None:
    # Country, two provinces, and two zones in the first province. Only
    # the first zone has recent cases.
    parent = np.array([-1, 0, 0, 1, 1])
    sub, pos, src = _live_tree(parent, np.array([3]))
    assert pos.tolist() == [0, 1, -1, 2, -1]
    assert sub.parent.tolist() == [-1, 0, 1]
    # A place left out takes its nearest ancestor in the tree.
    assert src.tolist() == [0, 1, 0, 2, 1]


def test_a_silent_place_takes_its_ancestors_delay(
    fitted: tuple[PPRandomWalkNowcast, pd.DataFrame],
) -> None:
    model, _ = fitted
    summary = model.delay_summary
    assert summary is not None
    share = summary.pivot(index="x", columns="location", values="q50")
    np.testing.assert_allclose(share["CD-KN"], share["CD"])
