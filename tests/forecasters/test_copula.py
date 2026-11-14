"""The Gaussian copula that pairs a model's marginal draws."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy.special import ndtri
from scipy.stats import spearmanr

from bdbvhubrenewal.forecasters import NATIONAL
from bdbvhubrenewal.forecasters.copula import (
    GaussianCopula,
    RollingCopula,
    estimate_copula,
    normal_scores,
    pair_samples,
)


def _draws(n: int = 2000, places: int = 4, steps: int = 6) -> np.ndarray:
    """Independent draws whose marginals differ by cell."""
    rng = np.random.default_rng(0)
    scale = np.arange(1, places * steps + 1).reshape(places, steps)
    return rng.gamma(2.0, scale, size=(n, places, steps))


def _rank_corr(a: np.ndarray, b: np.ndarray) -> float:
    return float(spearmanr(a, b).statistic)


def test_each_cell_keeps_its_draws() -> None:
    draws = _draws()
    copula = GaussianCopula(0.8, 0.5, 0.6)
    out = copula.couple(draws, np.random.default_rng(1))
    assert out.shape == draws.shape
    assert np.array_equal(np.sort(out, axis=0), np.sort(draws, axis=0))


def test_draws_follow_the_correlations() -> None:
    copula = GaussianCopula(0.8, 0.5, 0.3)
    out = copula.couple(_draws(), np.random.default_rng(2))
    # Spearman's rank correlation of a Gaussian copula with correlation r
    # is 6 / pi * arcsin(r / 2).
    rank = {r: 6 / np.pi * np.arcsin(r / 2) for r in (0.8, 0.64, 0.5, 0.3)}
    assert _rank_corr(out[:, 1, 0], out[:, 1, 1]) == pytest.approx(
        rank[0.8], abs=0.05
    )
    assert _rank_corr(out[:, 1, 0], out[:, 1, 2]) == pytest.approx(
        rank[0.64], abs=0.05
    )
    assert _rank_corr(out[:, 1, 3], out[:, 2, 3]) == pytest.approx(
        rank[0.5], abs=0.05
    )
    assert _rank_corr(out[:, 0, 3], out[:, 3, 3]) == pytest.approx(
        rank[0.3], abs=0.05
    )
    # Across places and weeks the correlations multiply.
    assert _rank_corr(out[:, 1, 0], out[:, 2, 1]) == pytest.approx(
        6 / np.pi * np.arcsin(0.4 / 2), abs=0.05
    )


def test_no_correlation_leaves_the_draws_unpaired() -> None:
    out = GaussianCopula().couple(_draws(), np.random.default_rng(3))
    assert abs(_rank_corr(out[:, 1, 0], out[:, 1, 1])) < 0.06
    assert abs(_rank_corr(out[:, 0, 0], out[:, 1, 0])) < 0.06


def test_daily_steps_take_the_weekly_correlation_over_seven_days() -> None:
    copula = GaussianCopula(0.5, 0.0, 0.0)
    out = copula.couple(
        _draws(steps=15), np.random.default_rng(4), days=np.arange(15)
    )
    assert _rank_corr(out[:, 1, 0], out[:, 1, 7]) == pytest.approx(
        6 / np.pi * np.arcsin(0.5 / 2), abs=0.05
    )
    assert _rank_corr(out[:, 1, 0], out[:, 1, 1]) == pytest.approx(
        6 / np.pi * np.arcsin(0.5 ** (1 / 7) / 2), abs=0.05
    )


def test_an_impossible_place_correlation_still_pairs_the_draws() -> None:
    # Strong ties to the country and none between 26 provinces is no
    # correlation matrix, so the nearest one is used.
    draws = _draws(n=500, places=27, steps=2)
    out = GaussianCopula(0.3, 0.0, 0.9).couple(draws, np.random.default_rng(5))
    assert np.array_equal(np.sort(out, axis=0), np.sort(draws, axis=0))
    assert _rank_corr(out[:, 0, 0], out[:, 1, 0]) > 0.05


@pytest.mark.parametrize(
    "rho", [(-0.1, 0, 0), (0, 1.0, 0), (0, 0, -0.5), (1.0, 0, 0)]
)
def test_correlations_must_be_in_zero_to_one(rho) -> None:
    with pytest.raises(ValueError, match="between 0 and 1"):
        GaussianCopula(*rho)


def _forecasts(values: dict[tuple[str, str], np.ndarray]) -> pd.DataFrame:
    return pd.concat(
        pd.DataFrame(
            {
                "model": "m",
                "origin": "2026-10-24",
                "location": loc,
                "target_end_date": week,
                "sample": np.arange(1, len(v) + 1),
                "value": v,
            }
        )
        for (loc, week), v in values.items()
    )


def test_normal_scores_are_the_truths_place_in_the_forecast() -> None:
    forecasts = _forecasts(
        {
            ("CD", "2026-10-17"): np.arange(100.0),
            ("CD", "2026-10-24"): np.arange(100.0),
            ("CD-IT", "2026-10-17"): np.zeros(100),
        }
    )
    truth = pd.DataFrame(
        {
            "location": ["CD", "CD", "CD-IT"],
            "target_end_date": ["2026-10-17", "2026-10-24", "2026-10-17"],
            "truth": [49.5, 1000.0, 0.0],
        }
    )
    out = normal_scores(forecasts, truth, seed=0)
    z = out.set_index(["location", "target_end_date"])["z"]
    assert z[("CD", "2026-10-17")] == pytest.approx(0.0, abs=0.03)
    assert z[("CD", "2026-10-24")] > 2.3
    # A forecast with no spread says nothing of where the truth falls.
    assert ("CD-IT", "2026-10-17") not in z.index
    assert {"model", "origin", "location", "target_end_date"} <= set(out)


def test_estimate_recovers_the_correlations() -> None:
    truth = GaussianCopula(0.6, 0.4, 0.5)
    rng = np.random.default_rng(6)
    places = [NATIONAL, *(f"CD-{i:02d}" for i in range(5))]
    weeks = [f"2026-10-{d:02d}" for d in (3, 10, 17, 24)]
    rows = []
    for origin in range(400):
        z = truth.normals(rng, 1, len(places), 7 * np.arange(len(weeks)))[0]
        for i, loc in enumerate(places):
            for h, week in enumerate(weeks):
                rows.append(("m", str(origin), loc, week, z[i, h]))
    scores = pd.DataFrame(
        rows,
        columns=["model", "origin", "location", "target_end_date", "z"],
    )
    fit = estimate_copula(scores)
    assert fit.rho_time == pytest.approx(0.6, abs=0.05)
    assert fit.rho_provinces == pytest.approx(0.4, abs=0.05)
    assert fit.rho_national == pytest.approx(0.5, abs=0.05)


def test_estimate_of_negative_correlation_is_floored_at_zero() -> None:
    # Every pair of scores, across weeks and places, is of opposite sign.
    rows = [
        ("m", str(o), loc, week, z)
        for o in range(5)
        for loc, sign in ((NATIONAL, 1), ("CD-IT", -1))
        for week, z in (
            ("2026-10-17", sign * (o + 1.0)),
            ("2026-10-24", -sign * (o + 1.0)),
        )
    ]
    scores = pd.DataFrame(
        rows,
        columns=["model", "origin", "location", "target_end_date", "z"],
    )
    fit = estimate_copula(scores)
    assert fit.rho_time == 0.0
    assert fit.rho_national == 0.0


def test_a_truth_tied_with_draws_falls_among_them() -> None:
    draws = np.r_[np.zeros(60), np.ones(40)]
    forecasts = _forecasts({("CD", "2026-10-17"): draws})
    truth = pd.DataFrame(
        {"location": ["CD"], "target_end_date": ["2026-10-17"], "truth": [0.0]}
    )
    z = [
        normal_scores(forecasts, truth, seed=s)["z"].iloc[0] for s in range(200)
    ]
    # The PIT is uniform on (0, 61 / 101], so finite and below its top.
    assert np.isfinite(z).all()
    assert max(z) <= ndtri(61 / 101)
    assert min(z) < ndtri(0.1)


def test_weeks_with_no_truth_give_no_score() -> None:
    forecasts = _forecasts(
        {
            ("CD", "2026-10-17"): np.arange(100.0),
            ("CD", "2026-10-24"): np.arange(100.0),
        }
    )
    truth = pd.DataFrame(
        {
            "location": ["CD", "CD"],
            "target_end_date": ["2026-10-17", "2026-10-24"],
            "truth": [10.0, np.nan],
        }
    )
    out = normal_scores(forecasts, truth, seed=0)
    assert out["target_end_date"].tolist() == ["2026-10-17"]
    assert np.isfinite(out["z"]).all()


def test_steps_are_correlated_by_their_days_apart() -> None:
    out = GaussianCopula(0.8, 0.0, 0.0).couple(
        _draws(steps=3), np.random.default_rng(8), days=np.array([0, 7, 21])
    )
    assert _rank_corr(out[:, 1, 0], out[:, 1, 2]) == pytest.approx(
        6 / np.pi * np.arcsin(0.8**3 / 2), abs=0.05
    )


def test_with_no_country_every_place_is_a_province() -> None:
    out = GaussianCopula(0.0, 0.5, 0.0).couple(
        _draws(), np.random.default_rng(9), national=False
    )
    assert _rank_corr(out[:, 0, 2], out[:, 1, 2]) == pytest.approx(
        6 / np.pi * np.arcsin(0.5 / 2), abs=0.05
    )


def _structured_scores(n_origins: int = 400) -> pd.DataFrame:
    rng = np.random.default_rng(6)
    places = [NATIONAL, *(f"CD-{i:02d}" for i in range(5))]
    weeks = [f"2026-10-{d:02d}" for d in (3, 10, 17, 24)]
    copula = GaussianCopula(0.6, 0.4, 0.5)
    rows = []
    for origin in range(n_origins):
        z = copula.normals(rng, 1, len(places), 7 * np.arange(len(weeks)))
        for i, loc in enumerate(places):
            for h, week in enumerate(weeks):
                rows.append(("m", str(origin), loc, week, z[0, i, h]))
    return pd.DataFrame(
        rows,
        columns=["model", "origin", "location", "target_end_date", "z"],
    )


def test_estimate_shrinks_towards_the_prior() -> None:
    scores = _structured_scores()
    prior = GaussianCopula(0.1, 0.1, 0.1)
    raw = estimate_copula(scores)
    assert estimate_copula(scores, prior, weight=0.0) == raw
    # Each correlation counts its own pairs: of 6 places by 4 weeks at
    # 400 origins, 7,200 pairs of weeks and 16,000 of provinces.
    time = estimate_copula(scores, prior, weight=7200.0)
    assert time.rho_time == pytest.approx((raw.rho_time + 0.1) / 2)
    places = estimate_copula(scores, prior, weight=16000.0)
    assert places.rho_provinces == pytest.approx((raw.rho_provinces + 0.1) / 2)
    far = estimate_copula(scores, prior, weight=1e12)
    assert far.rho_time == pytest.approx(0.1)


def _samples(n: int = 1000, national: str = "own") -> pd.DataFrame:
    """Independent draws of three provinces over three weeks, and the
    country as its own series or as their sum."""
    rng = np.random.default_rng(10)
    weeks = ["2026-10-10", "2026-10-17", "2026-10-24"]
    rows = []
    for p in ("CD-A", "CD-B", "CD-C"):
        for w in weeks:
            values = rng.gamma(2.0, 10.0, size=n).round()
            rows.append(
                pd.DataFrame(
                    {
                        "location": p,
                        "target_end_date": w,
                        "sample": np.arange(1, n + 1),
                        "value": values,
                    }
                )
            )
    out = pd.concat(rows, ignore_index=True)
    if national == "own":
        nat = out.loc[out["location"] == "CD-A"].assign(
            location=NATIONAL, value=lambda x: x["value"] * 3
        )
        nat["value"] = rng.permutation(nat["value"].to_numpy())
    else:
        nat = out.groupby(["target_end_date", "sample"], as_index=False)[
            "value"
        ].sum()
        nat.insert(0, "location", NATIONAL)
    return pd.concat([out, nat[list(out.columns)]], ignore_index=True)


def _cells(samples: pd.DataFrame) -> pd.Series:
    return samples.groupby(["location", "target_end_date"])["value"].apply(
        lambda v: tuple(np.sort(v.to_numpy()))
    )


def _wide(samples: pd.DataFrame) -> pd.DataFrame:
    return samples.pivot(
        index="sample", columns=["location", "target_end_date"], values="value"
    )


def test_pairing_a_table_keeps_each_cells_draws() -> None:
    samples = _samples()
    out = pair_samples(
        samples, GaussianCopula(0.8, 0.5, 0.6), np.random.default_rng(11)
    )
    assert set(out.columns) == set(samples.columns)
    pd.testing.assert_series_equal(_cells(out), _cells(samples))
    wide = _wide(out)
    a, b = ("CD-A", "2026-10-10"), ("CD-A", "2026-10-17")
    assert _rank_corr(wide[a], wide[b]) > 0.6
    assert _rank_corr(wide[a], wide[(NATIONAL, "2026-10-10")]) > 0.4


def test_a_summed_country_is_summed_again() -> None:
    samples = _samples(national="sum")
    out = pair_samples(
        samples, GaussianCopula(0.0, 0.7, 0.0), np.random.default_rng(12)
    )
    prov = out.loc[out["location"] != NATIONAL]
    pd.testing.assert_series_equal(
        _cells(prov), _cells(samples.loc[samples["location"] != NATIONAL])
    )
    wide = _wide(out)
    week = "2026-10-17"
    total = sum(wide[(p, week)] for p in ("CD-A", "CD-B", "CD-C"))
    assert np.array_equal(wide[(NATIONAL, week)], total)
    # Paired provinces make the country's sum wider.
    before = _wide(samples)[(NATIONAL, week)].std()
    assert wide[(NATIONAL, week)].std() > 1.2 * before


def test_reported_rows_move_with_their_sample() -> None:
    samples = _samples()
    final = samples.assign(as_of=np.nan)
    reported = samples.assign(as_of="2026-11-01", value=samples["value"] + 0.5)
    both = pd.concat([final, reported], ignore_index=True)
    out = pair_samples(
        both, GaussianCopula(0.8, 0.5, 0.6), np.random.default_rng(13)
    )
    keys = ["location", "target_end_date", "sample"]
    joined = out.loc[out["as_of"].isna()].merge(
        out.loc[out["as_of"].notna()], on=keys
    )
    assert len(joined) == len(samples)
    assert np.allclose(joined["value_y"] - joined["value_x"], 0.5)


def test_pairing_is_reproducible() -> None:
    samples = _samples()
    copula = GaussianCopula(0.8, 0.5, 0.6)
    a = pair_samples(samples, copula, np.random.default_rng(14))
    b = pair_samples(samples, copula, np.random.default_rng(14))
    pd.testing.assert_frame_equal(a, b)


def _origin_scores() -> pd.DataFrame:
    """Scores of forecasts at three origins, with each truth's release."""
    scores = _structured_scores(n_origins=30)
    origin = np.where(
        scores["origin"].astype(int) < 10,
        "2026-10-03",
        np.where(scores["origin"].astype(int) < 20, "2026-10-17", "2026-10-31"),
    )
    return scores.assign(origin=origin, truth_vintage="2026-10-25")


def test_the_first_origin_takes_the_prior() -> None:
    rolling = RollingCopula(GaussianCopula(0.2, 0.2, 0.2))
    fit, n = rolling.fit(_origin_scores(), "2026-10-03")
    assert fit == GaussianCopula(0.2, 0.2, 0.2)
    assert n == 0


def test_rolling_fit_uses_matured_weeks_of_earlier_origins() -> None:
    scores = _origin_scores()
    prior = GaussianCopula(0.2, 0.2, 0.2)
    rolling = RollingCopula(prior, window_weeks=2, weight=10.0)
    fit, n = rolling.fit(scores, "2026-10-31")
    # Earlier origins, truth released by the origin, and the last two
    # matured weeks.
    used = scores.loc[
        (scores["origin"] < "2026-10-31")
        & (scores["target_end_date"] > "2026-10-10")
    ]
    assert n == len(used)
    assert fit == estimate_copula(used, prior, weight=10.0)
    # No truth released by the origin: the prior.
    late = scores.assign(truth_vintage="2026-11-07")
    assert rolling.fit(late, "2026-10-31")[0] == prior


def test_rolling_pairing_keeps_draws_and_records_each_fit() -> None:
    a = _samples(n=200).assign(model="m", origin="2026-10-24")
    b = _samples(n=200).assign(model="m", origin="2026-10-31")
    forecasts = pd.concat([a, b], ignore_index=True)
    truth = (
        forecasts.groupby(["location", "target_end_date"], as_index=False)[
            "value"
        ]
        .median()
        .rename(columns={"value": "truth"})
        .assign(truth_vintage="2026-10-30")
    )
    rolling = RollingCopula(GaussianCopula(0.5, 0.5, 0.5))
    paired, fits = rolling.pair(forecasts, truth, seed=1)
    for origin in ("2026-10-24", "2026-10-31"):
        mine = forecasts["origin"] == origin
        pd.testing.assert_series_equal(
            _cells(paired.loc[paired["origin"] == origin]),
            _cells(forecasts.loc[mine]),
        )
    assert list(fits["origin"]) == ["2026-10-24", "2026-10-31"]
    assert list(fits["n_scores"]) == [0, 12]
    assert {"rho_time", "rho_provinces", "rho_national"} <= set(fits)


def test_a_correlation_with_no_variation_takes_the_prior() -> None:
    scores = _structured_scores(n_origins=5).assign(z=0.5)
    fit = estimate_copula(scores, GaussianCopula(0.3, 0.2, 0.1))
    assert fit == GaussianCopula(0.3, 0.2, 0.1)


def test_rolling_fit_leaves_out_each_truth_not_yet_released() -> None:
    scores = _origin_scores()
    late = scores["location"] == NATIONAL
    scores.loc[late, "truth_vintage"] = "2026-11-07"
    rolling = RollingCopula(GaussianCopula(), weight=0.0)
    fit, n = rolling.fit(scores, "2026-10-31")
    used = scores.loc[(scores["origin"] < "2026-10-31") & ~late]
    assert n == len(used)
    assert fit == estimate_copula(used)


def test_a_models_pairing_does_not_depend_on_other_models() -> None:
    a = _samples(n=200).assign(model="m", origin="2026-10-31")
    b = a.assign(model="another")
    truth = (
        a.groupby(["location", "target_end_date"], as_index=False)["value"]
        .median()
        .rename(columns={"value": "truth"})
        .assign(truth_vintage="2026-10-30")
    )
    earlier = a.assign(origin="2026-10-24")
    alone, _ = RollingCopula().pair(
        pd.concat([earlier, a], ignore_index=True), truth, seed=3
    )
    both, _ = RollingCopula().pair(
        pd.concat([earlier, a, earlier.assign(model="another"), b]),
        truth,
        seed=3,
    )
    keys = ["origin", "location", "target_end_date", "sample"]
    pd.testing.assert_frame_equal(
        alone.sort_values(keys).reset_index(drop=True),
        both.loc[both["model"] == "m"].sort_values(keys).reset_index(drop=True),
    )


def test_a_cells_normal_score_does_not_depend_on_other_cells() -> None:
    one = _forecasts({("CD", "2026-10-17"): np.zeros(50)})
    many = _forecasts(
        {
            ("CD-IT", "2026-10-10"): np.zeros(50),
            ("CD", "2026-10-17"): np.zeros(50),
        }
    )
    many = pd.concat(
        [many, many.assign(value=many["value"] + 1)], ignore_index=True
    )
    one = pd.concat([one, one.assign(value=1.0)], ignore_index=True)
    truth = pd.DataFrame(
        {
            "location": ["CD", "CD-IT"],
            "target_end_date": ["2026-10-17", "2026-10-10"],
            "truth": [0.0, 0.0],
        }
    )
    a = normal_scores(one, truth, seed=4).set_index("location")["z"]
    b = normal_scores(many, truth, seed=4).set_index("location")["z"]
    assert a["CD"] == b["CD"]
    assert normal_scores(one, truth, seed=5)["z"].iloc[0] != a["CD"]


def test_zones_move_with_their_province() -> None:
    samples = _samples(n=300)
    prov = samples.loc[samples["location"] == "CD-A"]
    half = (prov["value"] // 2).to_numpy()
    zones = pd.concat(
        [
            prov.assign(location="CD-A-01", value=half),
            prov.assign(location="CD-A-02", value=prov["value"] - half),
        ],
        ignore_index=True,
    )
    both = pd.concat([samples, zones], ignore_index=True)
    out = pair_samples(
        both, GaussianCopula(0.8, 0.5, 0.6), np.random.default_rng(15)
    )
    keys = ["target_end_date", "sample"]
    is_zone = out["location"].str.count("-") == 2
    summed = out.loc[is_zone].groupby(keys)["value"].sum()
    province = out.loc[out["location"] == "CD-A"].set_index(keys)["value"]
    # Each draw's zones still sum to its province.
    pd.testing.assert_series_equal(
        summed.sort_index(), province.sort_index(), check_names=False
    )
    pd.testing.assert_series_equal(_cells(out.loc[~is_zone]), _cells(samples))
