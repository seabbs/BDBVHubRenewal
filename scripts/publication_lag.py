# Copyright (c) 2026 Sam Abbott. MIT licence, see LICENSE.
r"""Release-pair evidence on the report to publication lag.

A delay-file row present at release $k + 1$ with report day $t$ was
already present at release $k$ with probability

$$(1 - b_k) \frac{\sum_u F(T_k - t_u)}{\sum_u F(T_{k+1} - t_u)},$$

with $F$ the lag cdf, $b_k$ a per-pair background (onsets recorded later
for cases already counted) and $t_u$ the report day for an onset $u$ days
before the end of the onset week. $T_k$ is the release date, as in the
model, or the release's data cut-off.
The script fits lag families by maximum likelihood, profiles the tail
weight, compares the growth each lag predicts in national totals with the
releases, and writes the figures and tables to `OUT`.

Run with `uv run python scripts/publication_lag.py`.
"""

from __future__ import annotations

import itertools
from collections.abc import Callable
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit

OUT = Path("docs/model-development/publication-lag")
TAIL = 14
DMAX = 60
KEY = ["location", "onset_week", "delay"]
DAY = np.timedelta64(1, "D")
COLOURS = {"no lag": "#888888", "uniform 0..3": "#1f77b4"}
COLOURS["uniform 0..3 + tail"] = "#d62728"

Pair = tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]


def uniform(top: int, tail: float = 0.0) -> np.ndarray:
    """Pmf on 0..TAIL: uniform on 0..top with weight `tail` on the rest."""
    pmf = np.zeros(TAIL + 1)
    pmf[: top + 1] = (1 - tail) / (top + 1)
    pmf[top + 1 :] += tail / (TAIL - top)
    return pmf


LAGS = {
    "no lag": uniform(0),
    "uniform 0..3": uniform(3),
    "uniform 0..3 + tail": uniform(3, 0.105),
}


def cdf_at(pmf: np.ndarray, k: np.ndarray) -> np.ndarray:
    """Lag cdf at integer lags `k`, zero below 0."""
    cdf = np.cumsum(pmf)
    return np.where(k < 0, 0.0, cdf[np.clip(k, 0, TAIL)])


def release_pairs(delays: pd.DataFrame, cut: dict) -> list[Pair]:
    """Report ages at both releases and kept and new counts, per pair.

    Rows that could not have been reported by the earlier release, or
    were reported more than 60 days before it, are left out.
    """
    out = []
    for a, b in itertools.pairwise(sorted(cut)):
        before = delays[delays.vintage == a].groupby(KEY).size()
        after = delays[delays.vintage == b].groupby(KEY).size()
        after = after.reset_index(name="n")
        seen = after.set_index(KEY).index.map(before).fillna(0).to_numpy()
        old = np.minimum(seen, after.n.to_numpy())
        new = after.n.to_numpy() - old
        report = (
            after.onset_week.to_numpy()[:, None]
            - np.arange(7)[None, :] * DAY
            + after.delay.to_numpy()[:, None] * DAY
        )
        age1 = ((np.datetime64(cut[a], "D") - report) / DAY).astype(int)
        age2 = ((np.datetime64(cut[b], "D") - report) / DAY).astype(int)
        keep = (age1.max(1) >= 0) & (age1.min(1) <= DMAX)
        out.append((age1[keep], age2[keep], old[keep], new[keep]))
    return out


def kept_share(pmf: np.ndarray, b: float, pair: Pair) -> np.ndarray:
    """Probability that each row was already in the earlier release."""
    age1, age2, _, _ = pair
    later = np.maximum(cdf_at(pmf, age2).sum(1), 1e-12)
    return (1 - b) * cdf_at(pmf, age1).sum(1) / later


def nll(pmf: np.ndarray, bs: np.ndarray, pairs: list[Pair]) -> float:
    """Binomial negative log likelihood over all pairs."""
    out = 0.0
    for pair, b in zip(pairs, bs, strict=True):
        p = np.clip(kept_share(pmf, b, pair), 1e-9, 1 - 1e-9)
        out -= (pair[2] * np.log(p) + pair[3] * np.log1p(-p)).sum()
    return out


def fit(
    make: Callable[[float], np.ndarray], free: bool, pairs: list[Pair]
) -> tuple[float, float, np.ndarray]:
    """Fit the backgrounds and, if `free`, the logit tail weight."""
    n = len(pairs)

    def objective(theta: np.ndarray) -> float:
        tail = expit(theta[0]) if free else 0.0
        return nll(make(tail), expit(theta[-n:]), pairs)

    start = np.r_[[-2.0] if free else [], np.full(n, -3.0)]
    res = minimize(
        objective,
        start,
        method="Nelder-Mead",
        options={"maxiter": 4000, "xatol": 1e-4, "fatol": 1e-4},
    )
    tail = float(expit(res.x[0])) if free else 0.0
    return float(res.fun), tail, expit(res.x[-n:])


def family_table(delays: pd.DataFrame, cuts: dict) -> pd.DataFrame:
    """Fit no lag, uniform 0..L and uniform 0..L plus a tail."""
    rows = []
    for label, cut in cuts.items():
        pairs = release_pairs(delays, cut)
        families = [("none", lambda _: uniform(0), False)]
        families += [
            (f"uniform 0..{top}", lambda _, top=top: uniform(top), False)
            for top in range(1, 6)
        ]
        families += [
            (
                f"uniform 0..{top} + tail",
                lambda w, top=top: uniform(top, w),
                True,
            )
            for top in (1, 2, 3, 4, 5)
        ]
        for name, make, free in families:
            value, tail, bs = fit(make, free, pairs)
            rows.append(
                {
                    "cut-off": label,
                    "lag": name,
                    "nll": round(value, 1),
                    "tail": round(tail, 3) if free else None,
                    "background": " ".join(f"{b:.3f}" for b in bs),
                }
            )
    return pd.DataFrame(rows)


def tail_profile(pairs: list[Pair]) -> pd.DataFrame:
    """Profile negative log likelihood of the tail weight on 0..0.25."""
    rows = []
    for w in np.linspace(0.0, 0.25, 26):
        value, _, _ = fit(lambda _, w=w: uniform(3, w), False, pairs)
        rows.append({"tail": round(w, 2), "nll": value})
    out = pd.DataFrame(rows)
    out["nll"] = (out.nll - out.nll.min()).round(2)
    return out


def tail_by_pair(pairs: list[Pair], names: list[str]) -> pd.DataFrame:
    """Tail weight fitted to each pair alone and with each pair left out."""
    rows = []
    subsets = [([k], names[k]) for k in range(len(pairs))]
    subsets += [
        ([j for j in range(len(pairs)) if j != k], f"without {names[k]}")
        for k in range(len(pairs))
    ]
    for keep, label in subsets:
        _, tail, _ = fit(
            lambda w: uniform(3, w), True, [pairs[k] for k in keep]
        )
        rows.append({"pairs": label, "tail": round(tail, 3)})
    return pd.DataFrame(rows)


def total_growth(
    daily: pd.DataFrame, delays: pd.DataFrame, weeks: tuple[int, int] = (8, 99)
) -> pd.DataFrame:
    r"""National total of each onset week, earlier over later release.

    The prediction averages, over the days $s$ of the week,
    $\sum_d f(d) F(T_k - s - d) / \sum_d f(d) F(T_{k+1} - s - d)$, with
    $f$ the latest delay file's pmf for onset weeks `weeks` old (at least
    8 by default).
    """
    latest = delays.vintage.max()
    age = (latest - delays.onset_week).dt.days // 7
    old = delays[(delays.vintage == latest) & age.between(weeks[0], weeks[1])]
    f = np.bincount(old.delay.clip(0, DMAX), minlength=DMAX + 1)
    f = f / f.sum()
    cd = daily[daily.location == "CD"].copy()
    shift = (5 - cd.onset_date.dt.day_of_week) % 7
    cd["week"] = cd.onset_date + pd.to_timedelta(shift, "D")
    total = (cd.observed + cd.imputed).groupby([cd.vintage, cd.week]).sum()

    def reported(pmf: np.ndarray, release: pd.Timestamp, week) -> float:
        days = pd.date_range(week - pd.Timedelta(days=6), week)
        return np.mean(
            [
                (
                    f * cdf_at(pmf, (release - s).days - np.arange(DMAX + 1))
                ).sum()
                for s in days
            ]
        )

    rows = []
    for a, b in itertools.pairwise(sorted(delays.vintage.unique())):
        end = a - pd.Timedelta(days=(a.day_of_week - 5) % 7)
        for age in range(6):
            week = end - pd.Timedelta(weeks=age)
            row = {
                "v1": a.date(),
                "v2": b.date(),
                "age": age,
                "observed": total[a, week] / total[b, week],
            }
            for name, pmf in LAGS.items():
                row[name] = reported(pmf, a, week) / reported(pmf, b, week)
            rows.append(row)
    return pd.DataFrame(rows).round(3)


def coverage(delays: pd.DataFrame) -> pd.DataFrame:
    r"""Coverage of the stored nowcasts of weeks 0 to 3 back, and with the tail.

    Uses the backtest's nowcasts of what each later release reported
    (`results/nowcast-quantiles.csv`) for Ituri, North Kivu and national.
    The tail column rescales each quantile by
    $\{P_w(V) / P_w(T)\} / \{P_u(V) / P_u(T)\}$, with $P(T)$ the share of
    an onset week's cases published by $T$ under the tail ($w$) or the
    current lag ($u$), from the same fixed delay pmf as `total_growth`.
    That is an approximation to refitting with the tail.
    """
    latest = delays.vintage.max()
    old = delays[
        (delays.vintage == latest)
        & (delays.onset_week < latest - pd.Timedelta(weeks=8))
    ]
    f = np.bincount(old.delay.clip(0, DMAX), minlength=DMAX + 1)
    f = f / f.sum()

    def published(pmf: np.ndarray, release: str, week: str) -> float:
        end = pd.Timestamp(week)
        days = pd.date_range(end - pd.Timedelta(days=6), end)
        lags = np.arange(DMAX + 1)
        return np.mean(
            [
                (f * cdf_at(pmf, (pd.Timestamp(release) - s).days - lags)).sum()
                for s in days
            ]
        )

    q = pd.read_csv("results/nowcast-quantiles.csv")
    q = q[
        (q.target == "reported")
        & q.weeks_back.between(0, 3)
        & q.location.isin(["CD", "CD-IT", "CD-NK"])
        & q.model.isin(["renewal", "renewalbvd", "renewalsp"])
    ].copy()
    tail, now = LAGS["uniform 0..3 + tail"], LAGS["uniform 0..3"]
    q["scale"] = [
        published(tail, v, w)
        / published(tail, o, w)
        / (published(now, v, w) / published(now, o, w))
        for o, v, w in zip(
            q.origin, q.truth_vintage, q.target_end_date, strict=True
        )
    ]
    rows = []
    for (model, back), g in q.groupby(["model", "weeks_back"]):
        row = {"model": model, "weeks back": back, "n": len(g)}
        for label, k in [("current", 1.0), ("tail", g.scale)]:
            lo5, lo25 = g.q5 * k, g.q25 * k
            hi75, hi95, mid = g.q75 * k, g.q95 * k, g.q50 * k
            row[f"{label} 50%"] = ((g.truth >= lo25) & (g.truth <= hi75)).mean()
            row[f"{label} 90%"] = ((g.truth >= lo5) & (g.truth <= hi95)).mean()
            row[f"{label} above 95%"] = (g.truth > hi95).mean()
            row[f"{label} 90% width"] = ((hi95 - lo5) / mid).mean()
        rows.append(row)
    return pd.DataFrame(rows).round(2)


def plot_pairs(pairs: list[Pair], table: pd.DataFrame, names: list[str]):
    """Share of rows new at the later release, by report age."""
    fits = table[table["cut-off"] == "release date"].set_index("lag")
    keys = {
        "no lag": "none",
        "uniform 0..3": "uniform 0..3",
        "uniform 0..3 + tail": "uniform 0..3 + tail",
    }
    bins = np.array([-1, 1, 3, 5, 7, 10, 14, 21, 28, 42, 61])
    mid = (bins[:-1] + bins[1:]) / 2
    fig, axes = plt.subplots(1, len(pairs), figsize=(11, 3.4), sharey=True)
    for k, (pair, ax) in enumerate(zip(pairs, axes, strict=True)):
        n = pair[2] + pair[3]
        idx = np.clip(np.digitize(pair[0].mean(1), bins) - 1, 0, len(mid) - 1)
        size = np.bincount(idx, n, len(mid))
        has = size > 0
        share = np.bincount(idx, pair[3], len(mid)) / np.maximum(size, 1)
        ax.scatter(
            mid[has],
            share[has],
            s=12 + size[has] / 20,
            color="black",
            zorder=3,
            label="observed",
        )
        for name, pmf in LAGS.items():
            b = float(fits.loc[keys[name], "background"].split()[k])
            new = n * (1 - kept_share(pmf, b, pair))
            pred = np.bincount(idx, new, len(mid)) / np.maximum(size, 1)
            ax.plot(mid[has], pred[has], color=COLOURS[name], label=name)
        ax.set_title(names[k], fontsize=10)
        ax.set_xlabel("Report age at the earlier release (days)")
    axes[0].set_ylabel("Share not yet in the earlier release")
    axes[0].legend(fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(OUT / "release-pairs.png", dpi=150, facecolor="white")


def plot_totals(growth: pd.DataFrame) -> None:
    """Observed and predicted ratio of national totals by week age."""
    groups = list(growth.groupby(["v1", "v2"]))
    fig, axes = plt.subplots(1, len(groups), figsize=(11, 3.4), sharey=True)
    for ax, ((v1, v2), g) in zip(axes, groups, strict=True):
        ax.plot(g.age, g.observed, "o", color="black", label="observed")
        for name in LAGS:
            ax.plot(g.age, g[name], color=COLOURS[name], label=name)
        ax.set_title(f"{v1:%m-%d} to {v2:%m-%d}", fontsize=10)
        ax.set_xlabel("Onset week age at the earlier release (weeks)")
    axes[0].set_ylabel("National total, earlier / later")
    axes[0].legend(fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(OUT / "total-growth.png", dpi=150, facecolor="white")


def plot_prototype() -> None:
    """Nowcasts of the two prototype fits, from `prototype-nowcasts.csv`."""
    nc = pd.read_csv(OUT / "prototype-nowcasts.csv")
    places = {"CD-IT": "Ituri", "CD-NK": "North Kivu", "CD": "National"}
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.4))
    for ax, (loc, title) in zip(axes, places.items(), strict=True):
        g = nc[nc.location == loc]
        for j, (lag, name) in enumerate(
            [("uniform", "uniform 0..3"), ("tail", "uniform 0..3 + tail")]
        ):
            h = g[g.lag == lag]
            x = np.arange(len(h)) + (j - 0.5) * 0.2
            ax.errorbar(
                x,
                h["median"],
                yerr=[h["median"] - h.q05, h.q95 - h["median"]],
                fmt="o",
                color=COLOURS[name],
                label=f"{name} (median, 90%)",
            )
        rep = g[g.lag == "uniform"]
        ax.plot(np.arange(len(rep)), rep.reported, "kx", label="reported")
        ax.set_xticks(np.arange(len(rep)), [w[5:] for w in rep.week])
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("Onset week ending")
        ax.set_ylim(bottom=0)
    axes[0].set_ylabel("Cases, release 2026-11-01")
    axes[0].legend(fontsize=8, frameon=False, loc="lower left")
    fig.tight_layout()
    fig.savefig(OUT / "prototype-nowcasts.png", dpi=150, facecolor="white")


def main() -> None:
    """Write the tables and figures."""
    OUT.mkdir(parents=True, exist_ok=True)
    delays = pd.read_csv(
        "data/delays.csv.gz", parse_dates=["onset_week", "vintage"]
    )
    daily = pd.read_csv("data/daily.csv", parse_dates=["onset_date", "vintage"])
    releases = sorted(delays.vintage.unique())
    last_onset = daily.groupby("vintage").onset_date.max()
    first_report = delays.onset_week + pd.to_timedelta(delays.delay - 6, "D")
    last_report = first_report.groupby(delays.vintage).max()
    cuts = {
        "release date": {v: v for v in releases},
        "data cut-off": {
            v: max(last_onset[v], last_report[v]) for v in releases
        },
    }
    table = family_table(delays, cuts)
    table.to_csv(OUT / "lag-fits.csv", index=False)
    print(table.to_string())
    pairs = release_pairs(delays, cuts["release date"])
    names = [f"{a:%m-%d} to {b:%m-%d}" for a, b in itertools.pairwise(releases)]
    profile = tail_profile(pairs)
    profile.to_csv(OUT / "tail-profile.csv", index=False)
    inside = profile.loc[profile.nll <= 1.92, "tail"]
    print(f"tail 95% profile interval {inside.min()} to {inside.max()}")
    growth = total_growth(daily, delays)
    growth.to_csv(OUT / "total-check.csv", index=False)
    print(growth.to_string())
    recent = total_growth(daily, delays, (3, 6))
    recent.to_csv(OUT / "total-check-recent-delay.csv", index=False)
    print(recent.to_string())
    by_pair = tail_by_pair(pairs, names)
    by_pair.to_csv(OUT / "tail-by-pair.csv", index=False)
    print(by_pair.to_string())
    plot_pairs(pairs, table, names)
    plot_totals(growth)
    plot_prototype()
    cover = coverage(delays)
    cover.to_csv(OUT / "coverage.csv", index=False)
    print(cover.to_string())


if __name__ == "__main__":
    main()
