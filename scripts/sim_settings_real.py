"""Summaries of real outbreaks to compare with simulated ones.

The same summaries as `scripts/sim_settings_check.R`, from the German
hospitalisation triangle (Wolffram et al. 2023) and the BDBV hub's data,
per series: Germany, its states and its age groups, and the DRC and its
provinces.

Run from the repo root after `task data` and `bdbvhubrenewal germany-fetch`:

    uv run python scripts/sim_settings_real.py <out.csv>
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

GERMANY = Path("data/raw/germany/COVID-19_hospitalizations.csv")
TRUTH_DAYS = 80
"""Days after a Meldedatum when its count is taken as final."""


def growth(series: pd.Series, days: int, floor: float) -> float:
    """Return the fastest daily growth of a series over `days` steps.

    The log ratio of the series to its value `days` steps before, over
    the days, where both are at least `floor`.
    """
    ratio = series / series.shift(1)
    ok = (series >= floor) & (series.shift(1) >= floor)
    r = np.log(ratio[ok]) / days
    return float(r.max()) if len(r) else np.nan


def germany() -> pd.DataFrame:
    """Return each German series' summaries."""
    tri = pd.read_csv(GERMANY)
    tri["reference_date"] = pd.to_datetime(tri["date"])
    rows = []
    keys = ["location", "age_group"]
    for (loc, age), g in tri.groupby(keys):
        if loc != "DE" and age != "00+":
            continue
        g = g.sort_values("reference_date").set_index("reference_date")
        lags = [c for c in g.columns if re.fullmatch(r"value_\d+d", c)]
        lags = sorted(lags, key=lambda c: int(c[len("value_") : -1]))
        cells = g[lags].fillna(0).to_numpy()
        cum = np.cumsum(cells, axis=1)
        final = pd.Series(cum[:, min(TRUTH_DAYS, cum.shape[1] - 1)], g.index)
        week = final.rolling(7).sum()
        mean7 = final.rolling(7, center=True).mean()
        weekday = (final / mean7).groupby(final.index.day_of_week).mean()
        big = mean7 >= 500
        effect = weekday.reindex(final.index.day_of_week).to_numpy()
        resid = np.log(final / mean7) - np.log(effect)
        total = cum[:, min(TRUTH_DAYS, cum.shape[1] - 1)]
        weights = np.where(total > 0, total, np.nan)
        share = {
            d: float(np.nansum(cum[:, d]) / np.nansum(weights))
            for d in (7, 14, 28)
        }
        surv = 1 - np.nansum(cum[:, :TRUTH_DAYS], axis=0) / np.nansum(weights)
        cdf = 1 - surv
        rows.append(
            {
                "source": "Germany",
                "series": f"{loc} {age}",
                "max_growth": growth(week, 7, 50),
                "peak_daily": float(mean7.max()),
                "total": float(final.sum()),
                "delay_mean": float(np.sum(surv)),
                "delay_q95": int(np.argmax(cdf >= 0.95)),
                "reported_7": share[7],
                "reported_14": share[14],
                "reported_28": share[28],
                "cv_large": float(resid[big].std()) if big.any() else np.nan,
                "onset_dow_ratio": float(weekday.max() / weekday.min()),
            }
        )
    return pd.DataFrame(rows)


def bdbv(data: Path = Path("data")) -> pd.DataFrame:
    """Return each BDBV series' summaries, from the latest release."""
    target = pd.read_csv(data / "target.csv")
    target = target.loc[target["vintage"] == target["vintage"].max()]
    delays = pd.read_csv(data / "delays.csv.gz")
    delays = delays.loc[delays["vintage"] == delays["vintage"].max()]
    rows = []
    for loc, g in target.groupby("location"):
        week = g.sort_values("target_end_date")["total"].astype(float)
        ends = g.sort_values("target_end_date")["target_end_date"]
        week.index = pd.to_datetime(ends)
        mine = delays if loc == "CD" else delays.loc[delays["location"] == loc]
        d = mine["delay"].to_numpy()
        rows.append(
            {
                "source": "BDBV",
                "series": loc,
                "max_growth": growth(week, 7, 10),
                "peak_daily": float(week.max() / 7),
                "total": float(week.sum()),
                "delay_mean": float(d.mean()) if len(d) else np.nan,
                "delay_q95": float(np.quantile(d, 0.95)) if len(d) else np.nan,
                "reported_7": float((d <= 7).mean()) if len(d) else np.nan,
                "reported_14": float((d <= 14).mean()) if len(d) else np.nan,
                "reported_28": float((d <= 28).mean()) if len(d) else np.nan,
                "cv_large": np.nan,
                "onset_dow_ratio": np.nan,
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    """Write every real series' summaries and print them by source."""
    out = pd.concat([germany(), bdbv()], ignore_index=True)
    out.to_csv(sys.argv[1], index=False)
    print(out.groupby("source").describe().T.to_string())


if __name__ == "__main__":
    main()
