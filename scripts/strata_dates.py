"""Which German strata and dates the Chronos-2 models do worst on.

Reads the per-cell scores `ft_review.py` writes and the forecasts, and
writes tables, a relative-WIS-by-date figure and panels of the worst
cells to OUT.

Usage: python scripts/strata_dates.py SCORES RESULTS CHECKPOINTS OUT
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from ft_review import german_forecasts
from nowcast_panels import GRID, MODEL_COLOURS, SURFACE, TEXT, draw_panels

from bdbvhubrenewal.data.germany import PLACE_NAMES, hub_target
from bdbvhubrenewal.data.germany import hub_triangle as read_triangle
from bdbvhubrenewal.evaluation import germany as de
from bdbvhubrenewal.evaluation.nowcast import rolling_truth

BASE = "-/randomwalknowcast"
SHOWN = {
    BASE: "randomwalknowcast",
    "-/MeanEnsemble": "Hub MeanEnsemble",
    "daily/zero-shot": "Chronos-2 zero-shot",
    "daily/round 1": "Chronos-2 ft round 1",
    "daily/r3 n1000": "Chronos-2 ft r3 n1000",
    "daily/r3 n2000": "Chronos-2 ft r3 n2000",
}
HOLIDAYS = ("2021-12-24", "2021-12-26", "2021-12-31", "2022-01-01")


def by_location(sc: pd.DataFrame) -> pd.DataFrame:
    """Relative WIS per place, and r3 n1000 against zero-shot."""
    wis = sc.groupby(["pool", "location", "model"])["wis"].mean()
    wis = wis.unstack("model")
    out = wis[[m for m in SHOWN if m in wis]].div(wis[BASE], axis=0)
    out.columns = [SHOWN[m] for m in out.columns]
    out["n1000 vs zero-shot"] = wis["daily/r3 n1000"] / wis["daily/zero-shot"]
    out.index = out.index.set_names(["level", "place"])
    names = [PLACE_NAMES.get(p, p) for p in out.index.get_level_values(1)]
    out.insert(0, "name", names)
    return out.reset_index()


def by_date(sc: pd.DataFrame) -> pd.DataFrame:
    """WIS relative to randomwalknowcast on each nowcast date."""
    wis = sc.groupby(["origin", "model"])["wis"].mean().unstack("model")
    out = wis[[m for m in SHOWN if m in wis]].div(wis[BASE], axis=0)
    return out.rename(columns=SHOWN)


def draw_dates(rel: pd.DataFrame, phase: pd.Series, out: Path) -> None:
    """One line per model over the dates, national phase shaded."""
    fig, ax = plt.subplots(figsize=(11, 4.2), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    x = pd.to_datetime(rel.index)
    shade = {"rising": "#f3e3d6", "falling": "#d9e6f5", "peak/flat": SURFACE}
    for day, p in phase.items():
        d = pd.Timestamp(day)
        ax.axvspan(
            d - pd.Timedelta(hours=12),
            d + pd.Timedelta(hours=12),
            color=shade[p],
            lw=0,
            zorder=0,
        )
    for h in HOLIDAYS:
        ax.axvline(pd.Timestamp(h), color="#8a8984", lw=1, ls=":", zorder=1)
    for model in rel.columns:
        if model == "randomwalknowcast":
            continue
        ax.plot(x, rel[model], color=MODEL_COLOURS[model], lw=2, zorder=3)
        ax.annotate(
            model.replace("Chronos-2 ", ""),
            (x[-1], rel[model].iloc[-1]),
            xytext=(4, 0),
            textcoords="offset points",
            va="center",
            fontsize=8,
            color=TEXT,
        )
    ax.axhline(1, color="#0b0b0b", lw=1, zorder=2)
    ax.set_yscale("log")
    ax.set_ylabel("WIS relative to randomwalknowcast (log)", color=TEXT)
    ax.grid(axis="y", color=GRID, lw=0.8)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.tick_params(colors=TEXT, labelsize=8)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))
    ax.set_xlim(x[0] - pd.Timedelta(days=1), x[-1] + pd.Timedelta(days=12))
    ax.set_title(
        "German development half, all places: WIS relative to "
        "randomwalknowcast by nowcast date",
        fontsize=10,
        loc="left",
    )
    fig.text(
        0.01,
        0.01,
        "Shading: national phase at the date (orange rising, blue falling). "
        "Dotted: 24, 26, 31 Dec and 1 Jan. Black line: parity.",
        fontsize=8,
        color=TEXT,
    )
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(out, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def main() -> None:
    """Write the strata and dates tables and figures."""
    scores, results, ckpts, out = map(Path, sys.argv[1:5])
    sc = pd.read_csv(scores, dtype={"origin": str, "target_end_date": str})
    sc = sc.loc[sc["model"].isin(SHOWN)]
    md = ["# Tables", ""]
    loc = by_location(sc)
    md += ["## WIS relative to randomwalknowcast by place", ""]
    md += [loc.round(2).to_markdown(index=False), ""]
    dates = by_date(sc)
    dates.round(3).to_csv(out / "by-date.csv")
    national = sc.loc[(sc["location"] == "DE") & (sc["model"] == BASE)]
    phase = (
        national.drop_duplicates("origin")
        .set_index("origin")["phase"]
        .fillna("peak/flat")
    )
    draw_dates(dates, phase, out / "by-date.png")
    worst = dates["Chronos-2 ft r3 n1000"].nlargest(8)
    md += ["## Worst dates for r3 n1000 (all places)", ""]
    md += [
        dates.loc[worst.index].assign(phase=phase).round(2).to_markdown(),
        "",
    ]
    # Worst place x date cells for r3 n1000: the largest WIS relative to
    # randomwalknowcast, summed over the five horizons, among the states
    # and age groups with randomwalknowcast's WIS above its lower quartile
    # (so a near-zero denominator does not pick the cell).
    cell = sc.groupby(["origin", "location", "model"])["wis"].sum()
    cell = cell.unstack("model")
    sub = cell.loc[cell.index.get_level_values("location") != "DE"]
    sub = sub.loc[sub[BASE] >= sub[BASE].quantile(0.25)]
    excess = (sub["daily/r3 n1000"] / sub[BASE]).nlargest(3)
    md += ["## Worst place x date cells for r3 n1000", ""]
    rows = []
    for origin, place in excess.index:
        part = sc.loc[(sc["origin"] == origin) & (sc["location"] == place)]
        h = part.pivot_table(index="horizon", columns="model", values="wis")
        h = h[[m for m in SHOWN if m in h]].rename(columns=SHOWN)
        md += [
            f"### {PLACE_NAMES.get(place, place)} ({place}), {origin}, "
            f"phase {part['phase'].iloc[0]}: WIS by days back",
            "",
            h.round(0).to_markdown(),
            "",
        ]
        rows.append((origin, place, f"{place}, {origin}"))
    md += ["## WIS relative to randomwalknowcast by days back", ""]
    hz = sc.groupby(["horizon", "model"])["wis"].mean().unstack("model")
    hz = hz[[m for m in SHOWN if m in hz]].div(hz[BASE], axis=0)
    md += [hz.rename(columns=SHOWN).round(2).to_markdown(), ""]
    (out / "strata-dates-tables.md").write_text("\n".join(md))
    print("\n".join(md))
    # Panels of the worst cells.
    fc = german_forecasts(results, ckpts)
    models = {
        SHOWN[m]: fc.loc[fc["model"] == m] for m in SHOWN if m in set(fc.model)
    }
    rel = {
        SHOWN[m]: cell.loc[list(excess.index), m].sum()
        / cell.loc[list(excess.index), BASE].sum()
        for m in SHOWN
        if m in cell
    }
    config = de.load_germany_config("config/germany.toml")
    triangle = read_triangle(
        Path("data/raw/germany") / Path(config.triangle).name
    )
    origins = sorted({o for o, _ in excess.index})
    asof = de.asof_sums(triangle, origins)
    ends = sorted(set(asof["target_end_date"]))
    truth = rolling_truth(hub_target(triangle, ends), config.truth_delay)
    draw_panels(
        models,
        rel,
        truth,
        asof,
        rows,
        "The three place x date cells where r3 n1000 does worst against "
        "randomwalknowcast",
        out / "worst-cells.png",
        scored_on="on these three cells",
    )


if __name__ == "__main__":
    main()
