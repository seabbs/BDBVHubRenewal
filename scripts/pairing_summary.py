"""Summarise the pairing evidence: a table and two figures.

Usage:
  pairing_summary.py GERMANY_DIR BDBV_DIR OUT [ZONES_DIR]

GERMANY_DIR holds pairing_germany.py's <model>-scores.csv.gz and
BDBV_DIR pairing_bdbv.py's <model>-scores.csv.gz and
<model>-energy.csv.gz, and ZONES_DIR pairing_zones.py's
<model>-zone-scores.csv.gz. Writes OUT/pairing-table.md,
OUT/pairing-germany.png and OUT/pairing-bdbv.png.
"""

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

germany, bdbv, out = (Path(p) for p in sys.argv[1:4])
out.mkdir(parents=True, exist_ok=True)
ARMS = ["off", "fixed", "rolling"]
LABELS = {
    "randomwalknowcast": "randomwalknowcast",
    "fm_chronos_daily_ragged_ft": "Chronos-2 daily ft",
}
SHORT = {
    "randomwalknowcast": "rwnc",
    "fm_chronos_daily_ragged_ft": "Chronos ft",
}
COLOURS = {"off": "#9aa5b1", "fixed": "#5b8bd0", "rolling": "#1f4e9c"}

de = pd.concat(
    [pd.read_csv(p) for p in sorted(germany.glob("*-scores.csv.gz"))],
    ignore_index=True,
)
de = de.loc[de["pool"].isin(["states", "national"])]
de_tab = (
    de.groupby(["model", "pool", "arm"])
    .agg(
        n=("wis", "size"),
        wis=("wis", "mean"),
        cover50=("draw_coverage_50", "mean"),
        cover90=("draw_coverage_90", "mean"),
    )
    .reset_index()
)
eb = pd.concat(
    [pd.read_csv(p) for p in sorted(bdbv.glob("*-scores.csv.gz"))],
    ignore_index=True,
)
eb_tab = (
    eb.groupby(["model", "target", "arm"])
    .agg(
        n=("crps", "size"),
        crps=("crps", "mean"),
        cover50=("cover50", "mean"),
        cover90=("cover90", "mean"),
    )
    .reset_index()
)
en = pd.concat(
    [pd.read_csv(p) for p in sorted(bdbv.glob("*-energy.csv.gz"))],
    ignore_index=True,
)
en_tab = (
    en.groupby(["model", "scale", "arm"])[["energy", "log_energy"]]
    .mean()
    .reset_index()
)


def rel(tab, value, by):
    """Each arm's value over the off arm's, within `by`."""
    base = tab.loc[tab["arm"] == "off"].set_index(by)[value]
    return tab.apply(lambda r: r[value] / base[tuple(r[b] for b in by)], 1)


de_tab["rel"] = rel(de_tab, "wis", ["model", "pool"])
eb_tab["rel"] = rel(eb_tab, "crps", ["model", "target"])
en_tab["rel"] = rel(en_tab, "energy", ["model", "scale"])


def order(tab):
    tab = tab.assign(arm=pd.Categorical(tab["arm"], ARMS, ordered=True))
    return tab.sort_values([c for c in tab.columns[:2]] + ["arm"])


lines = ["## Germany, development half (80 dates, states task)", ""]
lines += [
    "| model | target | arm | n | WIS | rel. WIS | 50% cov. | 90% cov. |",
    "|---|---|---|---|---|---|---|---|",
]
for r in order(de_tab).itertuples():
    lines.append(
        f"| {LABELS.get(r.model, r.model)} | {r.pool} | {r.arm} | {r.n} | "
        f"{r.wis:.1f} | {r.rel:.3f} | {r.cover50:.2f} | {r.cover90:.2f} |"
    )
lines += ["", "## BDBV release backtest", ""]
lines += [
    "| model | target | arm | n | CRPS | rel. CRPS | 50% cov. | 90% cov. |",
    "|---|---|---|---|---|---|---|---|",
]
for r in order(eb_tab).itertuples():
    lines.append(
        f"| {LABELS.get(r.model, r.model)} | {r.target} | {r.arm} | {r.n} | "
        f"{r.crps:.1f} | {r.rel:.3f} | {r.cover50:.2f} | {r.cover90:.2f} |"
    )
lines += ["", "| model | scale | arm | energy | rel. energy | log energy |"]
lines += ["|---|---|---|---|---|---|"]
for r in order(en_tab).itertuples():
    lines.append(
        f"| {LABELS.get(r.model, r.model)} | {r.scale} | {r.arm} | "
        f"{r.energy:.1f} | {r.rel:.3f} | {r.log_energy:.3f} |"
    )
if len(sys.argv) > 4:
    zs = pd.concat(
        [
            pd.read_csv(p)
            for p in sorted(Path(sys.argv[4]).glob("*-zone-scores.csv.gz"))
        ],
        ignore_index=True,
    )
    lines += ["", "## BDBV health zones (zones forecast directly)", ""]
    lines += ["| model | target | arm | n | score | 50% cov. | 90% cov. |"]
    lines += ["|---|---|---|---|---|---|---|"]
    for (m, t, a), g in zs.groupby(["model", "target", "arm"]):
        score = (
            f"energy {g['energy'].mean():.2f}, log {g['log_energy'].mean():.3f}"
            if t == "energy"
            else f"CRPS {g['crps'].mean():.2f}"
        )
        c50 = "" if t == "energy" else f"{g['cover50'].mean():.2f}"
        c90 = "" if t == "energy" else f"{g['cover90'].mean():.2f}"
        lines.append(
            f"| {LABELS.get(m, m)} | {t} | {a} | {len(g)} | {score} | {c50} | {c90} |"
        )
(out / "pairing-table.md").write_text("\n".join(lines) + "\n")
print("\n".join(lines))


def panel(ax, tab, group, value, title, ref=None):
    groups = list(dict.fromkeys(tab[group]))
    models = list(LABELS)
    width = 0.8 / len(ARMS)
    ticks, names = [], []
    x = 0.0
    for m in models:
        for g in groups:
            rows = tab.loc[(tab["model"] == m) & (tab[group] == g)]
            if rows.empty:
                continue
            for i, arm in enumerate(ARMS):
                v = rows.loc[rows["arm"] == arm, value]
                if len(v):
                    ax.bar(
                        x + i * width,
                        float(v.iloc[0]),
                        width,
                        color=COLOURS[arm],
                        label=(
                            ("prior" if arm == "fixed" else arm)
                            if not ticks
                            else None
                        ),
                    )
            ticks.append(x + width)
            names.append(f"{SHORT[m]}\n{g}".replace("_", " "))
            x += 1.0
    if ref is not None:
        ax.axhline(ref, color="black", lw=0.8, ls="--")
    ax.set_xticks(ticks, names, fontsize=7)
    ax.set_title(title, fontsize=9)


fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
panel(axes[0], de_tab, "pool", "rel", "WIS relative to no pairing", 1.0)
panel(axes[1], de_tab, "pool", "cover90", "90% interval coverage", 0.9)
axes[0].legend(fontsize=7, frameon=False)
fig.suptitle("Germany, development half: states and national", fontsize=10)
fig.tight_layout()
fig.savefig(out / "pairing-germany.png", dpi=150)

keep = ["national", "provinces_sum", "province_total", "national_total"]
sub = eb_tab.loc[eb_tab["target"].isin(keep)]
fig, axes = plt.subplots(1, 2, figsize=(15, 4))
panel(axes[0], sub, "target", "rel", "CRPS relative to no pairing", 1.0)
panel(axes[1], sub, "target", "cover90", "90% interval coverage", 0.9)
axes[0].legend(fontsize=7, frameon=False)
fig.suptitle("BDBV release backtest: summed targets", fontsize=10)
fig.tight_layout()
fig.savefig(out / "pairing-bdbv.png", dpi=150)
