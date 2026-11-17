"""QC plots and tables for the grid r5 pilot: trajectories by new level,
observation families, case sizes and days below one expected onset."""
import glob
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

sims, out = Path(sys.argv[1]), Path(sys.argv[2])
out.mkdir(parents=True, exist_ok=True)
s = pd.concat(pd.read_csv(f) for f in sorted(glob.glob(f"{sims}/settings-*.csv")))
c = pd.concat(pd.read_csv(f) for f in sorted(glob.glob(f"{sims}/counts-*.csv.gz")))
c["sim"] = c["outbreak"].str.rsplit("-", n=1).str[0]
for col in ("reference_date", "report_date"):
    c[col] = pd.to_datetime(c[col])
s = s.set_index("outbreak")

def panel(ax, sim, title):
    x = c[c.sim == sim]
    onset = x.groupby("reference_date")["count"].sum()
    report = x.groupby("report_date")["count"].sum()
    ax.plot(onset.index, onset.values, lw=1, label="by onset (final)")
    ax.plot(report.index, report.values, lw=0.8, alpha=0.7, label="by report date")
    r = s.loc[sim]
    ax.set_title(f"{title}\n{sim}: {r.strata} {r.n_places}x{r.n_ages}, "
                 f"{r.trajectory}, {r.delay_change}, {r.observation}", fontsize=7)
    ax.tick_params(labelsize=6)

groups = {
    "bunching": s.index[s.delay_change == "bunching"],
    "slowdown": s.index[s.delay_change == "slowdown"],
    "multi_wave (uneven)": s.index[s.trajectory == "multi_wave"],
    "52 weeks, places x ages": s.index[s.strata == "places_ages"],
}
rows = [(k, i) for k, v in groups.items() for i in list(v)[:3]]
fig, axes = plt.subplots(len(rows) // 3 + (len(rows) % 3 > 0), 3,
                         figsize=(12, 2.6 * (len(rows) // 3 + 1)))
for ax, (k, sim) in zip(axes.flat, rows):
    panel(ax, sim, k)
for ax in list(axes.flat)[len(rows):]:
    ax.axis("off")
axes.flat[0].legend(fontsize=6)
fig.tight_layout()
fig.savefig(out / "trajectories.png", dpi=110)

fig, (a, b) = plt.subplots(1, 2, figsize=(10, 3))
fam = s.observation.value_counts()
a.bar(fam.index, fam.values)
a.set_title(f"Observation family (n = {len(s)})")
b.hist(s.cases, bins=pd.Series([10, 30, 100, 300, 1e3, 3e3, 1e4, 3e4, 1e5, 3e5]))
b.set_xscale("log")
b.set_title("Cases per simulation")
fig.tight_layout()
fig.savefig(out / "families_sizes.png", dpi=110)

growth = s[s.trajectory == "growth"]
tab = pd.DataFrame({
    "simulations": [len(s)],
    "negbin1d": [(s.observation == "negbin1d").sum()],
    "poisson": [(s.observation == "poisson").sum()],
    "cases_min": [s.cases.min()], "cases_median": [s.cases.median()],
    "cases_max": [s.cases.max()],
    "growth_runs": [len(growth)],
    "growth_low_start_days": [";".join(map(str, growth.low_start_days))],
    "low_start_days_median_all": [s.low_start_days.median()],
})
tab.to_csv(out / "summary.csv", index=False)
s[["strata", "n_places", "n_ages", "weeks", "max_delay", "trajectory",
   "delay_change", "observation", "obs_strata", "shocks", "cases",
   "low_start_days", "seconds"]].to_csv(out / "pilot_settings.csv")
print(tab.T.to_string())
