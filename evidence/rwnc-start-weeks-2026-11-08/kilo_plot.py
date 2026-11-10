import json, sys, pandas as pd, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
S = "/tmp/claude-1000/-home-seabbs-code-seabbs-BDBVHubRenewal/54f1e2de-e074-4f63-90d5-9393f9cb5b1b/scratchpad/rwnc"
c, name, out = sys.argv[1], sys.argv[2], sys.argv[3]
z = json.load(open(f"{S}/zcounts.json"))
w = pd.to_datetime(z["weeks"]); y = z["places"][c]["reported"]
runs = [("main: full week, 5%, 5 cases", "z_rwnc-base.csv", "#4C72B0"), ("this PR: walk on from the completed cut-off week", "z_b8.csv", "#DD8452"), ("share only: 5%", "z_nowcast-5pct.csv", "#55A868")]
fig, ax = plt.subplots(figsize=(7, 3.4), layout="constrained")
keep = w >= pd.Timestamp("2026-06-01")
ax.bar(w[keep], pd.Series(y)[keep], width=5, color="0.75", label="reported at 11-08")
for i, (lab, f, col) in enumerate(runs):
    d = pd.read_csv(f"{S}/{f}").query("model == 'randomwalknowcast' and kind == 'zone' and location == @c")
    d = d.assign(t=pd.to_datetime(d.target_end_date) + pd.Timedelta(days=(i - 1) * 1.5))
    ax.errorbar(d.t, d.q50, yerr=[d.q50 - d.q05, d.q95 - d.q50], fmt="o", ms=3, color=col, capsize=2, label=f"{lab} (start {d.start.iloc[0][5:]})")
    ax.axvline(pd.Timestamp(d.start.iloc[0]), color=col, ls=":", lw=1)
ax.set_title(f"{name} ({c}), 11-08 origin: median and 90% interval, raw zone samples", fontsize=9)
ax.set_ylabel("Weekly onsets"); ax.legend(fontsize=7, frameon=False, loc="upper left")
fig.savefig(out, dpi=130)
