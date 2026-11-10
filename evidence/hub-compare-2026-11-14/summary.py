"""Print the short CD, Ituri and Nord Kivu table as markdown.

Usage: summary.py <evidence dir>

Reads `table.csv`; writes `summary.md` with median [q05, q95] at the
first forecast week (2026-11-14) and the last (2026-12-05).
"""

import sys
from pathlib import Path

import pandas as pd

E = Path(sys.argv[1])
t = pd.read_csv(E / "table.csv")
weeks = ["2026-11-14", "2026-12-05"]
locs = {"CD": "DRC", "CD-IT": "Ituri", "CD-NK": "Nord Kivu"}
t = t[t.location.isin(locs) & t.target_end_date.isin(weeks)]
models = [c for c in t.columns if c not in
          ("location", "location_name", "target_end_date")]
head = ["model"] + [f"{locs[loc]} {w[5:]}" for loc in locs for w in weeks]
rows = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
for m in models:
    cells = [m]
    for loc in locs:
        for w in weeks:
            v = t.loc[(t.location == loc) & (t.target_end_date == w), m]
            cells.append(v.iloc[0] if len(v) and isinstance(v.iloc[0], str)
                         else "")
    rows.append("| " + " | ".join(cells) + " |")
(E / "summary.md").write_text("\n".join(rows) + "\n")
print("\n".join(rows))
