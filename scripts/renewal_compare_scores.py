r"""Score stored renewal fits before and after a change, origin by origin.

Each argument is `LABEL=PATH[,PATH...]`, the stored fits
(`MODEL/ORIGIN/KEY.json.gz`) of one arm. The province and national
forecasts at `[forecast]` horizons (`config/backtest.toml`) are scored
against the newest real vintage in `data/target.csv`, as the release
backtest does (`score_forecasts`). Zones and as-reported rows are left
out. Prints mean CRPS, log CRPS, WIS and 50%/90% coverage by arm and
origin, then each arm's ratio to the first arm on the forecasts both
arms have.

Usage::

    uv run python scripts/renewal_compare_scores.py \\
      before=store/renewalbvd/2026-11-08/a.json.gz,... \\
      after=output/probe/bvd-centred-1108/store/...
"""

from __future__ import annotations

import gzip
import json
import sys

import pandas as pd

from bdbvhubrenewal.data import load_target, vintages
from bdbvhubrenewal.data.zones import is_zone
from bdbvhubrenewal.evaluation.backtest import load_config
from bdbvhubrenewal.evaluation.scoring import score_forecasts

SCORES = ["crps", "log_crps", "wis", "coverage_50", "coverage_90"]
KEYS = ["model", "kind", "vintage", "location", "target_end_date"]


def _forecasts(path: str) -> pd.DataFrame:
    with gzip.open(path) as fh:
        stored = json.load(fh)
    f = stored["forecasts"]
    df = pd.DataFrame(f["data"], columns=f["columns"])
    df = df.loc[df["as_of"].isna() & ~df["location"].map(is_zone)]
    return df.drop(columns=["as_of", "reference_date"], errors="ignore")


def main(arms: list[str]) -> None:
    """Print the score comparison."""
    config = load_config()
    real = load_target().query("kind == 'real'")
    latest = real.loc[real["vintage"] == vintages(real)[-1]]
    truth = latest.rename(columns={"total": "truth"})[
        ["location", "target_end_date", "truth"]
    ]
    print(f"Truth: real vintage {vintages(real)[-1]}")
    scored = []
    for arm in arms:
        label, paths = arm.split("=", 1)
        for path in paths.split(","):
            fc = _forecasts(path)
            fc = fc.loc[
                fc["horizon"].between(config.min_horizon, config.max_horizon)
            ]
            s = score_forecasts(fc, truth, keys=KEYS)
            scored.append(s.assign(arm=label))
    s = pd.concat(scored, ignore_index=True)
    s["model"] = s["model"].astype(str)
    by = (
        s.groupby(["arm", "vintage"])[SCORES]
        .mean()
        .join(s.groupby(["arm", "vintage"]).size().rename("n"))
    )
    print(by.round(3).to_string())
    first = arms[0].split("=", 1)[0]
    ids = ["vintage", "location", "target_end_date", "horizon"]
    base = s.loc[s["arm"] == first].set_index(ids)[SCORES]
    for label in s["arm"].unique():
        if label == first:
            continue
        other = s.loc[s["arm"] == label].set_index(ids)[SCORES]
        both = base.index.intersection(other.index)
        print(f"\n{label} / {first} on {len(both)} shared forecasts")
        rel = pd.DataFrame(
            {
                "origin": [str(v) for v in sorted(both.unique(0))] + ["all"],
            }
        )
        rows = []
        for v in [*sorted(both.unique(0)), None]:
            idx = both if v is None else both[both.get_level_values(0) == v]
            a, b = other.loc[idx], base.loc[idx]
            rows.append(
                {
                    "crps": a["crps"].mean() / b["crps"].mean(),
                    "log_crps": a["log_crps"].mean() / b["log_crps"].mean(),
                    "wis": a["wis"].mean() / b["wis"].mean(),
                    "cov50": f"{a['coverage_50'].mean():.2f} vs "
                    f"{b['coverage_50'].mean():.2f}",
                    "cov90": f"{a['coverage_90'].mean():.2f} vs "
                    f"{b['coverage_90'].mean():.2f}",
                }
            )
        print(pd.concat([rel, pd.DataFrame(rows)], axis=1).round(3).to_string())


if __name__ == "__main__":
    main(sys.argv[1:])
