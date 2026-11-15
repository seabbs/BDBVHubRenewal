"""Score fine-tuned Chronos-2 checkpoints against randomwalknowcast.

Writes `evidence/checkpoint-scores.csv`: one row per study, level, model
and checkpoint, with relative WIS and log WIS against randomwalknowcast
and interval coverage, on the cells every model of that study and level
made, with how many origins those cells come from and the first and
last of them. Zero-shot rows have checkpoint `zero-shot` and 0 outbreaks.

Germany: the development half, levels national, states and age, scored
on the 40-day truth (`germany_chronos_dev.scores`). BDBV: the release
backtest's complete target, levels national and provinces.

Usage: python scripts/checkpoint_curve.py CHECKPOINTS GERMANY BDBV OUT

CHECKPOINTS is a CSV of `checkpoint,outbreaks`. GERMANY holds
`zero-shot.csv.gz` (randomwalknowcast and the zero-shot ragged models'
forecasts with truth) and `<checkpoint>/<date>.csv.gz`. BDBV holds
`zero-shot/nowcast-scores.csv` and `<checkpoint>/nowcast-scores.csv`.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from germany_chronos_dev import scores as germany_scores

from bdbvhubrenewal.evaluation import germany as de

BASELINE = "randomwalknowcast"
MODELS = ("fm_chronos_daily_ragged", "fm_chronos_weekly_ragged")
GERMANY_CELL = ["origin", "location", "target_end_date"]
BDBV_CELL = ["origin", "truth_vintage", "location", "target_end_date"]
COLUMNS = [
    "study",
    "level",
    "model",
    "checkpoint",
    "outbreaks",
    "n",
    "n_origins",
    "first_origin",
    "last_origin",
    "rel_wis",
    "rel_log_wis",
    "coverage_50",
    "coverage_90",
    "coverage_95",
]


def _tag(rows: pd.DataFrame, checkpoint: str) -> pd.DataFrame:
    """Name each fine-tuned model's rows by its checkpoint."""
    ft = rows["model"].str.endswith("_ft")
    return rows.assign(
        checkpoint=ft.map({True: checkpoint, False: "zero-shot"}),
        model=rows["model"].str.removesuffix("_ft"),
    )


def _matched(rows: pd.DataFrame, cell: list[str]) -> pd.DataFrame:
    """Keep, in each level, the cells every model and checkpoint made."""
    key = rows["model"] + "@" + rows["checkpoint"]
    out = []
    for _, part in rows.assign(_k=key).groupby("level"):
        n = part.groupby(cell)["_k"].nunique()
        full = n.index[n == part["_k"].nunique()]
        out.append(part.set_index(cell).loc[full].reset_index())
    return pd.concat(out, ignore_index=True).drop(columns="_k")


def _relative(rows: pd.DataFrame, study: str) -> pd.DataFrame:
    covers = [
        c for c in ("coverage_50", "coverage_90", "coverage_95") if c in rows
    ]
    keys = ["level", "model", "checkpoint"]
    mean = rows.groupby(keys)[["wis", "log_wis", *covers]].mean()
    mean["n"] = rows.groupby(keys).size()
    origins = rows.groupby(keys)["origin"]
    mean["n_origins"] = origins.nunique()
    mean["first_origin"] = origins.min()
    mean["last_origin"] = origins.max()
    base = mean.xs((BASELINE, "zero-shot"), level=("model", "checkpoint"))
    for m in ("wis", "log_wis"):
        mean[f"rel_{m}"] = (
            mean[m]
            / base[m].reindex(mean.index.get_level_values("level")).to_numpy()
        )
    out = mean.reset_index().assign(study=study)
    return out.reindex(columns=COLUMNS)


def germany(root: Path, checkpoints: pd.DataFrame) -> pd.DataFrame:
    """Return the German development half's rows."""
    config = de.load_germany_config("config/germany.toml")
    read = {"dtype": {"origin": str, "target_end_date": str}}
    zero = pd.read_csv(root / "zero-shot.csv.gz", **read)
    zero = zero.loc[zero["model"].isin([BASELINE, *MODELS])]
    parts = [zero.assign(checkpoint="zero-shot")]
    for ckpt in checkpoints["checkpoint"]:
        files = sorted((root / ckpt).glob("*.csv.gz"))
        # A checkpoint enters once every development date is nowcast.
        if len(files) < len(de.nowcast_dates(config, held_out=False)):
            continue
        fc = pd.concat([pd.read_csv(f, **read) for f in files])
        parts.append(_tag(fc, ckpt))
    fc = pd.concat(parts, ignore_index=True)
    fc = fc.loc[fc["origin"] < config.split_date]
    rows = []
    for ckpt, part in fc.groupby("checkpoint"):
        sc = germany_scores(part.drop(columns="checkpoint"), config.truth_delay)
        rows.append(sc.assign(checkpoint=ckpt))
    sc = pd.concat(rows, ignore_index=True).rename(columns={"pool": "level"})
    return _relative(_matched(sc, GERMANY_CELL), "germany")


def bdbv(root: Path, checkpoints: pd.DataFrame) -> pd.DataFrame:
    """Return the BDBV release backtest's rows."""
    zero = pd.read_csv(root / "zero-shot" / "nowcast-scores.csv")
    zero = zero.loc[zero["model"].isin([BASELINE, *MODELS])]
    parts = [zero.assign(checkpoint="zero-shot")]
    for ckpt in checkpoints["checkpoint"]:
        path = root / ckpt / "nowcast-scores.csv"
        if not path.exists():
            continue
        sc = pd.read_csv(path)
        parts.append(_tag(sc.loc[sc["model"].str.endswith("_ft")], ckpt))
    sc = pd.concat(parts, ignore_index=True)
    sc = sc.loc[sc["target"] == "complete"]
    sc["level"] = (sc["location"] == "CD").map(
        {True: "national", False: "provinces"}
    )
    return _relative(_matched(sc, BDBV_CELL), "ebola")


def main() -> None:
    """Write the checkpoint rows."""
    checkpoints = pd.read_csv(sys.argv[1], dtype=str)
    out = pd.concat(
        [
            germany(Path(sys.argv[2]), checkpoints),
            bdbv(Path(sys.argv[3]), checkpoints),
        ],
        ignore_index=True,
    )
    outbreaks = dict(
        zip(checkpoints["checkpoint"], checkpoints["outbreaks"], strict=True)
    )
    out["outbreaks"] = out["checkpoint"].map(outbreaks).fillna("0").astype(int)
    out = out.sort_values(["study", "level", "model", "outbreaks"])
    out.round(4).to_csv(sys.argv[4], index=False)
    print(out.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
