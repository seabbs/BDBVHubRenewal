"""Static HTML report of fine-tuned checkpoints on the German benchmark.

Reads forecast folders of the German development half, one folder per
label with one file per nowcast date (rows `model`, `origin`,
`location`, `target_end_date`, `horizon` in days, `quantile`, `value`,
`truth`), scores them with the scorer's own `score_quantiles` and writes
`index.html` with the definitions, scores by horizon relative to
zero-shot and to randomwalknowcast, and the dashboard's charts and
colours. BDBV is never read.

    python workflow/scripts/german_report.py IN_DIR OUT_DIR \
        reference=Zero-shot randomwalknowcast="Random walk nowcast" \
        r7-n800="r7 n800" ...

Each LABEL=NAME names a folder of IN_DIR. `reference` is zero-shot and
`randomwalknowcast` the baseline; the others are fine-tuned checkpoints,
shaded in the order given. Scores are cached in OUT_DIR/scores by each
folder's files, so a rerun scores only changed folders.
"""

from __future__ import annotations

import hashlib
import html
import json
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

ROOT = Path(__file__).parents[2]
sys.path.insert(0, str(ROOT / "workflow"))

from flowlib.dashboard import nowcasts, page, svg  # noqa: E402
from flowlib.dashboard import runs as spec  # noqa: E402

from nowcasteval import labels  # noqa: E402
from nowcasteval.scoring import score_quantiles  # noqa: E402

LEVELS = (0.025, 0.1, 0.25, 0.5, 0.75, 0.9, 0.975)
"""The German hub's quantile levels."""
KEYS = ["model", "origin", "location", "target_end_date"]
CELL = ["origin", "location", "target_end_date", "horizon"]
PARTS = ("dispersion", "overprediction", "underprediction")
ZERO_SHOT = "reference"
BASELINE = "randomwalknowcast"
FAMILIES = {
    "fm_chronos_weekly_ragged": "Weekly ragged",
    "fm_chronos_daily_ragged": "Daily ragged",
}
"""Model families, each with its zero-shot and fine-tuned models."""
FORECAST_WEEKS = (7, 14, 21, 28)
"""Horizons of the forecast weeks: the 7-day sums ending 7 to 28 days
after the nowcast date."""

DEFINITIONS = [
    (
        "Weekly ragged, zero-shot",
        "Each place's weekly counts up to 4 weeks before the nowcast "
        "date; the recent, still incomplete weeks enter only through its "
        "lag series (what each release added, lags 0-3). Lag rows of "
        "places with no cases dropped.",
        "Chronos-2's native quantiles per place and week.",
        "The native quantiles interpolated to the hub's levels: one "
        "week is one step, so no draws and no summing.",
    ),
    (
        "Weekly ragged, fine-tuned",
        "As zero-shot, with the adapter and series bias; lag rows of "
        "places with no cases kept, as trained.",
        "As zero-shot.",
        "As zero-shot.",
    ),
    (
        "Daily ragged, zero-shot",
        "Each place's daily counts up to 28 days before the nowcast "
        "date; the recent, still incomplete days enter only through its "
        "lag series (lags 0-27). No weekly row.",
        "Chronos-2's quantiles per place and day.",
        "1,000 draws, an independent uniform per series and day (not "
        "joint paths); the 7 days summed per draw; quantiles of the "
        "1,000 sums. Independent days make the sums too narrow. Being "
        "replaced by native quantiles of a 7-day rolling row.",
    ),
    (
        "Daily ragged, fine-tuned (current)",
        "As zero-shot, plus a weekly-totals row per place (daily+weekly: "
        "Sunday-Saturday sums on Saturdays only); lag rows kept.",
        "Quantiles per place and day for the count rows and the totals rows.",
        "1,000 independent draws; the 7-day sum is the totals row's "
        "draw on the week's Saturday (not summed daily draws); quantiles "
        "of the draws. Being replaced by native quantiles of a 7-day "
        "rolling row, the same block as zero-shot.",
    ),
    (
        "Random walk nowcast",
        "The reporting triangle by release.",
        "A negative binomial chain-ladder nowcast of each week, then a "
        "random walk on log(1 + y) from the latest well-reported week.",
        "The same model and settings as the hub submission. Its "
        "observation model is fitted on past nowcast errors of the same "
        "7-day sums. 1,000 draws per week (the hub takes 100); the "
        "country is the sum of its places by draw; quantiles of the "
        "draws.",
    ),
]
"""What each model is given, outputs, and how that becomes the score."""

TRUTH = (
    "Every row is scored on the hub target: the 7-day sum ending h days "
    "after the nowcast date (h = -28 to 0 the hub's nowcast days; h = 1 "
    "to 28 the forecast days), against that sum as reported 40 days "
    "after it ends. A weekly model's steps are Sunday-Saturday weeks, "
    "so for each target weekday its dates are shifted so that its weeks "
    "end on that day. In the runs shown, every Chronos-2 forecast is "
    "raised to at least the count already reported; this floor is not "
    "part of the model's output and is being dropped. Quantile WIS at "
    "the hub's seven levels, on "
    "the counts and on log(1 + counts), with its dispersion, "
    "overprediction and underprediction; coverage of the central 50% "
    "and 95% intervals. These runs kept only the hub's levels, which "
    "stop at 2.5% and 97.5%; reruns add 5% and 95% for the 90% "
    "interval. Every model is scored on the cells all "
    "of them have."
)


def folder_key(folder: Path) -> str:
    """Return a key of a folder's files: names, sizes and times."""
    h = hashlib.sha256()
    for f in sorted(folder.glob("*.csv.gz")):
        st = f.stat()
        h.update(f"{f.name}{st.st_size}{st.st_mtime_ns}".encode())
    return h.hexdigest()[:16]


def score_folder(folder: Path) -> pd.DataFrame:
    """Return the scores of a folder's forecasts, natural and log."""
    rows = pd.concat(
        (pd.read_csv(f) for f in sorted(folder.glob("*.csv.gz"))),
        ignore_index=True,
    ).dropna(subset=["truth"])
    if "target" in rows:
        rows = rows[rows["target"].fillna("7day") == "7day"]
    rows["model"] = rows["model"].str.removesuffix("_ft")
    out = []
    for scale, f in (("", np.asarray), ("log_", np.log1p)):
        scored = rows.assign(
            value=f(rows["value"].clip(lower=0)),
            truth=f(rows["truth"].clip(lower=0)),
        )
        s = score_quantiles(
            scored[[*KEYS, "horizon", "quantile", "value"]],
            scored[[*KEYS, "truth"]].drop_duplicates(KEYS),
            KEYS,
            LEVELS,
        )
        keep = ["wis", *PARTS] + (
            [c for c in s if c.startswith("coverage_")] if not scale else []
        )
        s = s.set_index([*KEYS, "horizon"])[keep].add_prefix(scale)
        out.append(s)
    return pd.concat(out, axis=1).reset_index()


def scores(
    in_dir: Path, out_dir: Path, names: Mapping[str, str]
) -> pd.DataFrame:
    """Return every label's scores, scoring only changed folders."""
    cache = out_dir / "scores"
    cache.mkdir(parents=True, exist_ok=True)
    frames = []
    for label in names:
        folder = in_dir / label
        if not folder.is_dir() or not any(folder.glob("*.csv.gz")):
            continue
        path = cache / f"{label}-{folder_key(folder)}.csv.gz"
        if path.exists():
            s = pd.read_csv(path)
        else:
            for old in cache.glob(f"{label}-*.csv.gz"):
                old.unlink()
            s = score_folder(folder)
            s.to_csv(path, index=False)
        frames.append(s.assign(label=label))
    return pd.concat(frames, ignore_index=True)


def family_rows(s: pd.DataFrame, family: str) -> pd.DataFrame:
    """Return a family's rows and the baseline's, on common cells."""
    mine = s[(s["model"] == family) | (s["model"] == BASELINE)].copy()
    mine["who"] = np.where(mine["model"] == BASELINE, BASELINE, mine["label"])
    chronos = mine[mine["who"] != BASELINE]
    common = None
    for _, g in chronos.groupby("who"):
        k = set(map(tuple, g[CELL].to_numpy()))
        common = k if common is None else common & k
    cells = pd.Series(list(map(tuple, mine[CELL].to_numpy())))
    return mine[cells.isin(common or set()).to_numpy()]


def group_of(h: int) -> str | None:
    """Return a horizon's summary group."""
    if h <= 0:
        return "nowcast days 0 to -28"
    return "forecast days 1 to 28"


def summary(rows: pd.DataFrame) -> pd.DataFrame:
    """Return mean scores by label and horizon, plus the summary groups."""
    cols = [
        c
        for c in rows
        if c.startswith(("wis", "log_", "coverage_")) or c in PARTS
    ]
    by_h = rows.groupby(["who", "horizon"])[cols].mean()
    by_h["n"] = rows.groupby(["who", "horizon"]).size()
    grouped = rows.assign(horizon=rows["horizon"].map(group_of))
    by_g = grouped.groupby(["who", "horizon"])[cols].mean()
    by_g["n"] = grouped.groupby(["who", "horizon"]).size()
    return pd.concat([by_h, by_g]).reset_index()


def relative(tab: pd.DataFrame, to: str, col: str) -> pd.Series:
    """Return `col` relative to label `to` at the same horizon.

    Relative to the baseline only on cells it has: its own mean is over
    the same cells (`family_rows`), which may be fewer horizons.
    """
    ref = tab[tab["who"] == to].set_index("horizon")[col]
    return tab[col] / tab["horizon"].map(ref)


def colour(label: str, family: str, ft: Sequence[str]) -> tuple[str, str]:
    """Return a label's colour and dash, from the shared maps."""
    if label == BASELINE:
        c, d = labels.MODEL_STYLES[BASELINE]
    elif label == ZERO_SHOT:
        c, d = labels.MODEL_STYLES[family]
    else:
        i = list(ft).index(label)
        c = (
            labels.MODEL_STYLES[f"{family}_ft"][0]
            if i == len(ft) - 1
            else labels.RUN_SHADES[i % len(labels.RUN_SHADES)]
        )
        d = "solid"
    return c, {"solid": "", "dashed": "dash", "dotted": "dot"}.get(d, "")


def fmt(v: float, n: int = 2) -> str:
    """Return a number for a table cell."""
    return "" if v is None or not np.isfinite(v) else f"{v:.{n}f}"


def table(
    tab: pd.DataFrame, order: Sequence[str], names: Mapping[str, str]
) -> str:
    """Return the summary table of one family."""
    heads = [
        "model",
        "horizons",
        "n",
        "WIS",
        "rel. zero-shot",
        "rel. random walk",
        "log WIS",
        "log rel. zero-shot",
        "log rel. random walk",
        "dispersion",
        "over",
        "under",
        "cover 50%",
        "cover 95%",
    ]
    tab = tab.assign(
        rz=relative(tab, ZERO_SHOT, "wis"),
        rb=relative(tab, BASELINE, "wis"),
        lz=relative(tab, ZERO_SHOT, "log_wis"),
        lb=relative(tab, BASELINE, "log_wis"),
    )
    groups = [
        "nowcast days 0 to -28",
        0,
        -7,
        -14,
        -21,
        "forecast days 1 to 28",
        *FORECAST_WEEKS,
    ]
    out = ["<table><tr>" + "".join(f"<th>{h}</th>" for h in heads)]
    for g in groups:
        for who in order:
            r = tab[(tab["who"] == who) & (tab["horizon"] == g)]
            if r.empty:
                continue
            r = r.iloc[0]
            h = g if isinstance(g, str) else f"{g:+d} days"
            cells = [
                html.escape(names.get(who, who)),
                h,
                str(int(r["n"])),
                fmt(r["wis"], 1),
                fmt(r["rz"]),
                fmt(r["rb"]),
                fmt(r["log_wis"], 3),
                fmt(r["lz"]),
                fmt(r["lb"]),
                fmt(r["dispersion"], 1),
                fmt(r["overprediction"], 1),
                fmt(r["underprediction"], 1),
                fmt(r.get("coverage_50", np.nan)),
                fmt(r.get("coverage_95", np.nan)),
            ]
            out.append("<tr>" + "".join(f"<td>{c}</td>" for c in cells))
    return "\n".join([*out, "</table>"])


def chart(
    tab: pd.DataFrame,
    family: str,
    order: Sequence[str],
    names: Mapping[str, str],
    col: str,
    to: str,
    title: str,
) -> str:
    """Return a chart of `col` relative to `to` by horizon, every day."""
    ft = [w for w in order if w not in (ZERO_SHOT, BASELINE)]
    days = tab[tab["horizon"].map(lambda h: not isinstance(h, str))]
    days = days.assign(horizon=days["horizon"].astype(int))
    days = days.assign(rel=relative(days, to, col))
    series = []
    for who in order:
        d = days[days["who"] == who].sort_values("horizon")
        if d.empty or who == to:
            continue
        c, dash = colour(who, family, ft)
        pts = [
            (h + 28, v if np.isfinite(v) else None)
            for h, v in zip(d["horizon"], d["rel"], strict=True)
        ]
        style = (dash,) if dash else ()
        series.append((names.get(who, who), c, pts, "line", *style))
    # The chart's x starts at 0, so day h is drawn at h + 28.
    return svg.chart(
        title,
        series,
        marks=[(28, "nowcast date")],
        hlines=[1.0],
        xlabel="horizon (days after the nowcast date)",
        xfmt=lambda t: f"{t - 28:+g}",
    )


def held_out(path: Path | None) -> str:
    """Return the held-out validation table, if its file has landed."""
    if path is None or not path.exists():
        return (
            "<p class=mut>Held-out validation-round loss and coverage: "
            "not landed yet.</p>"
        )
    t = pd.read_csv(path)
    heads = "".join(f"<th>{html.escape(str(c))}</th>" for c in t.columns)
    body = "".join(
        "<tr>"
        + "".join(
            f"<td>{html.escape(fmt(v) if isinstance(v, float) else str(v))}"
            "</td>"
            for v in row
        )
        for row in t.itertuples(index=False)
    )
    return f"<table><tr>{heads}{body}</table>"


def _references() -> list[spec.Reference]:
    """Return the dashboard's references, as its config names them."""
    import yaml

    config = yaml.safe_load((ROOT / "workflow/config.yaml").read_text())
    out = []
    for e in config["dashboard"]["references"]:
        pm = e["paper_model"]
        colour, dash = labels.MODEL_STYLES[pm]
        out.append(
            spec.make_reference(
                e["key"],
                e["model"],
                label=labels.model_label(pm),
                colour=colour,
                checkpoint=e.get("checkpoint"),
                role=e.get("role"),
                dash=dash,
            )
        )
    return out


def _dataset() -> dict:
    """Return the dashboard's German dataset."""
    import yaml

    config = yaml.safe_load((ROOT / "workflow/config.yaml").read_text())
    return next(
        d for d in config["dashboard"]["datasets"] if d["key"] == "germany"
    )


def fans(
    in_dir: Path, out_dir: Path, names: Mapping[str, str]
) -> tuple[str, str]:
    """Return the dashboard's nowcast panels and the scripts they need.

    The panels read OUT_DIR/data/germany/<location>.json. Folders are
    linked into OUT_DIR/fans-src under the names the dashboard reads:
    zero-shot as `reference`, the baseline as `reference-rw`, and each
    checkpoint as its own run, `<run> n<N>` from its name.
    """
    src = out_dir / "fans-src"
    src.mkdir(parents=True, exist_ok=True)
    for old in src.iterdir():
        old.unlink()
    links = {ZERO_SHOT: ZERO_SHOT, BASELINE: "reference-rw"}
    ft = [w for w in names if w not in links]
    for label in names:
        target = in_dir / label
        if target.is_dir():
            (src / links.get(label, label)).symlink_to(target.resolve())
    run_list, sizes = [], {}
    for label in ft:
        key = spec.split_label(label)[0]
        run, _, size = names[label].partition(" n")
        n = int(size) if size.isdigit() else 0
        c, _ = colour(label, "fm_chronos_daily_ragged", ft)
        run_list.append(
            spec.Run(
                key=label,
                label=run or label,
                colour=c,
                hashes=(key,),
                n={key: n},
            )
        )
        sizes[key] = n
    refs = _references()
    ds = _dataset()
    order = [(g["name"], g["match"]) for g in ds.get("groups") or ()]
    built = nowcasts.build(src, run_list, refs, nowcasts.LEVELS, order=order)
    if not built:
        return "<p class=mut>No forecasts to draw.</p>", ""
    data = out_dir / "data" / ds["key"]
    data.mkdir(parents=True, exist_ok=True)
    for loc in built["locs"]:
        (data / f"{loc}.json").write_text(
            json.dumps(nowcasts.location(built, loc), separators=(",", ":"))
        )
    ctx = SimpleNamespace(
        datasets=[ds],
        runs=run_list,
        references=refs,
        excluded=set(),
        delays={},
        once=False,
        view=None,
    )
    asset = page._asset
    head = f"<script>{asset('player.js')}\n{asset('sel.js')}</script>"
    body = page.nowcasts_section(ctx, sizes, {ds["key"]: built})
    return body + f"<script>{asset('expand.js')}</script>", head


def build(
    in_dir: Path,
    out_dir: Path,
    names: Mapping[str, str],
    heldout: Path | None = None,
) -> Path:
    """Write OUT_DIR/index.html and return its path."""
    s = scores(in_dir, out_dir, names)
    order = [ZERO_SHOT, BASELINE, *[w for w in names if w not in (
        ZERO_SHOT, BASELINE)]]  # fmt: skip
    order = [w for w in order if w in names]
    defs = "".join(
        "<tr>" + "".join(f"<td>{html.escape(c)}</td>" for c in row)
        for row in DEFINITIONS
    )
    parts = [
        "<h2>Definitions</h2>",
        "<table><tr><th>model</th><th>input</th><th>output</th>"
        f"<th>scored quantity</th>{defs}</table>",
        f"<p>{html.escape(TRUTH)}</p>",
    ]
    have = sorted(set(s["label"]))
    missing = [names[w] for w in names if w not in have]
    if missing:
        parts.append(
            "<p class=mut>Not landed yet: "
            + html.escape(", ".join(missing))
            + ".</p>"
        )
    for family, fname in FAMILIES.items():
        rows = family_rows(s, family)
        if rows.empty:
            continue
        tab = summary(rows)
        parts.append(f"<h2>{html.escape(fname)}: 7-day hub target</h2>")
        charts = [
            chart(tab, family, order, names, "wis", ZERO_SHOT,
                  "WIS relative to zero-shot"),
            chart(tab, family, order, names, "log_wis", ZERO_SHOT,
                  "log WIS relative to zero-shot"),
        ]  # fmt: skip
        if BASELINE in set(rows["who"]):
            charts += [
                chart(tab, family, order, names, "wis", BASELINE,
                      "WIS relative to random walk nowcast"),
                chart(tab, family, order, names, "log_wis", BASELINE,
                      "log WIS relative to random walk nowcast"),
            ]  # fmt: skip
        parts.append(f"<div class=plots>{''.join(charts)}</div>")
        parts.append(table(tab, order, names))
    panels, head = fans(in_dir, out_dir, names)
    parts.append(
        "<h2>Nowcasts and forecasts</h2><p class=mut>Each model's 7-day "
        "sums by nowcast date, with the truth reported 40 days on. Pick "
        "the source (weekly or daily models), places and date.</p>"
    )
    parts.append(panels)
    parts.append("<h2>Held-out validation round</h2>")
    parts.append(held_out(heldout))
    style = (ROOT / "workflow/flowlib/dashboard/page/style.css").read_text()
    doc = (
        "<!doctype html><html lang=en><meta charset=utf-8>"
        "<meta name=viewport content='width=device-width,initial-scale=1'>"
        "<title>German r7 results</title>"
        f"<style>{style}{page.ui_colours()}</style>{head}"
        "<h1>Fine-tuned Chronos-2 on the German development half</h1>"
        "<p class=mut>German development half only; BDBV is held out. "
        f"Forecasts from {html.escape(str(in_dir))}.</p>" + "\n".join(parts)
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "index.html"
    tmp = path.with_name("index.html.tmp")
    tmp.write_text(doc)
    tmp.rename(path)
    (out_dir / "labels.json").write_text(json.dumps(dict(names)))
    return path


def main(argv: Sequence[str]) -> None:
    """Build the report from the command line."""
    in_dir, out_dir = Path(argv[0]), Path(argv[1])
    rest = list(argv[2:])
    heldout = None
    if "--heldout" in rest:
        i = rest.index("--heldout")
        heldout = Path(rest[i + 1])
        del rest[i : i + 2]
    names = dict(a.split("=", 1) for a in rest)
    print(build(in_dir, out_dir, names, heldout))


if __name__ == "__main__":
    main(sys.argv[1:])
