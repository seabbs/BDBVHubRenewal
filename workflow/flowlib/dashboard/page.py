"""The dashboard page: training status, losses, bias, scores and nowcasts.

`build` returns the whole page as HTML, in sections behind a sticky bar
of section links, the live run's step and one shared run and reference
selector that every chart follows. Its data come from the runs' folders
(`logs`, `training`, `bias`), the score table (`scores`), each
configured dataset's forecasts (`nowcasts`), optional tables and notes,
and an optional notes file, through `mirror.Source`; its runs,
references, datasets and defaults from configuration. The page's
scripts, style and tooltips are in `page/`.
"""

from __future__ import annotations

import csv
import html
import json
import math
import re
from collections import deque
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from . import bias, learning, logs, nowcasts, scores, svg, training, wis
from .mirror import Source
from .runs import Reference, Run, split_label

ASSETS = Path(__file__).with_name("page")
MODES = ("raw", "start", "base")
STALE = 600
"""Seconds after which the last log line is stale."""
ADAPTER = "finetuned-ckpt/adapter_model.safetensors"
"""A checkpoint's adapter, which the trainer writes before it scores it."""


def _asset(name: str) -> str:
    return (ASSETS / name).read_text()


EXPLAIN: dict[str, str] = json.loads(_asset("explain.json"))
"""Plain-language notes behind each info marker."""


@dataclass
class Context:
    """What one page shows and where its data come from."""

    runs: Sequence[Run]
    references: Sequence[Reference]
    source: Source
    rule: str
    outbreaks: int
    learning_rate: float
    prefixes: Sequence[int]  # `finetune.prefixes`
    datasets: Sequence[Mapping[str, Any]] = ()
    mailbox: Path | None = None
    mail_filter: str = ""
    mail_lines: int = 400
    once: bool = False
    view: Mapping[str, Any] = field(default_factory=dict)
    tables: Sequence[Mapping[str, str]] = ()
    notes: Sequence[Mapping[str, str]] = ()
    logs: dict[str, logs.RunLog] = field(default_factory=dict)
    stores: dict[str, dict] = field(default_factory=dict)
    excluded: set = field(default_factory=set)
    mirror_status: str = ""
    delays: dict[str, int | None] = field(default_factory=dict)
    summarised: dict[str, tuple] = field(default_factory=dict)

    def _runlog(self, run: Run) -> logs.RunLog:
        key = str(run.folder)
        if key not in self.logs:
            self.logs[key] = logs.RunLog(run.folder)
        return self.logs[key]

    def log(self, run: Run) -> list[dict[str, Any]]:
        """Return a run's logged steps (none for a run with no folder)."""
        if run.folder is None:
            return []
        return self._runlog(run).rows()

    def sims(self, run: Run) -> Callable[[float], float]:
        """Return the simulations a run has trained by a step.

        From its own checkpoints and logged chunks (`training.anchors`).
        """
        cks = _checkpoints(run)
        chunks = self._runlog(run).chunks() if run.folder else {}
        pairs = training.anchors(cks, chunks)
        spo = training.steps_per_simulation(0, [], cks)
        return lambda step: training.sims_at(pairs, step, spo)

    def dataset(self, key: str) -> Mapping[str, Any]:
        """Return a configured dataset by key (a quantiles one if unknown)."""
        return next(
            (d for d in self.datasets if d["key"] == key),
            {"key": key, "label": key, "format": "quantiles", "groups": []},
        )

    def forecasts(self, key: str) -> Path:
        """Return the quantile files a dataset's panels read.

        A samples dataset's are its summaries (`summarise`), never made
        while a page is built.
        """
        if self.dataset(key).get("format") != "samples":
            return self.source.dataset(key)
        return self.source.root / f"{key}-quantiles"

    def summarise(self) -> None:
        """Summarise each samples dataset whose truth or files changed.

        Run at start, before a static render and in the background loop.
        """
        for ds in self.datasets:
            if ds.get("format") != "samples":
                continue
            key = ds["key"]
            folder = self.source.dataset(key)
            truth = self.source.truth(key)
            if truth is None or not truth.is_file() or not folder.is_dir():
                continue
            sig = (
                truth.stat().st_mtime_ns,
                tuple(
                    (str(f), f.stat().st_mtime_ns)
                    for f in sorted(folder.glob("*/*.csv.gz"))
                ),
            )
            if self.summarised.get(key) == sig:
                continue
            self.delays[key] = nowcasts.summarise_samples(
                folder, self.forecasts(key), truth, ds
            )
            self.summarised[key] = sig

    def nowcasts(self, key: str) -> dict[str, Any] | None:
        """Return a dataset's nowcasts, rebuilt only when its files change."""
        ds = self.dataset(key)
        return nowcasts.cached_build(
            self.forecasts(key),
            self.runs,
            self.references,
            self.stores.setdefault(key, {}),
            levels=nowcasts.LEVELS,
            excluded=self.excluded,
            order=[(g["name"], g["match"]) for g in ds.get("groups") or ()],
        )


def _json(payload: Any) -> str:
    return json.dumps(payload, separators=(",", ":")).replace("</", "<\\/")


def tip(key: str) -> str:
    """Return the info marker of a note in `EXPLAIN`."""
    text = html.escape(EXPLAIN.get(key, ""), quote=True)
    return f'<span class=info tabindex=0 data-tip="{text}">ⓘ</span>'


def _fmt(v: Any, n: int = 4) -> str:
    if v is None:
        return ""
    return f"{v:.{n}f}" if isinstance(v, float) else html.escape(str(v))


def _dur(s: float | None) -> str:
    if s is None or s != s or s < 0:
        return "n/a"
    s = int(s)
    return f"{s // 3600}h {s % 3600 // 60:02d}m"


def _size(name: str) -> int:
    digits = re.sub(r"\D", "", name)
    return int(digits) if digits else 0


def _checkpoints(run: Run) -> list[tuple[str, dict, float | None]]:
    return [] if run.folder is None else logs.checkpoints(run.folder)


@dataclass
class Status:
    """The live run's state, for the tiles and the sticky bar."""

    step: int | None
    rate: float | None
    last_t: float | None
    params: dict[str, Any]
    spo: float
    saved: set[int]
    pending: list[int]
    nxt: int | None

    @property
    def target(self) -> int:
        """Return the final checkpoint's size."""
        return max(self.params["prefixes"] or [0])

    def at(self, n: int) -> int:
        """Return the step at which checkpoint `n` is saved."""
        return round(n * self.spo)

    def eta(self, n: int | None) -> str:
        """Return the time until checkpoint `n` at the recent speed."""
        if not (self.rate and self.step is not None and n):
            return "n/a"
        if self.at(n) <= self.step:
            return "due now"
        return _dur((self.at(n) - self.step) / self.rate)


def run_sizes(prefixes: Sequence[int], outbreaks: int) -> list[int]:
    """Return a run's checkpoint sizes as the trainer makes them.

    The prefixes below the run's simulations, then the run's simulations.
    """
    return sorted({int(p) for p in prefixes if p < outbreaks} | {outbreaks})


def live_status(
    live: Run,
    rows: list[dict[str, Any]],
    planned: Sequence[int],
    sims: Callable[[float], float] | None = None,
) -> Status:
    """Return the live run's step, speed, schedule and checkpoints.

    `planned` are its checkpoints unless the trainer's command line names
    them (`logs.prefixes`). `sims` (`Context.sims`) gives the
    simulations trained by its step, for a run that decays over
    simulations, not steps.
    """
    step = rows[-1]["step"] if rows else None
    last_t = rows[-1]["t"] if rows else None
    recent = [r for r in rows if r["t"] >= (last_t or 0) - 1800]
    rate = None
    if len(recent) >= 2 and recent[-1]["t"] > recent[0]["t"]:
        rate = (recent[-1]["step"] - recent[0]["step"]) / (
            recent[-1]["t"] - recent[0]["t"]
        )
    folder = live.folder
    args = logs.trainer_args(folder) if folder else ""
    params = (
        training.run_params(folder, training.history(live), step, args, planned)
        if folder
        else {
            "lr_steps": 0,
            "lr_outbreaks": 0,
            "groups": {},
            "current": {},
            "logged": {},
            "prefixes": list(planned),
        }
    )
    sizes = params["prefixes"]
    spo = training.steps_per_simulation(
        params["lr_steps"], sizes, _checkpoints(live)
    )
    if sims and step and not params["lr_steps"] and sims(step) > 0:
        spo = step / sims(step)
    saved = {
        n
        for n in sizes
        if folder and (folder / f"train/n{n}/settings.json").exists()
    }
    # Written but not yet scored at the checkpoint: its adapter is saved.
    # Snakemake makes a segment's empty output folders when it starts.
    pending = [
        n
        for n in sizes
        if n not in saved
        and folder
        and (folder / f"train/n{n}/{ADAPTER}").exists()
    ]
    nxt = next((n for n in sizes if n not in saved and n not in pending), None)
    return Status(step, rate, last_t, params, spo, saved, pending, nxt)


def status_tiles(st: Status, live: Run) -> tuple[str, str]:
    """Return the status tiles and the sticky bar's step and next checkpoint."""
    now = datetime.now().timestamp()
    folder = live.folder
    trained = st.step / st.spo if st.step is not None else None
    trainer, alive = logs.trainer_state(folder) if folder else ("n/a", None)
    age = now - st.last_t if st.last_t else None
    stale = age is not None and age > STALE
    # The next prefix with the simulations done of it, and when it is due;
    # a written checkpoint still being scored is named as itself.
    done = f"n{trained:.0f} / " if trained is not None else ""
    next_v = f"{done}n{st.nxt}" if st.nxt else "none"
    eta = f"ETA {st.eta(st.nxt)}" if st.nxt else ""
    next_s = f"step {st.at(st.nxt)}, {eta}" if st.nxt else ""
    if st.pending:
        since = datetime.fromtimestamp(
            (folder / f"train/n{st.pending[0]}/{ADAPTER}").stat().st_mtime
        ).strftime("%H:%M")
        next_s = "; ".join(
            x for x in (next_s, f"n{st.pending[0]} scoring since {since}") if x
        )
    p = st.params
    rates = "\n".join(f"{g} {v:.2e}" for g, v in p["current"].items()) or "n/a"
    if p["lr_steps"]:
        decay = f"linear decay to 0 at step {p['lr_steps']}"
    elif p["lr_outbreaks"]:
        decay = f"linear decay to 0 at {p['lr_outbreaks']} simulations"
    else:
        decay = "constant"
    decay += ", from history" if p["logged"] else ""
    tiles = [
        (
            "Step",
            "t_step",
            "n/a" if st.step is None else str(st.step),
            f"n{trained:.0f} of {st.target} simulations trained"
            if trained is not None
            else "",
            "",
        ),
        (
            "Steps / s (30 min)",
            "t_rate",
            f"{st.rate:.3f}" if st.rate else "n/a",
            f"{st.rate * 60:.1f} steps/min" if st.rate else "",
            "",
        ),
        ("Next checkpoint", "t_next", next_v, next_s, ""),
        (
            f"ETA n{st.target}",
            "t_eta",
            st.eta(st.target),
            f"step {st.at(st.target)}",
            "",
        ),
        ("Learning rates", "t_lr", rates, decay, ""),
        (
            "Trainer",
            "t_trainer",
            trainer,
            "",
            "" if alive is None else " ok" if alive else " bad",
        ),
        (
            "Last log line",
            "t_log",
            datetime.fromtimestamp(st.last_t).strftime("%H:%M:%S")
            if st.last_t
            else "n/a",
            ("STALE " if stale else "") + (_dur(age) + " ago" if age else ""),
            " bad" if stale else "",
        ),
    ]
    out = "".join(
        f'<div class="tile{cls}"><div class=k>{k}{tip(key)}</div>'
        f'<div class="v{" vl" if chr(10) in v else ""}">'
        f"{html.escape(v).replace(chr(10), '<br>')}</div>"
        f"<div class=s>{html.escape(s)}</div></div>"
        for k, key, v, s, cls in tiles
    )
    bar = (
        f"<span class=bt><span class=k>step</span> <b>{st.step}</b> "
        f"<span class=k>next</span> <b>{html.escape(next_v)}</b>"
        f"{' ' + html.escape(eta) if eta else ''}</span>"
        if st.step is not None
        else ""
    )
    return out, bar


def lr_card(params: Mapping[str, Any], step: int | None, run: str) -> str:
    """Return a small learning-rate-against-step chart of every group."""
    if not params["groups"]:
        return ""
    width, height = 190, 150
    left, right, top, bottom = 34, 6, 6, 18
    xmax = params["lr_steps"] or max([step or 0, 1])
    vals = [max(b, 1e-12) for b in params["groups"].values()]
    # A linear decay reaches zero, so the log axis stops at 1% of the base.
    lo = math.log10(min(vals) * (0.01 if params["lr_steps"] else 1)) - 0.1
    hi = math.log10(max(vals)) + 0.15

    def sx(x: float) -> float:
        return left + x / xmax * (width - left - right)

    def sy(v: float) -> float:
        frac = (math.log10(max(v, 1e-12)) - lo) / (hi - lo)
        return top + (1 - frac) * (height - top - bottom)

    o = [
        f'<svg viewBox="0 0 {width} {height}" width="{width}" '
        f'height="{height}">'
    ]
    for e in range(math.ceil(lo), math.floor(hi) + 1):
        o.append(
            f'<line x1="{left}" x2="{width - right}" y1="{sy(10**e):.1f}" '
            f'y2="{sy(10**e):.1f}" class=grid /><text x="{left - 3}" '
            f'y="{sy(10**e) + 3:.1f}" text-anchor=end class=ax>1e{e}</text>'
        )
    o.append(
        f'<text x="{left}" y="{height - 4}" class=ax>0</text><text '
        f'x="{width - right}" y="{height - 4}" text-anchor=end '
        f"class=ax>{xmax}</text>"
    )
    legend = []
    for i, (g, base) in enumerate(params["groups"].items()):
        col = svg.PALETTE[i % len(svg.PALETTE)]
        pts = params["logged"].get(g) or (
            [(0, base), (0.99 * xmax, base * 0.01)]
            if params["lr_steps"]
            else [(0, base), (xmax, base)]
        )
        o.append(
            '<polyline points="'
            + " ".join(f"{sx(x):.1f},{sy(max(y, 1e-12)):.1f}" for x, y in pts)
            + f'" fill=none stroke="{col}" stroke-width=1.6 />'
        )
        cur = params["current"].get(g)
        if cur is not None and step is not None:
            o.append(
                f'<circle cx="{sx(step):.1f}" cy="{sy(cur):.1f}" r=3 '
                f'fill="{col}"><title>{html.escape(g)} {cur:.3e} at step '
                f"{step}</title></circle>"
            )
        legend.append(
            f'<span><i style="background:{col}"></i>{html.escape(g)}</span>'
        )
    o.append("</svg>")
    return (
        f'<div class=card id=lrcard data-run="{html.escape(run, quote=True)}" '
        f'style="padding:4px;line-height:1.2"><div class=mut '
        f'style="font-size:11px">learning rate vs step{tip("lr")}</div>'
        f'{"".join(o)}<div class=leg style="font-size:11px">'
        f"{''.join(legend)}</div></div>"
    )


def loss_section(ctx: Context, live: Run, st: Status) -> str:
    """Return the loss controls and the client-side loss plots' data."""
    cks = {r.key: _checkpoints(r) for r in ctx.runs}
    own = {
        r.key: training.loss_run(
            r, ctx.log(r), cks[r.key], ctx.sims(r), None, r.colour
        )
        for r in ctx.runs
    }
    out = []
    for r in ctx.runs:
        base = own.get(r.base or "") if r.base else None
        refs = (
            {k: base[k] for k in ("full0", "src0", "parts0")} if base else None
        )
        out.append(
            training.loss_run(
                r, ctx.log(r), cks[r.key], ctx.sims(r), refs, r.colour
            )
        )
    # x is simulations trained: checkpoints sit at their own sizes.
    marks = [
        (int(s.get("n_outbreaks") or _size(nm)), nm)
        for nm, s, _ in cks[live.key]
    ] + [(n, f"n{n}?") for n in st.pending]
    now = ctx.sims(live)(st.step) if st.step else 0
    next_ck = next(
        (n for n in st.params["prefixes"] if n not in st.saved), None
    )
    xlim = max([1.2 * now, 50] + ([next_ck] if next_ck else []))
    data = _json({"runs": out, "marks": marks, "xlim": xlim})
    modebar = (
        '<div class="sctl modebar"><span class=mut>Loss scale</span>'
        "<button class=tog data-mode=raw>raw</button>"
        "<button class=tog data-mode=start title=\"divided by the run's own "
        'step-0 loss">relative to start</button>'
        '<button class=tog data-mode=base title="divided by base Chronos-2 on '
        "the same blocks (a run's base)\">relative to base Chronos-2</button>"
        "<span class=sep></span><span class=mut>x range</span>"
        "<button class=tog data-rng=zoom>live run</button>"
        "<button class=tog data-rng=full>full range</button>"
        + tip("lossscale")
        + "</div>"
    )
    return (
        f"{modebar}<div id=lossrefs class=refnote></div>"
        "<div class=plots3 id=lossplots></div>"
        f'<script type="application/json" id=ldata>{data}</script>'
        f"<script>{_asset('loss.js')}</script>"
    )


def bias_section(ctx: Context, live: Run, st: Status) -> str:
    """Return the attention bias row and the live run's learning rates."""
    data = _json(
        bias.payload(
            ctx.runs,
            st.step,
            {r.key: ctx.log(r) for r in ctx.runs},
            {r.key: ctx.sims(r) for r in ctx.runs},
        )
    )
    return (
        "<div id=biasrefs class=refnote></div><div class=biasrow>"
        f"<div id=biasdyn class=biasdyn></div>"
        f"{lr_card(st.params, st.step, live.key)}</div>"
        f'<script type="application/json" id=bdata>{data}</script>'
        f"<script>{_asset('bias.js')}</script>"
    )


def checkpoint_table(
    run: Run, info: Mapping[str, Any], mode: str, sources: Sequence[str] = ()
) -> str:
    """Return one run's checkpoint table at one loss scale.

    `sources` are the held-out sources given a column each.
    """
    rows = info["ck"]
    if not rows:
        label = html.escape(run.label)
        return f"<p class=mut>no checkpoints yet for {label}</p>"
    head = {
        "raw": "held-out",
        "start": "held-out / start",
        "base": "held-out / base",
    }[mode]
    den = {"raw": 1, "start": info["full0"], "base": info["bfull"]}[mode]
    by_head = "".join(f"<th>{html.escape(src)}" for src in sources)
    out = [
        f"<table><tr><th>run<th>ckpt<th>hash<th>{head}<th>held-out "
        f"start<th>real-val{by_head}<th>saved</tr>"
    ]
    for name, s, mtime in rows:
        by = s.get("held_out_loss_by_source") or {}
        ho = s.get("held_out_loss")
        cells = [
            html.escape(run.label),
            name,
            html.escape(str(s.get("hash", ""))),
            _fmt(ho / den if ho and den else None) or "n/a",
            _fmt(s.get("held_out_loss_start")),
            _fmt(s.get("real_validation_loss")),
            *(_fmt(by.get(src)) or "n/a" for src in sources),
            datetime.fromtimestamp(mtime).strftime("%m-%d %H:%M")
            if mtime
            else "",
        ]
        out.append("<tr>" + "".join(f"<td>{c}" for c in cells) + "</tr>")
    return "".join(out) + "</table>"


def notes_section(ctx: Context) -> str:
    """Return the configured notes (markdown files) as collapsed blocks."""
    out = []
    for note in ctx.notes:
        path = Path(note["path"]).expanduser()
        if path.is_file():
            out.append(
                f"<details><summary>{html.escape(note['label'])}</summary>"
                f"<pre>{html.escape(path.read_text())}</pre></details>"
            )
    return "".join(out)


def checkpoint_sizes(
    ctx: Context, info: Mapping, score_rows: Sequence[Mapping[str, str]]
) -> dict[str, int]:
    """Return each checkpoint hash's size: saved settings, then scores."""
    sizes: dict[str, int] = {}
    for row in score_rows:
        key = split_label(row.get("checkpoint", ""))[0]
        try:
            n = int(float(row.get("outbreaks") or 0))
        except ValueError:
            n = 0
        if n and key:
            sizes.setdefault(key, n)
    for r in ctx.runs:
        for _, s, _ in info[r.key]["ck"]:
            if s.get("n_outbreaks"):
                sizes[str(s.get("hash"))] = int(s["n_outbreaks"])
    return sizes


def _caption(ctx: Context, ds: Mapping[str, Any]) -> str:
    text = ds.get("truth", "")
    delay = ctx.delays.get(ds["key"])
    return text.replace("{delay}", "" if delay is None else str(delay))


def nowcasts_section(
    ctx: Context, sizes: Mapping[str, int], builts: Mapping[str, Any]
) -> str:
    """Return the nowcast panels of every dataset, behind a selector."""
    datasets, sources = [], set()
    for ds in ctx.datasets:
        built = builts.get(ds["key"])
        if not built or not built["origins"] or not built["locs"]:
            continue
        meta = nowcasts.meta(
            built, ctx.runs, ctx.references, sizes, ctx.excluded
        )
        first = built["series"][built["locs"][0]]
        for m in meta:
            m["count"] = max(
                (
                    len(t)
                    for src in first.get(m["id"], {}).values()
                    for t in src.values()
                ),
                default=0,
            )
        for loc in built["series"].values():
            for by_target in loc.values():
                for by_source in by_target.values():
                    sources |= set(by_source)
        datasets.append(
            {
                "key": ds["key"],
                "label": ds.get("label", ds["key"]),
                "metric": ds.get("metric", ""),
                "truth": _caption(ctx, ds),
                "meta": meta,
                "locs": built["locs"],
                "origins": built["origins"],
                "targets": built["targets"],
                "preload": {
                    loc: nowcasts.location(built, loc) for loc in built["locs"]
                }
                if ctx.once
                else {},
                "groups": [
                    {
                        "name": g["name"],
                        "locs": [
                            x for x in built["locs"] if re.match(g["match"], x)
                        ],
                    }
                    for g in ds.get("groups") or ()
                ],
            }
        )
    if not datasets:
        return "<p class=mut>No nowcasts yet.</p>"
    defaults = dict((ctx.view or {}).get("defaults") or {})
    data = _json(
        {
            "datasets": datasets,
            "runs": [
                {"key": r.key, "label": r.label, "color": r.colour}
                for r in ctx.runs
            ],
            "defaults": defaults,
            # WIS parts in the paper's order, each its share of the colour.
            "parts": [[p, wis.part_share(p)] for p in wis.PARTS],
        }
    )
    first = defaults.get("source")
    options = "".join(
        f'<option value="{html.escape(s, quote=True)}"'
        f"{' selected' if s == first else ''}>{html.escape(s)}</option>"
        for s in sorted(sources)
    )
    controls = _asset("nowcasts.html").replace("{sources}", options)
    controls = controls.replace("{tip}", tip("nowcast"))
    return (
        controls + f'<script type="application/json" id=gdata>{data}</script>'
        f"<script>{_asset('nowcasts.js')}</script>"
    )


def wis_section(ctx: Context, rows: Sequence[Mapping[str, str]]) -> str:
    """Return the mean WIS bars by component; none before any are scored.

    Runs' bars are of the view's default source (`dashboard.defaults`).
    """
    source = ((ctx.view or {}).get("defaults") or {}).get("source")
    bars = scores.wis_bars(
        rows, ctx.runs, ctx.references, ctx.rule, source, ctx.excluded
    )
    return wis.section(bars, ctx.view or {}, tip)


def mail_section(ctx: Context) -> str:
    """Return the notes file's last matching lines; none without one."""
    if not ctx.mailbox:
        return ""
    if not ctx.mailbox.exists():
        return "<p class=mut>no notes yet</p>"
    rx = re.compile(ctx.mail_filter, re.I) if ctx.mail_filter else None
    with ctx.mailbox.open(errors="replace") as f:
        tail = deque(f, maxlen=ctx.mail_lines)
    lines = [x.rstrip("\n") for x in tail if not rx or rx.search(x)]
    if not lines:
        return "<p class=mut>no matching lines</p>"

    def cut(s: str) -> str:
        return html.escape(s if len(s) <= 260 else s[:257] + "...")

    out = "<pre>" + "\n".join(cut(x) for x in lines[-5:]) + "</pre>"
    if len(lines) > 5:
        out += (
            f"<details><summary>show more ({len(lines) - 5})</summary><pre>"
            + html.escape("\n".join(lines[:-5]))
            + "</pre></details>"
        )
    return out


def tables_section(ctx: Context, info: Mapping[str, Any]) -> str:
    """Return the checkpoint tables and any configured tables (CSV files)."""
    sources = sorted(
        {
            src
            for r in ctx.runs
            for _, s, _ in info[r.key]["ck"]
            for src in (s.get("held_out_loss_by_source") or {})
        }
    )
    out = ["<h3>Checkpoints</h3><div class=wrap>"]
    out += [
        f'<div class="mode m-{m}">'
        + "".join(
            checkpoint_table(r, info[r.key], m, sources) for r in ctx.runs
        )
        + "</div>"
        for m in MODES
    ]
    out.append("</div>")
    for table in ctx.tables:
        path = Path(table["path"]).expanduser()
        out.append(f"<h3>{html.escape(table['label'])}</h3><div class=wrap>")
        if not path.is_file():
            out.append(f"<p class=mut>{html.escape(str(path))} not present</p>")
        else:
            with path.open() as f:
                rows = list(csv.reader(f))
            out.append(
                "<table>"
                + "".join(
                    "<tr>"
                    + "".join(
                        f"<{'th' if i == 0 else 'td'}>{html.escape(c)}"
                        for c in r
                    )
                    + "</tr>"
                    for i, r in enumerate(rows)
                )
                + "</table>"
            )
        out.append("</div>")
    return "".join(out)


def selector_data(ctx: Context, builts: Mapping[str, Any]) -> str:
    """Return the shared run and reference selector's data."""
    refs = [
        {"key": r.key, "label": r.label, "color": r.colour}
        for r in ctx.references
    ]
    for ds in ctx.datasets:
        built = builts.get(ds["key"])
        for s in built["sets"] if built else ():
            if s.startswith("model:") and not any(r["key"] == s for r in refs):
                refs.append(
                    {
                        "key": s,
                        "label": s.removeprefix("model:"),
                        "color": "#94a3b8",
                    }
                )
    shown = [r.key for r in ctx.runs] + [
        r.key for r in ctx.references if r.role in ("start", "baseline")
    ]
    return _json(
        {
            "runs": [
                {"key": r.key, "label": r.label, "color": r.colour}
                for r in ctx.runs
            ],
            "refs": refs,
            "defaults": shown,
        }
    )


def _parts(ctx: Context) -> tuple[Run, str, str, list]:
    """Return the live run, top bar status, selector data and sections.

    Each section is (id, title, content, open).

    Raises
    ------
    ValueError
        If no run is configured.
    """
    if not ctx.runs:
        raise ValueError("no runs to show: configure at least one run")
    live = next((r for r in ctx.runs if r.live), ctx.runs[0])
    rows = ctx.log(live)
    st = live_status(
        live, rows, run_sizes(ctx.prefixes, ctx.outbreaks), ctx.sims(live)
    )
    tiles, bar = status_tiles(st, live)
    status = (
        logs.CACHE.get(live.folder / "train/stream-status.json") or {}
        if live.folder
        else {}
    )
    sub = (
        f"stage: {html.escape(str(status.get('stage')))}; stream status "
        f"{html.escape(str(status.get('time', '')))}"
        if status
        else ""
    )
    info = {
        r.key: training.loss_run(
            r, ctx.log(r), _checkpoints(r), ctx.sims(r), None, r.colour
        )
        | {"ck": _checkpoints(r)}
        for r in ctx.runs
    }
    for r in ctx.runs:
        base = info.get(r.base or r.key, info[r.key])
        info[r.key]["bfull"] = base["full0"]
    score_rows = scores.read(ctx.source.scores)
    sizes = checkpoint_sizes(ctx, info, score_rows)
    mail = mail_section(ctx)
    builts = {ds["key"]: ctx.nowcasts(ds["key"]) for ds in ctx.datasets}
    live_note = "" if ctx.once else "updates in place every 60 s; "
    mirror = (
        f"scores {html.escape(ctx.mirror_status)}; "
        if ctx.mirror_status
        else ""
    )
    parts = [
        (
            "status",
            "Status",
            f"<div class=tiles>{tiles}</div>"
            f"<div class=mut>{live_note}{mirror}{sub}</div>",
            True,
        ),
        ("loss", "Loss", loss_section(ctx, live, st), True),
        ("bias", "Attention bias", bias_section(ctx, live, st), True),
        (
            "learning",
            "What fine-tuning learns",
            learning.section(ctx.runs, tip),
            True,
        ),
        ("nowcasts", "Nowcasts", nowcasts_section(ctx, sizes, builts), True),
    ]
    bars = wis_section(ctx, score_rows)
    if bars:
        parts.append(("wis", "Mean WIS", bars, True))
    if mail:
        parts.append(("mailbox", "Recent activity", mail, True))
    tables = tables_section(ctx, info) + notes_section(ctx)
    parts.append(("tables", "Tables", tables, False))
    bar = f"{bar} <span class=mut>{datetime.now():%H:%M:%S}</span>"
    return live, bar, selector_data(ctx, builts), parts


NAV = {
    "status": "Status",
    "mailbox": "Activity",
    "learning": "Learning",
    "wis": "WIS",
}
"""Short names in the top bar, where they differ from the section titles."""


def sections(ctx: Context) -> dict[str, str]:
    """Return the parts the page updates in place, by id.

    `bar` is the top bar's status; the rest are each section's content
    (`<details id=...>` without its summary).
    """
    _, bar, _, parts = _parts(ctx)
    return {"bar": bar} | {key: content for key, _, content, _ in parts}


def build(ctx: Context) -> str:
    """Return the dashboard page.

    Raises
    ------
    ValueError
        If no run is configured.
    """
    live, bar, seldata, parts = _parts(ctx)
    nav = "".join(
        f'<a href="#{key}">{NAV.get(key, title)}</a>'
        for key, title, _, _ in parts
    )
    rest = "\n".join(
        f"<details class=sec id={key}{' open' if opened else ''}><summary>"
        f"<h2>{title}</h2></summary>{content}</details>"
        for key, title, content, opened in parts
    )
    body = f"""<header class=topbar>
<div class=navrow><b>{html.escape(live.label)} run</b><nav class=nav>{nav}</nav>
<span id=livebar>{bar}</span></div>
<div id=selbar class=selbar></div>
<script type="application/json" id=seldata>{seldata}</script>
<script>ftSel.init(JSON.parse(document.getElementById("seldata").textContent))</script>
</header>
{rest}"""
    explain = _json(EXPLAIN)
    info_js = (
        f"window.ftExplain={explain};window.ftInfo=function(k){{var "
        "t=window.ftExplain[k];return t?'<span class=info tabindex=0 "
        'data-tip="\'+t.replace(/"/g,"&quot;")+\'">\\u24d8</span>\':""};'
    )
    return (
        "<!doctype html><html><head><meta charset=utf-8>"
        '<meta name=viewport content="width=device-width,initial-scale=1">'
        "<title>Fine-tuning dashboard</title>"
        f"<style>{_asset('style.css')}</style><script>{info_js}</script>"
        f"<script>{_asset('player.js')}\n{_asset('sel.js')}</script></head>"
        f"<body>{body}<script>{_asset('mode.js')}\n{_asset('nav.js')}\n"
        f"{_asset('expand.js')}\n{'' if ctx.once else _asset('live.js')}"
        "</script></body></html>"
    )
