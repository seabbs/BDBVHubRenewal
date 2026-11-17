"""The dashboard page: a training run's state, checkpoints and scores.

`build` returns the whole page as HTML. Its data come from the runs'
folders (`logs`), the score table and German forecast caches (`scores`,
`germany`, through `mirror.Source`) and the mailbox; its runs and
references from configuration (`runs`). The page's scripts and style
are in `page/`.
"""

from __future__ import annotations

import html
import json
import re
from collections import deque
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from . import germany, logs, scores, svg
from .mirror import Source
from .runs import Classifier, Reference, Run, split_label

ASSETS = Path(__file__).with_name("page")
MODES = ("raw", "start", "base")
SOURCES = ("weekly", "daily", "3day")
DASH_BY_RUN = (None, "dash", "dd", "dot")


@dataclass
class Context:
    """What one page shows and where its data come from."""

    runs: Sequence[Run]
    references: Sequence[Reference]
    source: Source
    rule: str
    outbreaks: int
    learning_rate: float
    lr_steps: int
    style: Mapping[str, Any]
    datasets: Sequence[Mapping[str, Any]] = ()
    mailbox: Path | None = None
    mail_filter: str = ""
    once: bool = False
    logs: dict[str, logs.Log] = field(default_factory=dict)
    germany_store: dict = field(default_factory=dict)
    excluded: set = field(default_factory=set)
    mirror_status: str = ""

    def germany(self) -> dict[str, Any] | None:
        """Return the German nowcasts, rebuilt only when the cache changes."""
        return germany.cached_build(
            self.source.dataset("germany"),
            self.runs,
            self.references,
            self.germany_store,
            levels=self.style["levels"],
        )

    @property
    def steps_per_outbreak(self) -> float:
        """Return the trainer's steps per simulated outbreak."""
        return self.lr_steps / self.outbreaks

    def log(self, run: Run) -> list[dict[str, Any]]:
        """Return a run's logged steps (none for a run with no folder)."""
        if run.folder is None:
            return []
        key = str(run.folder)
        if key not in self.logs:
            self.logs[key] = logs.Log(run.folder / "train.log")
        return self.logs[key].rows()


def _json(payload: Any) -> str:
    return json.dumps(payload, separators=(",", ":")).replace("</", "<\\/")


def _asset(name: str) -> str:
    return (ASSETS / name).read_text()


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


def _styled(series: tuple, i: int) -> tuple:
    """Return a series with the dash of the `i`th run (none for the first)."""
    dash = DASH_BY_RUN[i % len(DASH_BY_RUN)]
    return (*series, dash) if dash else series


def _checkpoints(run: Run) -> list[tuple[str, dict, float | None]]:
    return [] if run.folder is None else logs.checkpoints(run.folder)


def status_tiles(ctx: Context, live: Run, rows: list[dict]) -> str:
    """Return the live run's status tiles."""
    now = datetime.now().timestamp()
    folder = live.folder
    step = rows[-1]["step"] if rows else None
    last_t = rows[-1]["t"] if rows else None
    recent = [r for r in rows if r["t"] >= (last_t or 0) - 1800]
    rate = None
    if len(recent) >= 2 and recent[-1]["t"] > recent[0]["t"]:
        rate = (recent[-1]["step"] - recent[0]["step"]) / (
            recent[-1]["t"] - recent[0]["t"]
        )
    args = logs.trainer_args(folder) if folder else ""
    sizes = logs.prefixes(folder, args) if folder else list(logs.PREFIXES)
    saved = {
        n
        for n in sizes
        if folder and (folder / f"train/n{n}/settings.json").exists()
    }
    pending = [
        n
        for n in sizes
        if n not in saved
        and folder
        and (folder / f"train/n{n}/finetuned-ckpt").exists()
    ]
    nxt = next((n for n in sizes if n not in saved and n not in pending), None)
    spo = ctx.steps_per_outbreak

    def at(n: int) -> int:
        return round(n * spo)

    def eta(n: int | None) -> str:
        if not (rate and step is not None and n):
            return "n/a"
        return "due now" if at(n) <= step else _dur((at(n) - step) / rate)

    trained = step / spo if step is not None else None
    lr = (
        ctx.learning_rate * max(0.0, 1 - step / ctx.lr_steps)
        if step is not None
        else None
    )
    trainer, alive = logs.trainer_state(folder) if folder else ("n/a", None)
    age = now - last_t if last_t else None
    stale = age is not None and age > 600
    if pending:
        since = datetime.fromtimestamp(
            (folder / f"train/n{pending[0]}").stat().st_mtime
        ).strftime("%H:%M")
        nxt_v = f"n{pending[0]}: scoring since {since}"
        nxt_s = f"then n{nxt} at step {at(nxt)}, ETA {eta(nxt)}" if nxt else ""
    else:
        nxt_v = f"n{nxt}" if nxt else "none"
        nxt_s = f"at step {at(nxt)}, ETA {eta(nxt)}" if nxt else ""
    tiles = [
        (
            "Step",
            "n/a" if step is None else str(step),
            f"n{trained:.0f} of {ctx.outbreaks} simulations trained"
            if trained is not None
            else "",
            "",
        ),
        (
            "Steps / s (30 min)",
            f"{rate:.3f}" if rate else "n/a",
            f"{rate * 60:.1f} steps/min" if rate else "",
            "",
        ),
        ("Next checkpoint", nxt_v, nxt_s, ""),
        (
            f"ETA n{ctx.outbreaks}",
            eta(ctx.outbreaks),
            f"step {at(ctx.outbreaks)}",
            "",
        ),
        (
            "Learning rate",
            f"{lr:.3e}" if lr is not None else "n/a",
            f"linear decay, {ctx.lr_steps} steps",
            "",
        ),
        (
            "Trainer",
            trainer,
            "",
            "" if alive is None else "ok" if alive else "bad",
        ),
        (
            "Last log line",
            datetime.fromtimestamp(last_t).strftime("%H:%M:%S")
            if last_t
            else "n/a",
            ("STALE " if stale else "") + (_dur(age) + " ago" if age else ""),
            "bad" if stale else "",
        ),
    ]
    return "".join(
        f'<div class="tile {cls}"><div class=k>{k}</div>'
        f"<div class=v>{html.escape(v)}</div>"
        f"<div class=s>{html.escape(s)}</div></div>"
        for k, v, s, cls in tiles
    )


def run_info(ctx: Context) -> dict[str, dict[str, Any]]:
    """Return each run's losses, checkpoints and step-0 references."""
    info: dict[str, dict[str, Any]] = {}
    for r in ctx.runs:
        hist = (
            logs.CACHE.get(r.folder / "train/history.json") or []
            if r.folder
            else []
        )
        h0 = next(
            (
                h
                for h in hist
                if isinstance(h, dict)
                and h.get("step") == 0
                and "held_out_loss" in h
            ),
            {},
        )
        ck = _checkpoints(r)
        full0 = h0.get("held_out_loss") or next(
            (
                s.get("held_out_loss_start")
                for _, s, _ in ck
                if s.get("held_out_loss_start")
            ),
            None,
        )
        train = [
            (h["step"], h["loss"])
            for h in hist
            if isinstance(h, dict) and "loss" in h and "step" in h
        ] or [(x["step"], x["loss"]) for x in ctx.log(r)]
        info[r.key] = {
            "full0": full0,
            "src0": h0.get("by_source") or {},
            "train": train,
            "ck": ck,
            "tr0": sum(v for _, v in train[:20]) / len(train[:20])
            if train
            else None,
        }
    for r in ctx.runs:
        base = info.get(r.base or r.key, info[r.key])
        info[r.key]["bfull"] = base["full0"]
        info[r.key]["bsrc"] = base["src0"]
    return info


def loss_plots(ctx: Context, info: Mapping, live: Run, mode: str) -> str:
    """Return the three loss plots of one scale: raw, start or base."""
    spo = ctx.steps_per_outbreak
    rel = mode != "raw"
    suffix = {
        "raw": "",
        "start": ", relative to start",
        "base": ", relative to base Chronos-2",
    }[mode]
    ref = (1.0,) if rel else ()
    marks = [
        (s.get("steps") or round(_size(nm) * spo), nm)
        for nm, s, _ in info[live.key]["ck"]
    ]
    smax = max([x for x, _ in info[live.key]["train"]] + [1000])
    colour = dict(zip(SOURCES, svg.PALETTE, strict=False))
    trains, held, by_source = [], [], []
    for i, r in enumerate(ctx.runs):
        rows, it = ctx.log(r), info[r.key]
        tr = [q for q in it["train"] if r.live or q[0] <= smax]
        if rel and it["tr0"]:
            tr = [(x, y / it["tr0"]) for x, y in tr]
        if r.live:
            trains += [
                (f"{r.label} raw", r.colour, tr, "dots"),
                (
                    f"{r.label} rolling mean",
                    r.colour,
                    svg.rolling(tr, 20),
                    "line",
                ),
            ]
        else:
            trains.append(
                (
                    f"{r.label} rolling mean",
                    r.colour,
                    svg.rolling(tr, 20),
                    "faint",
                )
            )
        if not rel:
            held.append(
                (
                    r.label,
                    r.colour,
                    [(x["step"], x["held"]) for x in rows],
                    "pts",
                )
            )
        else:
            den = it["full0"] if mode == "start" else it["bfull"]
            if den and it["full0"]:
                pts = [(0, it["full0"] / den)] + [
                    (
                        s.get("steps") or round(_size(nm) * spo),
                        s["held_out_loss"] / den,
                    )
                    for nm, s, _ in it["ck"]
                    if s.get("held_out_loss")
                ]
                held.append((r.label, r.colour, pts, "pts"))
        for src in SOURCES:
            pts = [(x["step"], x["src"].get(src)) for x in rows]
            if not any(v is not None for _, v in pts):
                continue
            if rel:
                den = (it["src0"] if mode == "start" else it["bsrc"]).get(src)
                if not den:
                    continue
                z = it["src0"].get(src)
                pts = ([(0, z)] if z else []) + pts
                pts = [(x, v / den) for x, v in pts if v is not None]
            by_source.append(
                _styled((f"{src} ({r.label})", colour[src], pts, "pts"), i)
            )
    held_title = (
        "Held-out pooled loss by run (first 100 blocks)"
        if mode == "raw"
        else "Held-out pooled loss by run (checkpoints, full set)"
    )
    return "".join(
        [
            svg.chart(
                "Training loss (rolling mean of 20 logged points)" + suffix,
                trains,
                marks,
                ylog=True,
                hlines=ref,
            ),
            svg.chart(held_title + suffix, held, marks, ylog=True, hlines=ref),
            svg.chart(
                "Held-out loss by source (colour is the source, style the "
                "run)" + suffix,
                by_source,
                marks,
                ylog=True,
                hlines=ref,
            ),
        ]
    )


def checkpoint_table(run: Run, info: Mapping, mode: str) -> str:
    """Return one run's checkpoint table at one loss scale."""
    rows = info["ck"]
    if not rows:
        return (
            f"<p class=mut>no checkpoints yet for {html.escape(run.label)}</p>"
        )
    head = {
        "raw": "held-out",
        "start": "held-out / start",
        "base": "held-out / base",
    }[mode]
    den = {"raw": 1, "start": info["full0"], "base": info["bfull"]}[mode]
    out = [
        f"<table><tr><th>run<th>ckpt<th>hash<th>{head}<th>held-out "
        "start<th>real-val<th>weekly<th>daily<th>3day<th>saved</tr>"
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
            _fmt(by.get("weekly")) or "n/a",
            _fmt(by.get("daily")) or "n/a",
            _fmt(by.get("3day")) or "n/a",
            datetime.fromtimestamp(mtime).strftime("%m-%d %H:%M")
            if mtime
            else "",
        ]
        out.append("<tr>" + "".join(f"<td>{c}" for c in cells) + "</tr>")
    return "".join(out) + "</table>"


def checkpoint_payload(ctx: Context, info: Mapping) -> str:
    """Return the held-out-by-checkpoint plot's data."""
    out = []
    for r in ctx.runs:
        it = info[r.key]
        pts = []
        for nm, s, _ in it["ck"]:
            by = s.get("held_out_loss_by_source") or {}
            pts.append(
                {
                    "n": s.get("n_outbreaks") or _size(nm),
                    "step": s.get("steps"),
                    "hash": str(s.get("hash", "")),
                    "all": s.get("held_out_loss"),
                    "weekly": by.get("weekly"),
                    "daily": by.get("daily"),
                    "3day": by.get("3day"),
                    "real": s.get("real_validation_loss"),
                }
            )
        pts.sort(key=lambda p: p["n"])
        out.append(
            {
                "key": r.key,
                "label": r.label,
                "color": r.colour,
                "full0": it["full0"],
                "bfull": it["bfull"],
                "points": pts,
            }
        )
    return _json({"runs": out})


def heldout_chart(ctx: Context, rows: Sequence[Mapping[str, str]]) -> str:
    """Return the held-out loss by checkpoint size, from the score table."""
    which = Classifier(ctx.runs, ctx.references)

    def x(r: Mapping[str, str]) -> float | None:
        if r["checkpoint"] == "zero-shot":
            return 125
        try:
            return float(r.get("outbreaks") or "") or None
        except ValueError:
            return None

    rows = [r for r in rows if r.get("rule") == ctx.rule]
    series = []
    for lvl, colour in (
        ("all", svg.PALETTE[0]),
        ("real_validation", svg.PALETTE[3]),
    ):
        rs = [
            r
            for r in rows
            if r.get("study") == "held_out"
            and r.get("level") == lvl
            and r.get("loss")
        ]
        for i, run in enumerate(ctx.runs):
            sized = [
                (x(r), r)
                for r in rs
                if which.run_of(r["checkpoint"]) == run.key
            ]
            pts = sorted((n, float(r["loss"])) for n, r in sized if n)
            zero = [
                (125, float(r["loss_zero_shot"]))
                for r in rs
                if r.get("loss_zero_shot")
            ][:1]
            if pts:
                series.append(
                    _styled(
                        (f"{lvl}, {run.label}", colour, zero + pts, "pts"), i
                    )
                )
    return svg.chart(
        "Held-out loss (all blocks; real validation)",
        series,
        ylog=True,
        xticks=scores.TICKS,
        xlabel="simulations (log2)",
    )


def scores_section(ctx: Context, rows: list, local: list) -> str:
    """Return the score panels for the configured runs and references."""
    if not rows:
        return "<p class=mut>no checkpoint scores yet</p>"
    pay = _json(scores.payload(rows, ctx.runs, ctx.references, ctx.rule, local))
    return (
        "<div id=sctl class=sctl></div><div class=plots2><div class=card>"
        f"<div id=splot></div></div>{heldout_chart(ctx, rows)}</div>"
        f'<script type="application/json" id=sdata>{pay}</script>'
        f"<script>{_asset('scores.js')}</script>"
    )


def germany_payload(ctx: Context, sizes: Mapping[str, int]) -> dict | None:
    """Return the German nowcast panels' data, or None if there is none."""
    built = ctx.germany()
    if not built or not built["origins"]:
        return None
    meta = germany.meta(built, ctx.runs, ctx.references, sizes)
    for m in meta:
        m["count"] = max(
            (
                len(s)
                for s in built["series"].get("DE", {}).get(m["id"], {}).values()
            ),
            default=0,
        )
    datasets = []
    for ds in ctx.datasets or [
        {"key": "germany", "label": "Germany", "truth": "", "groups": []}
    ]:
        datasets.append(
            {
                "key": ds["key"],
                "label": ds["label"],
                "truth": ds["truth"],
                "meta": meta,
                "locs": built["locs"],
                "origins": built["origins"],
                "preload": {
                    loc: germany.location(built, loc) for loc in built["locs"]
                }
                if ctx.once
                else {},
                "groups": [
                    {
                        "name": g["name"],
                        "locs": [
                            loc
                            for loc in built["locs"]
                            if re.match(g["match"], loc)
                        ],
                    }
                    for g in ds["groups"]
                ],
            }
        )
    return {"datasets": datasets, "built": built}


def germany_section(ctx: Context, sizes: Mapping[str, int]) -> str:
    """Return the German nowcast panels."""
    pay = germany_payload(ctx, sizes)
    if pay is None:
        return "<p class=mut>No German forecasts yet.</p>"
    data = _json(
        {
            "datasets": pay["datasets"],
            "runs": [
                {"key": r.key, "label": r.label, "color": r.colour}
                for r in ctx.runs
            ],
            "style": ctx.style,
        }
    )
    return (
        _asset("germany.html")
        + f'<script type="application/json" id=gdata>{data}</script>'
        f"<script>{_asset('germany.js')}</script>"
    )


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


def mail_section(ctx: Context) -> str:
    """Return the notes file's last matching lines; none without one."""
    if not ctx.mailbox:
        return ""
    if not ctx.mailbox.exists():
        return "<h2>Recent activity</h2><p class=mut>no notes yet</p>"
    rx = re.compile(ctx.mail_filter, re.I) if ctx.mail_filter else None
    with ctx.mailbox.open(errors="replace") as f:
        tail = deque(f, maxlen=400)
    lines = [x.rstrip("\n") for x in tail if not rx or rx.search(x)][-30:]
    if not lines:
        return "<h2>Recent activity</h2><p class=mut>no matching lines</p>"
    return (
        "<h2>Recent activity</h2><pre>"
        + html.escape("\n".join(lines))
        + "</pre>"
    )


def build(ctx: Context) -> str:
    """Return the dashboard page.

    Raises
    ------
    ValueError
        If no run is configured.
    """
    if not ctx.runs:
        raise ValueError("no runs to show: configure at least one run")
    live = next((r for r in ctx.runs if r.live), ctx.runs[0])
    rows = ctx.log(live)
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
    info = run_info(ctx)
    plots = "".join(
        f'<div class="mode m-{m}"><div class=plots>'
        f"{loss_plots(ctx, info, live, m)}</div></div>"
        for m in MODES
    )
    tables = "".join(
        f'<div class="mode m-{m}">'
        + "".join(checkpoint_table(r, info[r.key], m) for r in ctx.runs)
        + "</div>"
        for m in MODES
    )
    local = [
        (r.key, _size(nm), str(s.get("hash") or ""))
        for r in ctx.runs
        for nm, s, _ in info[r.key]["ck"]
        if _size(nm)
    ]
    score_rows = scores.read(ctx.source.scores)
    sizes = checkpoint_sizes(ctx, info, score_rows)
    modebar = (
        '<div class="sctl modebar"><span class=mut>Loss scale</span>'
        "<button class=tog data-mode=raw>raw</button>"
        "<button class=tog data-mode=start>relative to start</button>"
        "<button class=tog data-mode=base>relative to base Chronos-2</button>"
        "</div>"
    )
    css = _asset("style.css") + (
        f":root{{--truth:{ctx.style['ink']['truth']}}}"
        "@media(prefers-color-scheme:dark){:root{--truth:var(--fg)}}"
    )
    refresh = "" if ctx.once else "auto-refresh 60 s; "
    mirror = (
        f"scores {html.escape(ctx.mirror_status)}; "
        if ctx.mirror_status
        else ""
    )
    ck_data = checkpoint_payload(ctx, info)
    body = f"""<h1>Chronos-2 fine-tuning: {html.escape(live.label)}</h1>
<div class=mut>{datetime.now():%Y-%m-%d %H:%M:%S}; {refresh}{mirror}{sub}</div>
<h2>Status</h2><div class=tiles>{status_tiles(ctx, live, rows)}</div>
<h2>Loss plots</h2>{modebar}
<p class=mut>Relative values divide by the run's own step-0 held-out loss
(start), or by the base model's held-out loss on the same blocks (base, set
with a run's base for runs that start with labels). Training loss uses its
own first 20 steps for both. Reference line is 1.</p>{plots}
<p class=mut>Dashed lines mark the live run's saved checkpoints.</p>
<h2>Held-out loss by checkpoint</h2>{modebar}
<div id=cksrc class=sctl></div><div class=card><div id=ckplot></div></div>
<script type="application/json" id=cdata>{ck_data}</script>
<script>{_asset("checkpoints.js")}</script>
<h2>Performance on real data</h2>{scores_section(ctx, score_rows, local)}
<h2>German nowcasts by checkpoint</h2>{germany_section(ctx, sizes)}
{mail_section(ctx)}
<details class=tbls><summary><b>Tables</b></summary>
<h3>Checkpoints</h3><div class=wrap>{tables}</div></details>"""
    return (
        "<!doctype html><html><head><meta charset=utf-8>"
        '<meta name=viewport content="width=device-width,initial-scale=1">'
        "<title>Fine-tuning dashboard</title>"
        f"<style>{css}</style><script>{_asset('player.js')}</script></head>"
        f"<body>{body}<script>{_asset('mode.js')}\n"
        f"{'' if ctx.once else _asset('refresh.js')}</script></body></html>"
    )
