#!/usr/bin/env python3
"""Local dashboard for the Chronos-2 fine-tuning run (not committed).

Usage: python scripts/ft_dashboard.py --port 8765   (or --once out.html)
Standard library only. Read-only. Logs are parsed incrementally.
"""
import argparse, csv, html, json, os, re, subprocess, threading, time
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RUN = ROOT / "output/finetune/continuous-slots"
PREV = Path("/Users/lshsa2/code/seabbs/bdbv-hub-baseline-forecasts/"
            ".worktrees/fm-finetune-r4/output/finetune/continuous")
CSV = ROOT / "output/finetune/slots-vs-noslots.csv"
# Datasets for the nowcast plot. "mirror" is the folder under
# ckpt-scores/cache/ on archie and under output/finetune/dashboard-cache/.
# "groups" are (name, regex on location) pairs that fill the location picker.
# "truth" says where the truth line comes from (shown in the caption).
DATASETS = [
    {"key": "germany", "label": "Germany", "mirror": "germany",
     "truth": "latest vintage of the forecast files",
     "groups": [("national", r"^DE$"), ("states", r"^DE-"),
                ("ages", r"^(\d\d-\d\d|80\+)$")]},
]
# Runs shown on the dashboard, in plot order. Any other hash in the archie
# mirror or the scores CSV is ignored.
#   key, label, hue  identity and colour
#   dir              local run folder (train.log, train/, trainer.pid)
#   hashes           explicit checkpoint hashes, or None to use the hashes in
#                    the saved train/n*/settings.json of "dir"
#   prefix           mirror folder prefix, e.g. slots-<hash> on archie. A run
#                    with prefix None only owns the hashes in its own saved
#                    settings (or its explicit list), whatever the mirror
#                    folders are called.
#   live             the run the status tiles follow (exactly one)
#   n                optional hash -> simulations, when settings lack it
#   base_from        optional key of a run whose step-0 losses are the base
#                    model's (use for runs whose labels change the step-0
#                    loss); default is the run's own step 0
RUNS = [
    {"key": "attention bias", "label": "attention bias", "hue": 217,
     "live": True, "dir": RUN, "hashes": None, "prefix": None, "n": {},
     "base_from": "slots zeroinit"},
    {"key": "attention bias v1", "label": "attention bias (separate layouts)",
     "hue": 160, "live": False,
     "dir": ROOT / "output/finetune/continuous-slots-bias4", "hashes": None,
     "prefix": None, "n": {}, "base_from": "slots zeroinit"},
    {"key": "slots zeroinit", "label": "slots (zero init)", "hue": 330,
     "live": False, "dir": ROOT / "output/finetune/continuous-slots-zeroinit",
     "hashes": ["7d7c5d9f247e", "06dbdcc2b48d"], "prefix": None,
     "n": {"7d7c5d9f247e": 250, "06dbdcc2b48d": 500}},
    {"key": "no slots", "label": "no slots", "hue": 25, "live": False,
     "dir": PREV, "hashes": ["de129103cacc", "7962fd718a47"], "prefix": None,
     "n": {"de129103cacc": 1000, "7962fd718a47": 2000}},
]


# Checkpoints that never appear anywhere (plots, tables, the mirror, the
# scores CSV), even when a prefix would match: explicit hashes, plus every hash
# in the saved settings of the folders in "dirs".
EXCLUDE = {"hashes": ["85babc01767c", "ea31bba0123e"],
           "dirs": [ROOT / "output/finetune/continuous-slots-labels"]}


def excluded_hashes():
    out = set(EXCLUDE["hashes"])
    for d in EXCLUDE["dirs"]:
        for f in Path(d).glob("train/n*/settings.json"):
            s = JC.get(str(f))
            if isinstance(s, dict) and s.get("hash"):
                out.add(str(s["hash"]))
    return out


def live_cfg():
    return next((c for c in RUNS if c.get("live")), RUNS[0])


def live_dir():
    return Path(live_cfg()["dir"])


CKPTS = [250, 500, 750, 1000, 2000, 4000, 8000]  # fallback prefixes
# Optional remote mailbox shown under Recent activity; None hides the section.
MAILBOX = {"host": "archie", "path": "~/bdbv-archie-mailbox.md", "lines": 30}
MAIL_RE = re.compile(
    r"slots|ckpt-scores|agg-slots|slots-check|fm-finetune3|sim round|"
    r"sim-round|sims? rounds?", re.I)
LOG_RE = re.compile(
    r"^(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d) \[train\] step (\d+): "
    r"loss ([\d.eE+-]+), held-out \([^)]*\) ([\d.eE+-]+)"
    r"(?:, by source 3day ([\d.eE+-]+), daily ([\d.eE+-]+), "
    r"weekly ([\d.eE+-]+))?")


def jload(p):
    try:
        return json.loads(Path(p).read_text())
    except Exception:
        return None


class Log:
    """Incrementally parsed train.log."""
    def __init__(self, path):
        self.path, self.off, self.rows = Path(path), 0, {}

    def update(self):
        try:
            size = self.path.stat().st_size
        except OSError:
            return
        if size < self.off:
            self.off, self.rows = 0, {}
        if size == self.off:
            return
        with open(self.path, "rb") as f:
            f.seek(self.off)
            data = f.read()
        end = data.rfind(b"\n") + 1  # only whole lines
        self.off += end
        for line in data[:end].decode("utf-8", "replace").splitlines():
            m = LOG_RE.match(line)
            if not m:
                continue
            g = m.groups()
            t = datetime.strptime(g[0], "%Y-%m-%d %H:%M:%S").timestamp()
            fl = lambda x: float(x) if x is not None else None
            self.rows[int(g[1])] = dict(
                t=t, step=int(g[1]), loss=fl(g[2]), held=fl(g[3]),
                src={"3day": fl(g[4]), "daily": fl(g[5]), "weekly": fl(g[6])})

    def sorted(self):
        return [self.rows[k] for k in sorted(self.rows)]


class Cache:
    """Re-read a JSON file only when its mtime or size changes."""
    def __init__(self):
        self.d = {}

    def get(self, p):
        try:
            st = Path(p).stat()
        except OSError:
            return None
        key = (st.st_mtime_ns, st.st_size)
        if p not in self.d or self.d[p][0] != key:
            self.d[p] = (key, jload(p))
        return self.d[p][1]


LOGS = {}


def run_log(cfg):
    d = str(Path(cfg["dir"]) / "train.log")
    if d not in LOGS:
        LOGS[d] = Log(d)
    LOGS[d].update()
    return LOGS[d].sorted()
JC = Cache()
MAIL = {"t": 0.0, "lines": [], "ok": None, "busy": False}


def mail_fetch():
    try:
        r = subprocess.run(
            ["ssh", "-o", "ConnectTimeout=10", "-o", "BatchMode=yes",
             MAILBOX["host"],
             f"tail -n {MAILBOX['lines']} {MAILBOX['path']}"],
            capture_output=True, text=True, timeout=20)
        if r.returncode == 0:
            MAIL["lines"] = [l for l in r.stdout.splitlines()
                             if MAIL_RE.search(l)]
            MAIL["ok"] = True
        else:
            MAIL["ok"] = False
    except Exception:
        MAIL["ok"] = False
    MAIL["t"] = time.time()
    MAIL["busy"] = False


def mail_kick(wait=False):
    if time.time() - MAIL["t"] < 60 or MAIL["busy"]:
        return
    MAIL["busy"] = True
    th = threading.Thread(target=mail_fetch, daemon=True)
    th.start()
    if wait:
        th.join(25)


def checkpoints(base):
    out = []
    for d in sorted(Path(base).glob("train/n*"),
                    key=lambda p: int(re.sub(r"\D", "", p.name) or 0)):
        s = JC.get(str(d / "settings.json"))
        if isinstance(s, dict):
            try:
                mt = (d / "settings.json").stat().st_mtime
            except OSError:
                mt = None
            out.append((d.name, s, mt))
    return out


def trainer_cmdline(rundir):
    """The trainer's command line (ps on trainer.pid), '' if unavailable."""
    try:
        pid = int((rundir / "trainer.pid").read_text().split()[0])
        r = subprocess.run(["ps", "-p", str(pid), "-o", "args="],
                           capture_output=True, text=True, timeout=5)
        return r.stdout
    except Exception:
        return ""


def trainer_prefixes(rundir, args=None):
    """Checkpoint sizes: trainer's --prefixes, else saved folders plus the
    default list."""
    args = trainer_cmdline(rundir) if args is None else args
    m = re.search(r"--prefixes((?:\s+\d+)+)", args)
    if m:
        return sorted(int(x) for x in m.group(1).split())
    done = [int(re.sub(r"\D", "", d.name)) for d in
            rundir.glob("train/n*") if re.sub(r"\D", "", d.name)]
    return sorted(set(CKPTS) | set(done))


def run_params(rundir, hist, step):
    """Schedule facts of a run, from its command line, settings and history.

    Returns lr_steps, the optimiser groups (name -> base rate), each group's
    current rate, its rate history and the prefixes.
    """
    args = trainer_cmdline(rundir)
    sets = [(int(re.sub(r"\D", "", d.name) or 0), JC.get(str(d / "settings.json")))
            for d in rundir.glob("train/n*")]
    sets = [s for _, s in sorted(
        ((n, s) for n, s in sets if isinstance(s, dict)),
        key=lambda x: x[0])]
    s = sets[-1] if sets else {}
    m = re.search(r"--lr-steps\s+(\d+)", args)
    lr_steps = int(m.group(1)) if m else int(s.get("lr_steps") or 0)
    groups = {}
    base_name = "LoRA" if s.get("lora") or "--series-bias" in args else \
        "adapter"
    m = re.search(r"--learning-rate\s+(\S+)", args)
    if m:
        groups[base_name] = float(m.group(1))
    elif s.get("learning_rate"):
        groups[base_name] = float(s["learning_rate"])
    for m in re.finditer(r"--([a-z][a-z-]*)-lr\s+(\S+)", args):
        groups[m.group(1)] = float(m.group(2))
    for k, v in s.items():
        if isinstance(v, dict) and v.get("learning_rate") and \
                not any(re.fullmatch(r"\w*" + k.split("_")[-1] + r"\w*", g)
                        for g in groups if g != base_name):
            groups.setdefault(k.replace("series_", ""), float(v["learning_rate"]))
    logged = {}
    for h in hist if isinstance(hist, list) else []:
        v = h.get("lr") if isinstance(h, dict) else None
        if v is None or "step" not in h:
            continue
        v = v if isinstance(v, dict) else (
            {f"group {i}": x for i, x in enumerate(v)}
            if isinstance(v, list) else {"lr": v})
        for g, x in v.items():
            logged.setdefault(g, []).append((h["step"], float(x)))
    # one entry per group name, ignoring case; history's spelling wins
    disp = {g.lower(): g for g in logged}
    for g in groups:
        disp.setdefault(g.lower(), g)
    merged = {}
    for g, v in groups.items():
        merged.setdefault(disp[g.lower()], v)
    groups = merged
    logged = {disp[g.lower()]: v for g, v in logged.items()}
    for g in logged:
        groups.setdefault(g, logged[g][0][1])
    current = {}
    for g, base in groups.items():
        if g in logged:
            current[g] = logged[g][-1][1]
        elif step is not None:
            current[g] = base * max(0.0, 1 - step / lr_steps) if lr_steps \
                else base
    return {"lr_steps": lr_steps, "groups": groups, "current": current,
            "logged": logged, "prefixes": trainer_prefixes(rundir, args)}


def lr_card(P, step, small=(190, 150)):
    """Small learning-rate-against-step chart for every optimiser group."""
    import math
    if not P["groups"]:
        return ""
    W, H = small
    L, R, T, B = 34, 6, 6, 18
    xmax = P["lr_steps"] or max([step or 0, 1])
    sx = lambda x: L + x / xmax * (W - L - R)
    vals = [max(b, 1e-12) for b in P["groups"].values()]
    # linear decay reaches zero, so the log axis stops at 1% of the base
    lo = math.log10(min(vals) * (0.01 if P["lr_steps"] else 1)) - 0.1
    hi = math.log10(max(vals)) + 0.15
    sy = lambda v: T + (1 - (math.log10(max(v, 1e-12)) - lo) / (hi - lo)) \
        * (H - T - B)
    o = [f'<svg viewBox="0 0 {W} {H}" width="{W}" height="{H}">']
    for e in range(math.ceil(lo), math.floor(hi) + 1):
        o.append(f'<line x1="{L}" x2="{W-R}" y1="{sy(10**e):.1f}" '
                 f'y2="{sy(10**e):.1f}" class=grid /><text x="{L-3}" '
                 f'y="{sy(10**e)+3:.1f}" text-anchor=end class=ax>1e{e}</text>')
    o.append(f'<text x="{L}" y="{H-4}" class=ax>0</text><text x="{W-R}" '
             f'y="{H-4}" text-anchor=end class=ax>{xmax}</text>')
    leg = []
    for i, (g, base) in enumerate(P["groups"].items()):
        col = PAL[i % len(PAL)]
        pts = P["logged"].get(g) or (
            [(0, base), (0.99 * xmax, base * 0.01)] if P["lr_steps"]
            else [(0, base), (xmax, base)])
        pts = [(x, max(y, 1e-12)) for x, y in pts]
        o.append('<polyline points="' + " ".join(
            f"{sx(x):.1f},{sy(y):.1f}" for x, y in pts)
            + f'" fill=none stroke="{col}" stroke-width=1.6 />')
        cur = P["current"].get(g)
        if cur is not None and step is not None:
            o.append(f'<circle cx="{sx(step):.1f}" cy="{sy(cur):.1f}" r=3 '
                     f'fill="{col}"><title>{g} {cur:.3e} at step {step}'
                     f'</title></circle>')
        leg.append(f'<span><i style="background:{col}"></i>{html.escape(g)}'
                   f'</span>')
    o.append("</svg>")
    return (f'<div class=card style="padding:4px;line-height:1.2">'
            f'<div class=mut style="font-size:11px">learning rate vs step'
            f'</div>{"".join(o)}<div class=leg style="font-size:11px">'
            f'{"".join(leg)}</div></div>')


def pid_state(rundir):
    pf = rundir / "trainer.pid"
    try:
        pid = int(pf.read_text().split()[0])
        try:
            os.kill(pid, 0)
            return f"alive (pid {pid})", True
        except PermissionError:
            return f"alive (pid {pid})", True
        except OSError:
            return f"DEAD (pid {pid})", False
    except Exception:
        pass
    try:
        r = subprocess.run(["pgrep", "-f", "finetune_stream"],
                           capture_output=True, text=True, timeout=5)
        pids = r.stdout.split()
        if pids:
            return f"alive (pid {pids[0]})", True
        if r.returncode == 1:
            return "DEAD (no process)", False
    except Exception:
        pass
    return "unknown", None


# ---------- SVG plotting ----------
PAL = ["var(--c1)", "var(--c2)", "var(--c3)", "var(--c4)"]


def nice(lo, hi, n=5):
    if hi <= lo:
        hi = lo + 1
    step = (hi - lo) / n
    mag = 10 ** int(f"{step:e}".split("e")[1])
    for m in (1, 2, 2.5, 5, 10):
        if m * mag >= step:
            step = m * mag
            break
    a = int(lo // step) * step
    ticks = []
    v = a
    while v <= hi + step:
        ticks.append(v)
        v += step
    return ticks


DASHES = {"dash": ("5 3", "dashed"), "dot": ("1.5 3", "dotted"),
          "dd": ("8 3 1.5 3", "dash-dot")}


def chart(title, series, marks=(), ylog=False, W=640, H=320,
          xticks=None, hlines=(), xlabel="step", note="", xmax=None):
    """series: list of (label, colour, points[(x,y)], kind) kind line|dots"""
    import math
    if xmax is not None:
        series = [s[:2] + ([p for p in s[2] if p[0] <= xmax],) + s[3:]
                  for s in series]
    pts = [p for s in series for p in s[2] if p[1] is not None]
    if not pts:
        return f"<div class=card><h3>{title}</h3><p class=mut>no data</p></div>"
    f = (lambda y: math.log10(max(y, 1e-9))) if ylog else (lambda y: y)
    xs = [p[0] for p in pts]
    ys = [f(p[1]) for p in pts]
    x0, x1 = 0, max(max(xs), 1)
    if xmax is not None:
        x1 = xmax
    if xticks:
        x0, x1 = math.log2(xticks[0][0]), math.log2(xticks[-1][0])
        xs = [math.log2(x) for x in xs]
    y0, y1 = min(ys), max(ys)
    pad = (y1 - y0) * 0.05 or 0.1
    y0, y1 = y0 - pad, y1 + pad
    L, R, T, B = 52, 12, 10, 34
    xm = math.log2 if xticks else (lambda x: x)
    sx = lambda x: L + (xm(x) - x0) / (x1 - x0) * (W - L - R)
    sy = lambda y: T + (1 - (f(y) - y0) / (y1 - y0)) * (H - T - B)
    o = [f'<svg viewBox="0 0 {W} {H}" width="100%" role="img">']
    for t in nice(y0, y1):
        if y0 <= t <= y1:
            yy = T + (1 - (t - y0) / (y1 - y0)) * (H - T - B)
            lab = f"{10 ** t:.3g}" if ylog else f"{t:.3g}"
            o.append(f'<line x1="{L}" x2="{W-R}" y1="{yy:.1f}" y2="{yy:.1f}" '
                     f'class=grid /><text x="{L-6}" y="{yy+4:.1f}" '
                     f'text-anchor=end class=ax>{lab}</text>')
    for t, lab in (xticks or [(t, f"{t:g}") for t in nice(x0, x1)
                              if x0 <= t <= x1]):
        xx = sx(t)
        o.append(f'<text x="{xx:.1f}" y="{H-14}" text-anchor=middle '
                 f'class=ax>{lab}</text>')
        if xticks:
            o.append(f'<line x1="{xx:.1f}" x2="{xx:.1f}" y1="{T}" '
                     f'y2="{H-B}" class=grid />')
    o.append(f'<text x="{(L+W-R)/2}" y="{H-1}" text-anchor=middle '
             f'class=ax>{xlabel}</text>')
    for hv in hlines:
        if y0 <= f(hv) <= y1:
            o.append(f'<line x1="{L}" x2="{W-R}" y1="{sy(hv):.1f}" '
                     f'y2="{sy(hv):.1f}" class=mark />')
    for m, lab in marks:
        if not xticks and x0 <= m <= x1:
            o.append(f'<line x1="{sx(m):.1f}" x2="{sx(m):.1f}" y1="{T}" '
                     f'y2="{H-B}" class=mark /><text x="{sx(m)+2:.1f}" '
                     f'y="{T+10}" class=ax>{lab}</text>')
    for i, s in enumerate(series):
        lab, col, p, kind = s[:4]
        dash = (f' stroke-dasharray="{DASHES[s[4]][0]}"' if len(s) > 4
                else "")
        op = ' opacity=".4"' if kind == "faint" else ""
        p = [q for q in p if q[1] is not None]
        if kind == "dots":
            for x, y in p:
                o.append(f'<circle cx="{sx(x):.1f}" cy="{sy(y):.1f}" r="1.8" '
                         f'fill="{col}" opacity=".35" />')
        elif kind == "pts":
            d = " ".join(f"{sx(x):.1f},{sy(y):.1f}" for x, y in p)
            o.append(f'<polyline points="{d}" fill=none stroke="{col}" '
                     f'stroke-width=1.6{dash} />')
            for x, y in p:
                o.append(f'<circle cx="{sx(x):.1f}" cy="{sy(y):.1f}" r=2.4 '
                         f'fill="{col}"><title>{lab} {xlabel} {x}: {y:.4f}'
                         f'</title></circle>')
        else:
            d = " ".join(f"{sx(x):.1f},{sy(y):.1f}" for x, y in p)
            o.append(f'<polyline points="{d}" fill=none stroke="{col}" '
                     f'stroke-width=1.6{dash}{op} />')
    o.append("</svg>")
    leg = "".join(
        f'<span><i style="background:{s[1]}'
        f'{";opacity:.4" if s[3] == "faint" else ""}"></i>{s[0]}'
        f'{" (" + DASHES[s[4]][1] + ")" if len(s) > 4 else ""}</span>'
        for s in series)
    return (f"<div class=card><h3>{title}</h3>{''.join(o)}"
            f"<div class=leg>{leg}</div>{note}</div>")


def rolling(rows, k):
    out, acc = [], []
    for x, y in rows:
        acc.append(y)
        if len(acc) > k:
            acc.pop(0)
        out.append((x, sum(acc) / len(acc)))
    return out



# ---------- remote (archie) data, fetched in background and cached ----------
CACHE = ROOT / "output/finetune/dashboard-cache"
SCORES_REMOTE = ("~/code/seabbs/BDBVHubRenewal/.worktrees/checkpoint-scoring/"
                 "evidence/checkpoint-scores.csv")
REVIEW = "~/bdbv-ckpt-review"
OVERRIDE = {"scores": None, "review_dir": None, "germany": None}  # local test overrides
REM = {k: {"t": 0.0, "busy": False, "ok": None, "msg": ""}
       for k in ("scores", "md", "germany")}
REM_EVERY = {"scores": 60, "md": 60, "germany": 600}


def ssh_cat(path):
    r = subprocess.run(
        ["ssh", "-o", "ConnectTimeout=10", "-o", "BatchMode=yes", "archie",
         f"cat {path}"], capture_output=True, timeout=60)
    if r.returncode != 0:
        raise RuntimeError(r.stderr.decode("utf-8", "replace")[-200:].strip()
                           or f"ssh exit {r.returncode}")
    return r.stdout


def fetch_review(rel):
    if OVERRIDE["review_dir"]:
        return (Path(OVERRIDE["review_dir"]) / rel).read_bytes()
    return ssh_cat(f"{REVIEW}/{rel}")


def job_scores():
    if OVERRIDE["scores"]:
        data = Path(OVERRIDE["scores"]).read_bytes()
    else:
        data = ssh_cat(SCORES_REMOTE)
    CACHE.mkdir(parents=True, exist_ok=True)
    (CACHE / "checkpoint-scores.csv").write_bytes(data)


def job_md():
    data = fetch_review("slots-vs-noslots.md")
    CACHE.mkdir(parents=True, exist_ok=True)
    (CACHE / "slots-vs-noslots.md").write_bytes(data)


JOBS = {"scores": job_scores, "md": job_md}


def rem_run(k):
    st = REM[k]
    try:
        JOBS[k]()
        st["ok"], st["msg"] = True, ""
    except Exception as e:
        st["ok"], st["msg"] = False, str(e)[:160]
    st["t"] = time.time()
    st["busy"] = False


def rem_kick(k, wait=False):
    st = REM[k]
    if st["busy"] or time.time() - st["t"] < REM_EVERY[k]:
        return
    st["busy"] = True
    th = threading.Thread(target=rem_run, args=(k,), daemon=True)
    th.start()
    if wait:
        th.join(90)


def note(k, what):
    st = REM[k]
    if st["ok"] is False:
        return (f'<p class=mut>{what}: fetch from archie failed '
                f'({html.escape(st["msg"])}); showing cached copy if any.</p>')
    if st["ok"] is None:
        return f'<p class=mut>{what}: fetching from archie, reload shortly.</p>'
    return ""


def read_scores():
    try:
        return list(csv.DictReader((CACHE / "checkpoint-scores.csv").open()))
    except Exception:
        return []


def num(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


REF_COL = {"zero": "#16a34a", "complete": "#9333ea", "rw": "#6b7280"}
XT = [(125, "zero-shot"), (250, "250"), (500, "500"), (1000, "1000"),
      (2000, "2000"), (4000, "4000"), (8000, "8000")]
RULE = "own-quantile-wis_crps-1000-common"


def run_hashes(cfg):
    """The hashes a run owns (saved settings for a directory-based run)."""
    if cfg["hashes"] is not None:
        return set(cfg["hashes"])
    return {str(s.get("hash")) for d in Path(cfg["dir"]).glob("train/n*")
            for s in [JC.get(str(d / "settings.json"))]
            if isinstance(s, dict) and s.get("hash")}


def hash_run(h, pref=None):
    """Run key for a checkpoint hash, or None to ignore it."""
    if h in excluded_hashes():
        return None
    for cfg in RUNS:
        if cfg["hashes"] is not None and h in cfg["hashes"]:
            return cfg["key"]
    for cfg in RUNS:
        if cfg["hashes"] is None and (
                (pref and pref == cfg.get("prefix")) or h in run_hashes(cfg)):
            return cfg["key"]
    return None


def base_src(model):
    """daily or weekly from a model name, None for other models."""
    return "daily" if "daily" in model else \
        "weekly" if "weekly" in model else None


def norm_hash(name):
    """'chronos-slots-abc', 'slots-abc' and 'abc' -> ('abc', 'slots'|None)."""
    m = re.match(r"^(?:chronos-)?(?:([a-z][a-z-]*?)-)?([0-9a-f]{6,}.*)$", name)
    if not m:
        return name, None
    return m.group(2), (m.group(1) if m.group(1) != "ft" else None)


def run_colour(run, rank=0, count=1):
    hue = next((c["hue"] for c in RUNS if c["key"] == run), None)
    if hue is None:
        return "#6b7280"
    light = 46 if count < 2 else 66 - 28 * rank / (count - 1)
    return f"hsl({hue},80%,{light:.0f}%)"


def score_classify(r):
    """Series key for a scores row, or None to skip."""
    mdl, ck = r["model"], r["checkpoint"]
    if mdl == "randomwalknowcast":
        return "rw"
    if mdl.endswith("_complete"):
        return "complete" if ck == "zero-shot" else None
    if ck == "zero-shot":
        return "zero"
    h, pref = norm_hash(ck)
    return hash_run(h, pref)
    return None


LEVEL_ORDER = ["national", "states", "age", "provinces"]
STUDY_ORDER = ["germany", "ebola", "heldout"]


def score_payload(rows, local):
    """Rows for the client-side score chart and not-yet-scored markers."""
    out, seen_h, seen_n = [], set(), {}
    variants, vbase, targets = {}, {}, {}
    levels, metrics, vlabel = {}, {}, {}
    run_keys = {c["key"] for c in RUNS}
    for r in rows:
        if r.get("rule") != RULE:
            continue
        ser = score_classify(r)
        if ser is None:
            continue
        # The source is the layout column when the row has one (held-out
        # synthetic scores), else daily or weekly from the model name.
        lay = (r.get("layout") or "").strip()
        base = base_src(r["model"])
        src = lay or base
        if src is None:
            continue
        h = norm_hash(r["checkpoint"])[0]
        n = int(num(r.get("outbreaks")) or 0)
        tgt = (r.get("target") or "").strip()
        for col, metric in (("rel_wis", "wis"), ("rel_crps", "crps")):
            v = num(r.get(col))
            if v is None:
                continue
            out.append([r["study"], r["level"], src, ser, n,
                        "" if r["checkpoint"] == "zero-shot" else h,
                        round(v, 4), tgt, metric])
            st = r["study"]
            targets.setdefault(st, set()).add(tgt)
            levels.setdefault(st, set()).add(r["level"])
            metrics.setdefault(st, set()).add(metric)
            vlabel[st] = "layout" if lay else vlabel.get(st, "source")
            if ser in run_keys:
                variants.setdefault(st, set()).add(src)
            vbase[src] = base or src
            seen_h.add(h)
            seen_n.setdefault(ser, set()).add(n)
    pending = []
    for run, n, h in local:
        scored = (h in seen_h) if h else (n in seen_n.get(run, set()))
        if not scored:
            pending.append({"run": run, "n": n, "hash": h or ""})
    series = [{"key": c["key"], "label": c["label"], "run": True,
               "color": run_colour(c["key"])} for c in RUNS] + [
        {"key": "zero", "label": "zero-shot ragged", "color": REF_COL["zero"]},
        {"key": "complete", "label": "complete-only",
         "color": REF_COL["complete"]},
        {"key": "rw", "label": "randomwalknowcast", "color": REF_COL["rw"]}]
    return {"rows": out, "pending": pending, "xt": XT, "series": series,
            "col": {s["key"]: s["color"] for s in series},
            "variants": {k: sorted(v) for k, v in variants.items()},
            "targets": {k: sorted(v) for k, v in targets.items()},
            "vbase": vbase, "vlabel": vlabel,
            "metrics": {k: sorted(v) for k, v in metrics.items()},
            "levels": {k: sorted(v, key=lambda x: (
                LEVEL_ORDER.index(x) if x in LEVEL_ORDER else 99, x))
                for k, v in levels.items()},
            "studies": sorted({r[0] for r in out}, key=lambda x: (
                STUDY_ORDER.index(x) if x in STUDY_ORDER else 99, x)),
            "defaults": [next((c["key"] for c in RUNS if c["key"] in seen_n),
                              live_cfg()["key"]), "zero", "rw"]}


def heldout_chart(rows):
    ho = []
    xval = lambda r: 125 if r["checkpoint"] == "zero-shot" else (
        num(r.get("outbreaks")) or None)
    rows = [r for r in rows if r.get("rule") == RULE and
            r["study"] == "held_out" and r["level"] == "all"
            and num(r.get("loss")) is not None]
    for c in RUNS:
        pts = sorted((xval(r), num(r["loss"])) for r in rows
                     if xval(r) and hash_run(*norm_hash(
                         r["checkpoint"])) == c["key"])
        z = [(125, num(r["loss_zero_shot"])) for r in rows
             if num(r.get("loss_zero_shot")) is not None][:1]
        if pts:
            ho.append((c["label"], run_colour(c["key"]), z + pts, "pts"))
    return chart("Held-out loss (all blocks)", ho, xticks=XT,
                 xlabel="simulations (log2)")


SC_JS = r"""
(function(){
var P=JSON.parse(document.getElementById("sdata").textContent);
var $=function(i){return document.getElementById(i)};
var SERS=P.series.map(function(s){return [s.key,s.label]}),RUNK=P.series.filter(function(s){return s.run}).map(function(s){return s.key});
var STUDIES=P.studies.map(function(s){return [s,{germany:"Germany",ebola:"BDBV",heldout:"held-out synthetic"}[s]||s]});
function LVL(st){return (P.levels[st]||[]).map(function(l){return [l,l=="age"?"ages":l]})}
var LV={};P.studies.forEach(function(s){LV[s]=LVL(s)});
var MK=["circle","square","triangle","diamond"];
var VDASH=["","6 3","2 3","8 3 2 3"],REFK=["zero","complete","rw"];
var S={study:STUDIES.length?STUDIES[0][0]:"",vars:null,target:null,metric:null,lv:["national"],cmp:false,log:false};
function SEL(){return ftSel.get()}
try{var sv=JSON.parse(sessionStorage.getItem("fts5")||"null");if(sv)S=sv}catch(e){}
function save(){try{sessionStorage.setItem("fts5",JSON.stringify(S))}catch(e){}}
function mark(k,x,y,c,t){var r=3.6,s="";
 if(k=="circle")s='<circle cx="'+x+'" cy="'+y+'" r="'+r+'" fill="'+c+'">';
 else if(k=="square")s='<rect x="'+(x-r)+'" y="'+(y-r)+'" width="'+2*r+'" height="'+2*r+'" fill="'+c+'">';
 else if(k=="triangle")s='<polygon points="'+x+','+(y-r-1)+' '+(x+r+1)+','+(y+r)+' '+(x-r-1)+','+(y+r)+'" fill="'+c+'">';
 else s='<polygon points="'+x+','+(y-r-1)+' '+(x+r+1)+','+y+' '+x+','+(y+r+1)+' '+(x-r-1)+','+y+'" fill="'+c+'">';
 return s+'<title>'+t+'</title>'+(k=="circle"?"</circle>":k=="square"?"</rect>":"</polygon>")}
function btn(id,label,on,sty){return '<button data-id="'+id+'" class="tog'+(on?" on":"")+'"'+(sty?' style="'+sty+'"':"")+'>'+label+'</button>'}
function MTS(){return P.metrics[S.study]||[]}
function MT(){var a=MTS();return a.indexOf(S.metric)>=0?S.metric:(a.indexOf("wis")>=0?"wis":a[0]||"")}
function TGTS(){return P.targets[S.study]||[]}
function TGT(){var a=TGTS();return a.indexOf(S.target)>=0?S.target:(a.indexOf("7day")>=0?"7day":a[0]||"")}
function tlab(v){return v=="7day"?"7-day":(v||"all")}
function VARS(){return P.variants[S.study]||[]}
function baseOf(v){return P.vbase[v]||v}
function curVars(){var av=VARS(),sel=(S.vars||[]).filter(function(v){return av.indexOf(v)>=0});
 if(!sel.length)sel=av.indexOf("daily")>=0?["daily"]:av.slice(0,1);
 return sel}
function controls(){
 var lv=LV[S.study];
 $("sctl").innerHTML=STUDIES.map(function(a){return btn("study:"+a[0],a[1],S.study==a[0])}).join("")
 +(MTS().length>1?'<span class=sep></span><span class=mut>score</span>'+MTS().map(function(a){return btn("met:"+a,a.toUpperCase(),MT()==a)}).join(""):"")
 +(TGTS().length>1?'<span class=sep></span><span class=mut>target</span>'+TGTS().map(function(a){return btn("tgt:"+a,tlab(a),TGT()==a)}).join(""):"")
 +'<span class=sep></span><span class=mut>'+(P.vlabel[S.study]||"source")+'</span>'+VARS().map(function(a){return btn("var:"+a,a,curVars().indexOf(a)>=0)}).join("")
 +'<span class=sep></span>'+lv.map(function(a){return btn("lv:"+a[0],a[1],S.lv.indexOf(a[0])>=0)}).join("")+btn("cmp","compare levels",S.cmp)

 +'<span class=sep></span>'+btn("log","log scale",S.log)}
function tick(arr,v){var i=arr.indexOf(v);if(i>=0)arr.splice(i,1);else arr.push(v)}
$("sctl").onclick=function(e){var b=e.target.closest("button");if(!b)return;var p=b.dataset.id.split(":"),k=p[0],v=p[1];
 if(k=="study"){S.study=v;S.vars=null;S.lv=S.lv.filter(function(l){return LV[v].some(function(a){return a[0]==l})});if(!S.lv.length)S.lv=[LV[v].length?LV[v][0][0]:"national"]}
 else if(k=="tgt")S.target=v;
 else if(k=="met")S.metric=v;
 else if(k=="var"){var cv=curVars().slice();var i=cv.indexOf(v);if(i>=0){if(cv.length>1)cv.splice(i,1)}else cv.push(v);S.vars=cv}
 else if(k=="lv"){if(S.cmp){tick(S.lv,v);if(!S.lv.length)S.lv=[v]}else S.lv=[v]}
 else if(k=="cmp"){S.cmp=!S.cmp;if(!S.cmp)S.lv=[S.lv[0]]}
 else if(k=="log")S.log=!S.log;
 else{}
 save();draw()};
function fmt(v){return (+v).toPrecision(3)}
function niceTicks(lo,hi){var st=(hi-lo)/4,m=Math.pow(10,Math.floor(Math.log10(st))),k=[1,2,2.5,5,10].find(function(x){return x*m>=st})*m,o=[];
 for(var v=Math.ceil(lo/k)*k;v<=hi+1e-9;v+=k)o.push(+v.toFixed(6));return o}
function draw(){
 var avl=LV[S.study]||[];
 if(avl.length&&!avl.some(function(a){return S.lv.indexOf(a[0])>=0}))S.lv=[avl[0][0]];
 controls();
 var lv=avl.filter(function(a){return S.lv.indexOf(a[0])>=0});
 var vs=curVars();
 var TG=TGT(),MM=MT();
 function rowsFor(v){return P.rows.filter(function(r){return r[0]==S.study&&(r[7]==TG||r[7]=="")&&r[8]==MM&&(r[2]==v||(REFK.indexOf(r[3])>=0&&r[2]==baseOf(v)))})}
 var rows=[];vs.forEach(function(v){rowsFor(v).forEach(function(r){if(rows.indexOf(r)<0)rows.push(r)})});
 var shown=rows.filter(function(r){return SEL().indexOf(r[3])>=0&&r[3]!="rw"&&lv.some(function(l){return l[0]==r[1]})});
 var W=640,H=320,L=44,R=16,T=10,B=44;
 var ns=shown.filter(function(r){return r[4]>0}).map(function(r){return r[4]});
 var nmin=ns.length?Math.min.apply(null,ns):250,nmax=ns.length?Math.max.apply(null,ns):250;
 var zx=nmin/2,x0=Math.log2(zx),x1=Math.log2(nmax*1.18);
 var sx=function(n){return L+(Math.log2(n||zx)-x0)/(x1-x0)*(W-L-R)};
 var vals=shown.map(function(r){return r[6]});
 // y range from the data shown; the randomwalknowcast line (1) counts only if close
 if(!vals.length)vals=[1];
 var vmin=Math.min.apply(null,vals),vmax=Math.max.apply(null,vals),spn=(vmax-vmin)||0.1*vmax;
 if(1>=vmin-spn&&1<=vmax+spn){vmin=Math.min(vmin,1);vmax=Math.max(vmax,1)}
 var showRw=1>=vmin&&1<=vmax,lo,hi,sy;
 if(S.log){lo=Math.log(vmin)-0.1*(Math.log(vmax)-Math.log(vmin)||0.1);hi=Math.log(vmax)+0.1*(Math.log(vmax)-Math.log(vmin)||0.1);sy=function(v){return T+(1-(Math.log(v)-lo)/(hi-lo))*(H-T-B)}}
 else{var pd=0.1*((vmax-vmin)||0.1*vmax);lo=Math.max(0,vmin-pd);hi=vmax+pd;sy=function(v){return T+(1-(v-lo)/(hi-lo))*(H-T-B)}}
 var h='<svg viewBox="0 0 '+W+' '+H+'" width="100%">';
 var yt=S.log?[0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1,1.1,1.25,1.5,2,3,4,6,8].filter(function(v){return Math.log(v)>=lo&&Math.log(v)<=hi}):niceTicks(lo,hi);
 yt.forEach(function(v){h+='<line x1="'+L+'" x2="'+(W-R)+'" y1="'+sy(v)+'" y2="'+sy(v)+'" class=grid /><text x="'+(L-5)+'" y="'+(sy(v)+3)+'" text-anchor=end class=ax>'+v+'</text>'});
 var xt=[[zx,"zero-shot"]].concat(Array.from(new Set(ns)).sort(function(a,b){return a-b}).map(function(n){return [n,""+n]}));
 xt.forEach(function(t){h+='<line x1="'+sx(t[0])+'" x2="'+sx(t[0])+'" y1="'+T+'" y2="'+(H-B)+'" class=grid /><text x="'+sx(t[0])+'" y="'+(H-B+14)+'" text-anchor=middle class=ax>'+t[1]+'</text>'});
 h+='<text x="'+(W/2)+'" y="'+(H-6)+'" text-anchor=middle class=ax>simulations (log2)</text>';
 P.pending.forEach(function(p){if(p.n<=nmax*1.18&&p.n>=zx){h+='<line x1="'+sx(p.n)+'" x2="'+sx(p.n)+'" y1="'+T+'" y2="'+(H-B)+'" class=mark /><text x="'+(sx(p.n)+3)+'" y="'+(T+10)+'" class=ax>n'+p.n+' scoring</text>'}});
 if(SEL().indexOf("rw")>=0&&showRw)h+='<line x1="'+L+'" x2="'+(W-R)+'" y1="'+sy(1)+'" y2="'+sy(1)+'" stroke="'+P.col.rw+'" stroke-width=2 ><title>randomwalknowcast = 1</title></line>';
 var any=false,sum=[],drawn={};
 lv.forEach(function(l,li){var mk=MK[li];
  RUNK.concat(["zero","complete"]).forEach(function(s){
   if(SEL().indexOf(s)<0)return;
   var lab=SERS.find(function(x){return x[0]==s})[1],c=P.col[s];
   vs.forEach(function(v,vi){
    var rowsV=rowsFor(v);
    var rs=rowsV.filter(function(r){return r[1]==l[0]&&r[3]==s}).sort(function(a,b){return a[4]-b[4]});
    if(!rs.length)return;any=true;
    var vtag=vs.length>1?" ["+v+"]":"",dash=vs.length>1&&VDASH[vi%4]?' stroke-dasharray="'+VDASH[vi%4]+'"':"";
    if(s=="zero"||s=="complete"){var val=rs[0][6],key=l[0]+s+val;if(drawn[key])return;drawn[key]=1;
     h+='<line x1="'+L+'" x2="'+(W-R)+'" y1="'+sy(val)+'" y2="'+sy(val)+'" stroke="'+c+'" stroke-width=1.5 stroke-dasharray="'+(s=="zero"?"6 3":"2 3")+'" />';
     h+=mark(mk,sx(zx),sy(val),c,l[1]+", "+lab+": "+fmt(val))}
    else{
     var zs=rowsV.filter(function(r){return r[1]==l[0]&&r[3]=="zero"})[0];
     var pts=(zs?[[zx,zs[6]]]:[]).concat(rs.map(function(r){return [r[4],r[6]]}));
     h+='<polyline points="'+pts.map(function(p){return sx(p[0])+","+sy(p[1])}).join(" ")+'" fill=none stroke="'+c+'" stroke-width=1.8'+dash+' />';
     rs.forEach(function(r){h+=mark(mk,sx(r[4]),sy(r[6]),c,l[1]+", "+lab+vtag+" n"+r[4]+" "+r[5]+": "+fmt(r[6]));
      if(s==RUNK[0]&&lv.length==1&&vs.length==1)h+='<text x="'+(sx(r[4])+6)+'" y="'+(sy(r[6])-6)+'" class=ax>'+fmt(r[6])+'</text>'});
     if(RUNK.indexOf(s)>=0){var last=rs[rs.length-1];
      sum.push((lv.length>1?l[1]+": ":"")+lab+vtag+" n"+last[4]+": "+fmt(last[6])+(zs?" vs zero-shot "+fmt(zs[6])+", "+((last[6]/zs[6]-1)*100>=0?"+":"")+((last[6]/zs[6]-1)*100).toFixed(0)+"%":""))}}})})});
 h+='</svg>';
 var leg=SERS.filter(function(s){return SEL().indexOf(s[0])>=0}).map(function(s){return '<span><i style="background:'+P.col[s[0]]+'"></i>'+s[1]+(s[0]=="zero"?" (dashed)":s[0]=="complete"?" (dotted)":"")+'</span>'}).join("");
 if(vs.length>1)leg+=vs.map(function(v,i){return '<span><svg width=26 height=8 viewBox="0 0 26 8"><line x1=0 x2=26 y1=4 y2=4 stroke="var(--fg)" stroke-width=2'+(VDASH[i%4]?' stroke-dasharray="'+VDASH[i%4]+'"':"")+' /></svg>'+v+'</span>'}).join("");
 if(lv.length>1)leg+=lv.map(function(l,i){return '<span><svg width=12 height=12 viewBox="0 0 12 12">'+mark(MK[i],6,6,"var(--fg)","")+'</svg>'+l[1]+'</span>'}).join("");
 var note="";
 if(SEL().indexOf("complete")>=0&&!rows.some(function(r){return r[3]=="complete"}))note+="<p class=mut>complete-only pending</p>";
 if(P.pending.length)note+='<p class=mut>Scoring on archie, not yet in the scores: '+P.pending.map(function(p){return p.run+" n"+p.n+(p.hash?" "+p.hash:"")}).join(", ")+'.</p>';
 if(!any)note+="<p class=mut>No rows for this selection.</p>";
 $("splot").innerHTML='<div class=sum><span class=mut>rel. '+MM.toUpperCase()+' vs randomwalknowcast</span> '+(sum.join("; ")||"")+'</div>'+h+'<div class=leg>'+leg+'</div>'+note;
}
window.addEventListener("ftsel",draw);
draw();
})();
"""


def scores_html(rows, local):
    pay = json.dumps(score_payload(rows, local),
                     separators=(",", ":")).replace("</", "<\\/")
    return (f'<div id=sctl class=sctl></div>'
            f'<div class=plots2><div class=card><div id=splot></div></div>'
            f'{heldout_chart(rows)}</div>'
            f'<script type="application/json" id=sdata>{pay}</script>'
            f'<script>{SC_JS}</script>')


# ---------- German nowcasts: mirrored per-checkpoint forecasts ----------
GER_REMOTE = "archie:ckpt-scores/cache/{}/"
QS = ["0.025", "0.1", "0.25", "0.5", "0.75", "0.9", "0.975"]
GER = {}


def gs(ds):
    return GER.setdefault(ds["key"], {"files": {}, "sig": None,
                                      "built": None, "loc_json": {}})


def ger_dir(ds):
    if OVERRIDE["germany"] and ds is DATASETS[0]:
        return Path(OVERRIDE["germany"])
    return CACHE / ds["mirror"]


def job_germany():
    for ds in DATASETS:
        if OVERRIDE["germany"] and ds is DATASETS[0]:
            continue
        d = ger_dir(ds)
        d.mkdir(parents=True, exist_ok=True)
        r = subprocess.run(
            ["rsync", "-a", "-e", "ssh -o ConnectTimeout=10 -o BatchMode=yes",
             GER_REMOTE.format(ds["mirror"]), str(d) + "/"],
            capture_output=True, text=True, timeout=300)
        if r.returncode != 0:
            raise RuntimeError(r.stderr[-200:].strip()
                               or f"rsync {r.returncode}")


JOBS["germany"] = job_germany


def wis_row(q, y):
    """WIS from 7 quantiles (0.025 .. 0.975): 3 intervals plus the median."""
    m = q[3]
    tot = 0.5 * abs(y - m)
    for a, lo, hi in ((0.05, q[0], q[6]), (0.2, q[1], q[5]),
                      (0.5, q[2], q[4])):
        tot += a / 2 * ((hi - lo) + 2 / a * max(lo - y, 0)
                        + 2 / a * max(y - hi, 0))
    return tot / 3.5


def parse_ger_file(path):
    import gzip
    out = {}
    with gzip.open(path, "rt") as f:
        for r in csv.DictReader(f):
            mdl = r["model"]
            special = ("rw" if mdl == "randomwalknowcast"
                       else "complete-only" if mdl.endswith("_complete")
                       else None)
            srcs = ["daily"] if "daily" in mdl else \
                ["weekly"] if "weekly" in mdl else \
                ["daily", "weekly"]
            if special is None and "daily" not in mdl and "weekly" not in mdl:
                special = "hub:" + mdl  # other models, e.g. hub teams
            tgt = (r.get("target") or "").strip()
            q = num(r["value"])
            if r["quantile"] not in QS or q is None:
                continue
            for s in srcs:
                k = (r["location"], special, s, tgt)
                e = out.setdefault(k, {}).setdefault(
                    r["target_end_date"], {"q": {}, "y": None, "v": ""})
                e["q"][r["quantile"]] = q
                y = num(r.get("truth"))
                if y is not None:
                    e["y"], e["v"] = y, r.get("truth_vintage") or ""
    return out


def ger_build(ds):
    """Assemble per-location data from the mirrored files (cached)."""
    base = ger_dir(ds)
    G = gs(ds)
    if not base.is_dir():
        return None
    files = []
    for d in sorted(p for p in base.iterdir() if p.is_dir()):
        for f in d.glob("*.csv.gz"):
            try:
                files.append((d.name, f.stem.replace(".csv", ""), f,
                              f.stat().st_mtime_ns))
            except OSError:
                pass
    sig = tuple((a, b, m) for a, b, _, m in files)
    if sig == G["sig"]:
        return G["built"]
    cache = G["files"]
    series, wis, truth, origins = {}, {}, {}, set()
    targets = set()
    prefix = {}
    for setid, origin, f, mt in files:
        key = str(f)
        if key not in cache or cache[key][0] != mt:
            try:
                cache[key] = (mt, parse_ger_file(f))
            except Exception:
                continue
        origins.add(origin) if setid == "reference" else None
        sid = "zero-shot" if setid == "reference" else norm_hash(setid)[0]
        if setid != "reference" and not hash_run(*norm_hash(setid)):
            continue
        if setid != "reference":
            prefix[sid] = norm_hash(setid)[1]
        for (loc, rw, src, tgt), dates in cache[key][1].items():
            if rw and rw.startswith("hub:") and setid != "reference":
                continue
            targets.add(tgt)
            s = rw or sid
            ds = sorted(dates)
            qm, ts = [], []
            for dte in ds:
                e = dates[dte]
                if all(k in e["q"] for k in QS):
                    qm.append([round(e["q"][k], 1) for k in QS])
                    ts.append(dte)
                    if e["y"] is not None:
                        tv = truth.setdefault(loc, {}).setdefault(tgt, {})
                        if dte not in tv or e["v"] >= tv[dte][0]:
                            tv[dte] = (e["v"], e["y"])
            if not ts:
                continue
            series.setdefault(loc, {}).setdefault(s, {}).setdefault(
                tgt, {}).setdefault(src, {})[origin] = {"t": ts, "q": qm}
            ws = [wis_row(q, dates[t]["y"]) for q, t in zip(qm, ts)
                  if dates[t]["y"] is not None]
            if ws:
                wis.setdefault(loc, {}).setdefault(s, {}).setdefault(
                    tgt, {}).setdefault(src, {})[origin] = round(
                        sum(ws) / len(ws), 2)
    sets = sorted({s for l in series.values() for s in l})
    locs = sorted(series, key=lambda l: (
        0 if l == "DE" else 1 if l.startswith("DE-") else 2, l))
    built = {"prefix": prefix, "series": series, "wis": wis, "truth": truth,
             "sets": sets, "locs": locs, "targets": sorted(targets),
             "origins": sorted(origins)}
    G["sig"], G["built"], G["loc_json"] = sig, built, {}
    return built


def ger_loc_json(ds, loc):
    b = ger_build(ds)
    if not b or loc not in b["series"]:
        return "{}"
    G = gs(ds)
    if loc not in G["loc_json"]:
        truth = {}
        for tgt, tvd in b["truth"].get(loc, {}).items():
            tv = sorted(tvd.items())
            truth[tgt] = {"t": [d for d, _ in tv], "y": [v[1] for _, v in tv]}
        G["loc_json"][loc] = json.dumps({
            "series": b["series"][loc], "wis": b["wis"].get(loc, {}),
            "truth": truth}, separators=(",", ":"))
    return G["loc_json"][loc]


def ger_meta(b, known_n):
    """Label each folder by run, n and hash."""
    csv_n = {}
    for r in read_scores():
        h = norm_hash(r["checkpoint"])[0]
        n = num(r.get("outbreaks"))
        if n:
            csv_n.setdefault(h, int(n))
    meta = []
    for s in b["sets"]:
        if s == "zero-shot":
            meta.append(dict(id=s, label="zero-shot (ragged)", run="ref",
                             n=0))
        elif s == "complete-only":
            meta.append(dict(id=s, label="complete-only", run="ref", n=2))
        elif s == "rw":
            meta.append(dict(id=s, label="randomwalknowcast", run="ref", n=1))
        elif s.startswith("hub:"):
            meta.append(dict(id=s, label=s[4:], run="ref", n=3))
        else:
            cfg = next(c for c in RUNS if c["key"] == hash_run(
                s, b["prefix"].get(s)))
            run = cfg["key"]
            n = known_n.get(s) \
                or cfg["n"].get(s) or csv_n.get(s)
            meta.append(dict(id=s, run=run, n=n or 0, label=(
                f"{cfg['label']} n{n} {s}" if n else f"{cfg['label']} {s}")))
    order = {"ref": 0, **{c["key"]: i + 1 for i, c in enumerate(RUNS)}}
    meta.sort(key=lambda m: (order.get(m["run"], 4), m["n"]))
    fixed = {"zero-shot": REF_COL["zero"], "complete-only": REF_COL["complete"],
             "rw": REF_COL["rw"]}
    for run in [c["key"] for c in RUNS]:
        grp = [m for m in meta if m["run"] == run]
        for i, m in enumerate(grp):
            m["color"] = run_colour(run, i, len(grp))
    for m in meta:
        m.setdefault("color", fixed.get(m["id"], "#6b7280"))
    return meta


# Reusable play/loop helper for any time slider. Usage:
#   timePlayer({id, btn, speed, len: () => n, get: () => i, set: i => ...})
# btn is a button element and speed a <select> of 0.5/1/2. It steps one index
# per second, loops at the end, and sets window.ftPlaying while running so
# the page refresh waits. Returns {play, pause, playing}.
PLAYER_JS = r'''
window.timePlayer=function(o){
 var P={playing:false},timer=null;
 function sync(){o.btn.textContent=P.playing?"Pause":"Play";
  window.ftPlayers=window.ftPlayers||{};window.ftPlayers[o.id]=P.playing;
  window.ftPlaying=Object.keys(window.ftPlayers).some(function(k){return window.ftPlayers[k]})}
 function rate(){return +(o.speed&&o.speed.value)||1}
 function arm(){clearInterval(timer);timer=setInterval(function(){o.set((o.get()+1)%o.len())},1000/rate())}
 P.play=function(){if(P.playing)return;P.playing=true;sync();arm()};
 P.pause=function(){if(!P.playing)return;P.playing=false;clearInterval(timer);sync()};
 o.btn.onclick=function(){P.playing?P.pause():P.play()};
 if(o.speed)o.speed.onchange=function(){if(P.playing)arm()};
 sync();return P};
'''


GER_JS = r'''
(function(){
var D=JSON.parse(document.getElementById("gdata").textContent);
var DSS=D.datasets,RUNS=D.runs;
var $=function(i){return document.getElementById(i)};
var DASH=["","6 3","8 3 2 3","2 3"],MK=["circle","square","triangle","diamond"];
var S={ds:DSS[0].key,locs:["DE"],src:"daily",target:null,oi:null,latest:true,nd:1,step:2,on:null,chk:{}};
var KEY="ftg5";
try{var sv=JSON.parse(sessionStorage.getItem(KEY)||"null");if(sv)Object.keys(sv).forEach(function(k){S[k]=sv[k]})}catch(e){}
function cur(){return DSS.find(function(d){return d.key==S.ds})||DSS[0]}
var C=cur();S.ds=C.key;
function ofRun(k){return C.meta.filter(function(m){return m.run==k}).sort(function(a,b){return a.n-b.n})}
function refs(){return C.meta.filter(function(m){return m.run=="ref"})}
function ORI(){return C.origins}
function TGT(){var a=C.targets||[];return a.indexOf(S.target)>=0?S.target:(a.indexOf("7day")>=0?"7day":a[0]||"")}
function tlab(v){return v=="7day"?"7-day":(v||"all")}
S.locs=(S.locs||[]).filter(function(l){return C.locs.indexOf(l)>=0});if(!S.locs.length)S.locs=[C.locs[0]];
if(S.latest||ORI().indexOf(S.origin)<0){S.oi=ORI().length-1;S.latest=true}else S.oi=ORI().indexOf(S.origin);
function onKey(id){return {"zero-shot":"zero","complete-only":"complete"}[id]||id}
function isOn(k){return ftSel.get().indexOf(k)>=0}
function save(){try{sessionStorage.setItem(KEY,JSON.stringify({ds:S.ds,locs:S.locs,src:S.src,target:S.target,latest:S.latest,origin:ORI()[S.oi],nd:S.nd,step:S.step,chk:S.chk}))}catch(e){}}
function chosen(k){var all=ofRun(k),ids=(S.chk[k]||[]).filter(function(i){return all.some(function(m){return m.id==i})});
 if(!ids.length&&all.length)ids=[all[all.length-1].id];return ids}
function effective(){var o=[];
 RUNS.forEach(function(r){if(isOn(r.key)){var ids=chosen(r.key);
  ids.forEach(function(id,i){o.push({id:id,run:r.key,rank:ids.length-1-i})})}});
 refs().forEach(function(m){if(isOn(onKey(m.id)))o.push({id:m.id,run:"ref",rank:0})});
 return o}
function ms(s){return Date.parse(s+"T00:00:00Z")}
function dates(){var O=ORI(),out=[];
 for(var k=0;k<S.nd;k++){var tg=ms(O[S.oi])-k*S.step*7*864e5,j=-1;
  for(var i=0;i<O.length;i++){if(ms(O[i])<=tg)j=i}
  if(j>=0&&out.indexOf(O[j])<0)out.unshift(O[j])}
 return out}
var cache={};
function load(loc){var pre=C.preload||{};
 if(pre[loc])return Promise.resolve(pre[loc]);
 var k=C.key+"/"+loc;if(cache[k])return Promise.resolve(cache[k]);
 return fetch("/data/"+encodeURIComponent(C.key)+"/"+encodeURIComponent(loc)+".json").then(function(r){return r.json()})
  .then(function(j){cache[k]=j;return j})}
function colour(id){var m=C.meta.find(function(x){return x.id==id});return m?m.color:"#64748b"}
function id2lab(id){var m=C.meta.find(function(x){return x.id==id});return m?m.label:id}
function fmt(v){return v>=1000?Math.round(v).toLocaleString("en-GB"):(+v).toPrecision(3)}
function dashAttr(rank){var d=DASH[rank%4];return d?' stroke-dasharray="'+d+'"':""}
function mark(k,x,y,c,t,r){r=r||3;var s="";
 if(k=="circle")s='<circle cx="'+x+'" cy="'+y+'" r="'+r+'" fill="'+c+'">';
 else if(k=="square")s='<rect x="'+(x-r)+'" y="'+(y-r)+'" width="'+2*r+'" height="'+2*r+'" fill="'+c+'">';
 else if(k=="triangle")s='<polygon points="'+x+','+(y-r-1)+' '+(x+r+1)+','+(y+r)+' '+(x-r-1)+','+(y+r)+'" fill="'+c+'">';
 else s='<polygon points="'+x+','+(y-r-1)+' '+(x+r+1)+','+y+' '+x+','+(y+r+1)+' '+(x-r-1)+','+y+'" fill="'+c+'">';
 return s+'<title>'+t+'</title>'+(k=="circle"?"</circle>":k=="square"?"</rect>":"</polygon>")}
function panel(d,loc,ds,eff,small){
 var W=small?300:640,H=small?200:320,L=small?36:48,R=8,T=small?8:10,B=small?24:34,ymax=1,i;
 var t0=ms(ds[0])-70*864e5,t1=ms(ds[ds.length-1])+35*864e5;
 var tr=d.truth[TGT()]||{t:[],y:[]},pts=[];
 for(i=0;i<tr.t.length;i++){var x=ms(tr.t[i]);if(x>=t0&&x<=t1){pts.push([x,tr.y[i]]);ymax=Math.max(ymax,tr.y[i])}}
 var fs=[];
 eff.forEach(function(e){ds.forEach(function(o,di){var s=((d.series[e.id]||{})[TGT()]||{})[S.src],f=s&&s[o];
  if(f){fs.push({e:e,f:f,di:di,o:o});f.q.forEach(function(q){ymax=Math.max(ymax,q[6])})}})});
 var runBands=eff.some(function(e){return e.run!="ref"&&fs.some(function(z){return z.e===e})});
 ymax*=1.06;
 // y from the data shown, not from zero
 var ylo=Infinity;pts.forEach(function(p){ylo=Math.min(ylo,p[1])});fs.forEach(function(z){z.f.q.forEach(function(q){ylo=Math.min(ylo,q[0])})});
 var ymin=isFinite(ylo)?Math.max(0,ylo-0.15*(ymax-ylo)):0;
 var sx=function(x){return L+(x-t0)/(t1-t0)*(W-L-R)},sy=function(y){return T+(1-(y-ymin)/(ymax-ymin))*(H-T-B)};
 var h='<svg viewBox="0 0 '+W+' '+H+'" width="100%">';
 var step=Math.pow(10,Math.floor(Math.log10(ymax/4)));
 var m=[1,2,2.5,5,10].find(function(k){return k*step>=ymax/(small?3:5)})*step;
 for(var v=Math.ceil(ymin/m)*m;v<=ymax;v+=m)h+='<line x1="'+L+'" x2="'+(W-R)+'" y1="'+sy(v)+'" y2="'+sy(v)+'" class=grid /><text x="'+(L-4)+'" y="'+(sy(v)+3)+'" text-anchor=end class=ax>'+fmt(v)+'</text>';
 var xs=small?28:14;
 for(var x=t0;x<=t1;x+=xs*864e5)h+='<text x="'+sx(x)+'" y="'+(H-B+12)+'" text-anchor=middle class=ax>'+new Date(x).toISOString().slice(5,10)+'</text>';
 ds.forEach(function(o){h+='<line x1="'+sx(ms(o))+'" x2="'+sx(ms(o))+'" y1="'+T+'" y2="'+(H-B)+'" class=mark />'});
 fs.forEach(function(z){var e=z.e,f=z.f,c=colour(e.id);
  var P=function(k){return f.t.map(function(d,j){return sx(ms(d))+","+sy(f.q[j][k])})};
  var latest=eff.filter(function(q){return q.run==e.run}).every(function(q){return q.rank>=e.rank});
  if(e.rank==0&&(e.run!="ref"||!runBands)){var op=ds.length>1?.5:1;
   h+='<polygon points="'+P(0).concat(P(6).reverse()).join(" ")+'" fill="'+c+'" opacity="'+.10*op+'" />';
   h+='<polygon points="'+P(2).concat(P(4).reverse()).join(" ")+'" fill="'+c+'" opacity="'+.22*op+'" />'}
  h+='<polyline points="'+P(3).join(" ")+'" fill=none stroke="'+c+'" stroke-width='+(small?1.5:2)+dashAttr(e.run=="ref"?0:e.rank)+' />';
  f.t.forEach(function(d,j){h+=mark(MK[z.di%4],sx(ms(d)),sy(f.q[j][3]),c,id2lab(e.id)+' (nowcast '+z.o+') '+d+' median '+fmt(f.q[j][3])+' (50%: '+fmt(f.q[j][2])+' to '+fmt(f.q[j][4])+'; 95%: '+fmt(f.q[j][0])+' to '+fmt(f.q[j][6])+')',small?2.2:2.8)})});
 if(pts.length){h+='<polyline points="'+pts.map(function(p){return sx(p[0])+","+sy(p[1])}).join(" ")+'" fill=none stroke="var(--fg)" stroke-width=1.4 />';
  pts.forEach(function(p){h+='<circle cx="'+sx(p[0])+'" cy="'+sy(p[1])+'" r=1.8 fill="var(--fg)"><title>truth '+new Date(p[0]).toISOString().slice(0,10)+': '+fmt(p[1])+'</title></circle>'})}
 return {svg:h+'</svg>',miss:eff.filter(function(e){return !fs.some(function(z){return z.e===e})}).map(function(e){return id2lab(e.id)})};
}
function legend(eff,ds){
 function sw(c,rank){return '<svg width=26 height=8 viewBox="0 0 26 8"><line x1=0 x2=26 y1=4 y2=4 stroke="'+c+'" stroke-width=2'+dashAttr(rank)+' /></svg>'}
 var leg='<div class=lg><b>truth</b> '+sw("var(--fg)",0)+'</div>';
 RUNS.forEach(function(r){var g=eff.filter(function(e){return e.run==r.key});if(!g.length)return;
  leg+='<div class=lg><b style="color:'+r.color+'">'+r.label+'</b> '+g.map(function(e){return sw(colour(e.id),e.rank)+"n"+C.meta.find(function(x){return x.id==e.id}).n}).join(" ")+'</div>'});
 var rf=eff.filter(function(e){return e.run=="ref"});
 if(rf.length)leg+='<div class=lg><b>reference</b> '+rf.map(function(e){return sw(colour(e.id),0)+id2lab(e.id)}).join(" ")+'</div>';
 if(ds.length>1)leg+='<div class=lg><b>nowcast date</b> '+ds.map(function(o,i){return '<svg width=12 height=12 viewBox="0 0 12 12">'+mark(MK[i%4],6,6,"var(--fg)","",3)+'</svg>'+o}).join(" ")+'</div>';
 return '<div class=leg2>'+leg+'</div>'}
function plots(){
 var eff=effective(),ds=dates(),small=S.locs.length>1;
 $("gwrap").className=small?"gwrap wide":"gwrap";
 Promise.all(S.locs.map(load)).then(function(all){
  var miss={},h='<div class=panels style="grid-template-columns:repeat('+(small?"auto-fit,minmax(260px,1fr)":"1,1fr")+')">';
  all.forEach(function(d,i){var p=panel(d,S.locs[i],ds,eff,small);p.miss.forEach(function(m){miss[m]=1});
   h+='<div>'+(small?'<h4>'+S.locs[i]+'</h4>':"")+p.svg+'</div>'});
  h+='</div>'+legend(eff,ds);
  var ms_=Object.keys(miss);
  if(ms_.length)h+='<p class=mut>No forecast for some nowcast dates: '+ms_.join(", ")+'</p>';
  $("gplot").innerHTML=h;
  wisChart(all[0],eff)}).catch(function(){$("gplot").innerHTML='<p class=mut>Could not load data.</p>'})}
function wisChart(d,eff){
 var W_=function(id){return ((d.wis[id]||{})[TGT()]||{})[S.src]};
 var sel=eff.map(function(e){return e.id}).filter(function(id){return W_(id)});
 if(!sel.length){$("gwis").innerHTML='<p class=mut>No WIS available.</p>';return}
 var com=Object.keys(W_(sel[0])).filter(function(o){return sel.every(function(id){return W_(id)[o]!==undefined})});
 if(!com.length){$("gwis").innerHTML='<p class=mut>No nowcast dates common to the selected checkpoints.</p>';return}
 var vals=sel.map(function(id){var a=com.map(function(o){return W_(id)[o]});return a.reduce(function(x,y){return x+y},0)/a.length});
 var W=260,H=12+sel.length*34,mx=Math.max.apply(null,vals)*1.05,h='<svg viewBox="0 0 '+W+' '+H+'" width="100%">';
 function shortLab(id){var m=C.meta.find(function(x){return x.id==id});if(!m)return id;var r=RUNS.find(function(r){return r.key==m.run});return r?r.label+" n"+m.n:m.label}
 sel.forEach(function(id,i){var y=4+i*34,w=vals[i]/mx*(W-48);
  h+='<text x="0" y="'+(y+9)+'" class=ax>'+shortLab(id)+'</text><rect x="0" y="'+(y+13)+'" width="'+w+'" height="14" fill="'+colour(id)+'" /><text x="'+(w+4)+'" y="'+(y+24)+'" class=ax>'+fmt(vals[i])+'</text>'});
 $("gwis").innerHTML=h+'</svg>';
}
function chips(){var h='';
 RUNS.forEach(function(r){if(!isOn(r.key))return;var all=ofRun(r.key),ch=chosen(r.key);
  h+='<span class=lg><b style="color:'+r.color+'">'+r.label+'</b> '+(all.length?all.map(function(m){var on=ch.indexOf(m.id)>=0;return '<button class="tog'+(on?' on':'')+'" data-chip="'+r.key+'|'+m.id+'" title="'+m.id+' ('+m.count+' dates)" style="'+(on?'background:'+r.color+';border-color:'+r.color:'')+'">n'+m.n+'</button>'}).join(""):'<span class=mut>no checkpoints scored yet</span>')+'</span>'});
 $("gck").innerHTML=h}
function locBtns(){
 $("glocs").innerHTML=C.groups.filter(function(g){return g.locs.length}).map(function(g){return '<div><span class=mut>'+g.name+'</span> '+g.locs.map(function(l){return '<button class="tog'+(S.locs.indexOf(l)>=0?' on':'')+'" data-loc="'+l+'">'+l+'</button>'}).join("")+'</div>'}).join("");
 $("gsum").textContent="Locations: "+S.locs.join(", ")}
function draw(){
 $("gdate").textContent=ORI()[S.oi]+(S.latest?" (latest)":"");
 $("gsl").value=S.oi;$("gnd").value=S.nd;$("gstep").value=S.step;$("gsrc").value=S.src;
 $("gtgtl").hidden=(C.targets||[]).length<2;
 $("gtgt").innerHTML=(C.targets||[]).map(function(v){return '<option value="'+v+'"'+(v==TGT()?' selected':'')+'>'+tlab(v)+'</option>'}).join("");
 chips();locBtns();plots();save()}
function build(){
 if(DSS.length>1){$("gdsl").hidden=false;$("gds").innerHTML=DSS.map(function(d){return '<option value="'+d.key+'"'+(d.key==S.ds?' selected':'')+'>'+d.label+'</option>'}).join("");
  $("gds").onchange=function(){S.ds=this.value;C=cur();S.locs=[C.locs[0]];S.oi=ORI().length-1;S.latest=true;$("gsl").max=S.oi;draw()}}
 $("gsl").min=0;$("gsl").max=ORI().length-1;
 $("gsrc").onchange=function(){S.src=this.value;draw()};
 $("gtgt").onchange=function(){S.target=this.value;draw()};
 $("gnd").onchange=function(){S.nd=+this.value;draw()};
 $("gstep").onchange=function(){S.step=+this.value;draw()};
 var pl=timePlayer({id:"german",btn:$("gplay"),speed:$("gspd"),len:function(){return ORI().length},
  get:function(){return S.oi},set:function(i){S.oi=i;S.latest=i==ORI().length-1;draw()}});
 $("gsl").oninput=function(){pl.pause();S.oi=+this.value;S.latest=S.oi==ORI().length-1;draw()};
 $("gprev").onclick=function(){S.oi=Math.max(0,S.oi-1);S.latest=false;draw()};
 $("gnext").onclick=function(){S.oi=Math.min(ORI().length-1,S.oi+1);S.latest=S.oi==ORI().length-1;draw()};
 $("glocs").onclick=function(e){var b=e.target.closest("button");if(!b)return;var l=b.dataset.loc,i=S.locs.indexOf(l);
  if(i>=0){if(S.locs.length>1)S.locs.splice(i,1)}else S.locs.push(l);draw()};
 $("gck").onclick=function(e){var b=e.target.closest("button");if(!b)return;var p=b.dataset.chip.split("|"),k=p[0],id=p[1];
  var ids=chosen(k).slice(),i=ids.indexOf(id);if(i>=0){if(ids.length>1)ids.splice(i,1)}else ids.push(id);S.chk[k]=ids;draw()};
 draw()}
window.addEventListener("ftsel",function(){draw()});
build();
})();
'''


def germany_html(known_n, once):
    dss = []
    for ds in DATASETS:
        b = ger_build(ds)
        if not b or not b["origins"]:
            continue
        meta = ger_meta(b, known_n)
        for m in meta:
            m["count"] = max((len(src) for src in b["series"].get(
                "DE", {}).get(m["id"], {}).values()), default=0)
        pre = {l: json.loads(ger_loc_json(ds, l)) for l in b["locs"]} \
            if once else {}
        dss.append({
            "key": ds["key"], "label": ds["label"], "truth": ds["truth"],
            "meta": meta, "locs": b["locs"], "origins": b["origins"],
            "targets": b["targets"],
            "preload": pre,
            "groups": [{"name": n, "locs": [l for l in b["locs"]
                                            if re.match(rx, l)]}
                       for n, rx in ds["groups"]]})
    if not dss:
        return note("germany", "German forecasts") or \
            "<p class=mut>No German forecasts mirrored yet.</p>"
    payload = json.dumps(
        {"datasets": dss,
         "runs": [{"key": c["key"], "label": c["label"],
                   "color": run_colour(c["key"])} for c in RUNS]},
        separators=(",", ":")).replace("</", "<\\/")
    return (note("germany", "German forecasts") + f"""
<div class=gctl>
<label id=gdsl hidden>Dataset <select id=gds></select></label>
<label id=gtgtl hidden>Target <select id=gtgt></select></label>
<label>Source <select id=gsrc><option value=daily>daily</option>
<option value=weekly>weekly</option></select></label>
<label class=gwide>Nowcast date <button id=gprev>&lt;</button>
<input type=range id=gsl><button id=gnext>&gt;</button>
<b id=gdate></b></label>
<label class=gwide><button id=gplay>Play</button>
<select id=gspd><option value=0.5>0.5x</option>
<option value=1 selected>1x</option><option value=2>2x</option></select>
</label>
<label>Dates overlaid <select id=gnd><option>1</option><option>2</option>
<option>3</option><option>4</option></select></label>
<label>every <select id=gstep><option value=1>1 week</option>
<option value=2>2 weeks</option><option value=4>4 weeks</option></select>
</label></div>
<details class=glocs><summary id=gsum>Locations</summary>
<div id=glocs></div></details>
<div id=gck class=gck></div>
<div class=gwrap id=gwrap><div class=card><div id=gplot></div></div>
<div class=card><h3>mean WIS</h3><div id=gwis></div></div></div>
<script type="application/json" id=gdata>{payload}</script>
<script>{GER_JS}</script>""")


# Shared run and reference selection, one set of toggles for every panel.
SEL_JS = r"""
(function(){var D=null,on=[];
 function render(){var el=document.getElementById("selbar");if(!el||!D)return;
  function b(x){return '<button class="tog'+(on.indexOf(x.key)>=0?' on':'')+'" data-k="'+x.key+'" style="border-left:4px solid '+x.color+'">'+x.label+'</button>'}
  el.innerHTML='<span class=mut>runs</span>'+D.runs.map(b).join("")+'<span class=sep></span><span class=mut>references</span>'+D.refs.map(b).join("");
  el.onclick=function(e){var bt=e.target.closest("button");if(!bt)return;window.ftSel.toggle(bt.dataset.k)}}
 function save(){try{sessionStorage.setItem("ftsel",JSON.stringify(on))}catch(e){}}
 window.ftSel={
  init:function(d){D=d;var s=null;try{s=JSON.parse(sessionStorage.getItem("ftsel")||"null")}catch(e){}
   var all=d.runs.concat(d.refs).map(function(x){return x.key});
   on=(Array.isArray(s)?s:d.defaults.slice()).filter(function(k){return all.indexOf(k)>=0});
   if(!on.length)on=d.defaults.slice();render()},
  get:function(){return on.slice()},
  toggle:function(k){var i=on.indexOf(k);if(i>=0){if(on.length>1)on.splice(i,1)}else on.push(k);
   save();render();window.dispatchEvent(new Event("ftsel"))}}})();
"""

NAV_JS = r"""
(function(){
 [].forEach.call(document.querySelectorAll("details.sec"),function(d){
  try{var v=sessionStorage.getItem("ftsec-"+d.id);if(v!==null)d.open=v=="1"}catch(e){}
  d.addEventListener("toggle",function(){try{sessionStorage.setItem("ftsec-"+d.id,d.open?"1":"0")}catch(e){}})});
 [].forEach.call(document.querySelectorAll(".nav a"),function(a){
  a.addEventListener("click",function(){var d=document.querySelector(a.getAttribute("href"));if(d&&d.tagName=="DETAILS")d.open=true})});
})();
"""

# Reload every 60 s, but wait while any time slider is playing.
MODE_JS = r"""
(function(){var m="start",r="zoom";
 try{m=sessionStorage.getItem("ftmode")||m;r=sessionStorage.getItem("ftrng")||r}catch(e){}
 function paint(){var c=document.body.className.replace(/\b[mr]-\w+/g,"").trim();
  document.body.className=c+" m-"+m+" r-"+r;
  [].forEach.call(document.querySelectorAll(".modebar button"),function(b){
   b.classList.toggle("on",b.dataset.mode==m||b.dataset.rng==r)});
  try{sessionStorage.setItem("ftmode",m);sessionStorage.setItem("ftrng",r)}catch(e){}}
 [].forEach.call(document.querySelectorAll(".modebar"),function(bar){
  bar.onclick=function(e){var b=e.target.closest("button");if(!b)return;
   if(b.dataset.mode)m=b.dataset.mode;if(b.dataset.rng)r=b.dataset.rng;paint()}});
 paint()})();
"""

REFRESH_JS = r"""
(function tick(){setTimeout(function(){
 if(window.ftPlaying)tick();else location.reload()},60000)})();
"""

ONCE = {"on": False}

# ---------- attention bias module (self, same, up, down; [layers, heads]) ----------
# The series bias holds one [layers, heads] tensor per relation (the trainer's
# series_bias.PAIRS): self, same place, child to parent, parent to child.
BIAS_PAIRS = [("self", "self"), ("same", "same"),
              ("up", "up (child to parent)"),
              ("down", "down (parent to child)")]
BIAS_COL = {"self": "#6b7280", "same": "#2563eb", "up": "#16a34a",
            "down": "#ea580c"}
BIAS_RE = re.compile(r"(?:^|[._/])(self_?|same|up|down)$")
TORCH_SCRIPT = r"""
import json, sys, torch
sd = torch.load(sys.argv[1], map_location="cpu", weights_only=True)
out = {}
def walk(o, p):
    if isinstance(o, dict):
        for k, v in o.items():
            walk(v, f"{p}.{k}" if p else str(k))
    elif hasattr(o, "shape") and len(o.shape) == 2:
        out[p] = o.float().tolist()
walk(sd, "")
print(json.dumps(out))
"""
BIAS = {"cache": {}}


def st_tensors(path):
    """2-D float tensors of a safetensors file as {name: rows} (stdlib)."""
    import array
    import struct
    out = {}
    with open(path, "rb") as f:
        n = int.from_bytes(f.read(8), "little")
        hdr = json.loads(f.read(n))
        for name, info in hdr.items():
            if name == "__metadata__" or len(info.get("shape", [])) != 2 \
                    or not BIAS_RE.search(name):
                continue
            a, b = info["data_offsets"]
            f.seek(8 + n + a)
            raw = f.read(b - a)
            dt = info["dtype"]
            if dt == "F32":
                v = array.array("f", raw)
            elif dt == "F64":
                v = array.array("d", raw)
            elif dt == "F16":
                v = struct.unpack(f"<{len(raw)//2}e", raw)
            elif dt == "BF16":
                v = [struct.unpack("<f", b"\0\0" + raw[i:i + 2])[0]
                     for i in range(0, len(raw), 2)]
            else:
                continue
            r, c = info["shape"]
            out[name] = [list(map(float, v[i * c:(i + 1) * c]))
                         for i in range(r)]
    return out


def pt_tensors(path):
    """Same for a torch file, via the repo venv's python (cached)."""
    py = ROOT / ".venv/bin/python"
    if not py.exists():
        raise RuntimeError("no .venv python to read torch files")
    r = subprocess.run([str(py), "-c", TORCH_SCRIPT, str(path)],
                       capture_output=True, text=True, timeout=180)
    if r.returncode != 0:
        raise RuntimeError(r.stderr[-150:])
    return {k: v for k, v in json.loads(r.stdout.splitlines()[-1]).items()
            if BIAS_RE.search(k)}


def bias_file(path):
    """{'same': rows, 'parent': rows} from one adapter file, or None."""
    try:
        st = Path(path).stat()
    except OSError:
        return None
    key = (str(path), st.st_mtime_ns)
    if key not in BIAS["cache"]:
        try:
            ts = pt_tensors(path) if str(path).endswith((".pt", ".bin")) \
                else st_tensors(path)
        except Exception:
            ts = {}
        res = {}
        for k, _ in BIAS_PAIRS:
            c = [n for n in ts if BIAS_RE.search(n).group(1).rstrip("_") == k
                 and "bias" in n.lower()]
            c.sort(key=lambda n: ("modules_to_save" not in n,
                                  "original" in n, n))
            if c:
                res[k] = ts[c[0]]
        BIAS["cache"][key] = res or None
    return BIAS["cache"][key]


def bias_stats(rows):
    flat = [x for r in rows for x in r]
    return {"mean": sum(flat) / len(flat), "min": min(flat), "max": max(flat)}


def bias_data(rundir, hist, cur_step):
    """Per-step stats and the latest [layers, heads] matrices of a run."""
    pts = {k: {} for k, _ in BIAS_PAIRS}
    latest = None
    # The trainer logs history entries {"step", "bias": {pair: {mean, min,
    # max, values[layers][heads]}}} at each eval.
    for h in hist if isinstance(hist, list) else []:
        if not isinstance(h, dict) or "step" not in h or \
                not isinstance(h.get("bias"), dict):
            continue
        for key, v in h["bias"].items():
            k = key.rstrip("_")
            if k not in pts:
                continue
            vals = v.get("values") if isinstance(v, dict) else v
            if isinstance(vals, list) and vals and isinstance(vals[0], list):
                pts[k][h["step"]] = (
                    {x: v[x] for x in ("mean", "min", "max")}
                    if isinstance(v, dict) and {"mean", "min", "max"} <= set(v)
                    else bias_stats(vals))
                if not latest or h["step"] > latest["step"]:
                    latest = {"step": h["step"]}
                if latest["step"] == h["step"]:
                    latest[k] = vals
    files = []
    for d in Path(rundir).glob("train/n*"):
        s = JC.get(str(d / "settings.json"))
        step = (s or {}).get("steps")
        if step is None:
            continue
        for f in sorted((d / "finetuned-ckpt").glob("*.safetensors")) + \
                sorted((d / "finetuned-ckpt").glob("*.pt")):
            files.append((step, f))
    st = Path(rundir) / "train/state"
    for f in ([st / "adapter.pt"] if (st / "adapter.pt").is_file() else
              sorted(st.glob("adapter*.safetensors"))):
        files.append((cur_step if cur_step is not None else 10 ** 9, f))
    for step, f in sorted(files, key=lambda x: x[0]):
        b = bias_file(f)
        if not b:
            continue
        for k, rows in b.items():
            pts[k].setdefault(step, bias_stats(rows))
        if not latest or step >= latest.get("step", -1):
            latest = {"step": step, **b}
    if not any(pts.values()) and not latest:
        return None
    return {"pts": {k: sorted(v.items()) for k, v in pts.items()},
            "latest": latest}


def heat_svg(rows, title, lo, hi, size=140):
    """One layer-by-head tile on a colour scale shared across tiles."""
    div = lo < 0 < hi
    amp = max(abs(lo), abs(hi)) or 1
    nr, nc = len(rows), len(rows[0])
    cw = size / max(nr, nc)
    h = [f'<svg viewBox="0 0 {size} {size}" width="{size}" height="{size}">']
    for i, r in enumerate(rows):
        for j, v in enumerate(r):
            if div:
                a = abs(v) / amp
                col = (f"rgba(220,38,38,{a:.2f})" if v > 0
                       else f"rgba(37,99,235,{a:.2f})")
            else:
                a = (v - lo) / ((hi - lo) or 1)
                col = f"rgba(37,99,235,{0.08 + 0.92 * a:.2f})"
            h.append(f'<rect x="{j * cw:.1f}" y="{i * cw:.1f}" '
                     f'width="{cw - 0.5:.1f}" height="{cw - 0.5:.1f}" '
                     f'fill="{col}"><title>{title} layer {i} head {j}: '
                     f'{v:.4g}</title></rect>')
    h.append("</svg>")
    return "".join(h)


def bias_html(rundir, label, hist, cur_step, lr_h=""):
    d = bias_data(rundir, hist, cur_step)
    if not d:
        return (f"<p class=mut>no bias module in the {html.escape(label)} "
                f"run</p>" + (f'<div class=biasrow>{lr_h}</div>' if lr_h
                              else ""))
    W, H, L, R, T, B = 480, 280, 44, 10, 8, 28
    names = dict(BIAS_PAIRS)
    allp = [(s, v) for k in d["pts"] for s, v in d["pts"][k]]
    svg = []
    if allp:
        xs = [s for s, _ in allp]
        x0, x1 = min(xs), max(max(xs), min(xs) + 1)
        ys = [v["mean"] for _, v in allp]  # bands may run past the axes
        y0, y1 = min(ys), max(ys)
        pad = (y1 - y0) * 0.25 or 0.01
        y0, y1 = y0 - pad, y1 + pad
        sx = lambda x: L + (x - x0) / (x1 - x0) * (W - L - R)
        sy = lambda y: T + (1 - (y - y0) / (y1 - y0)) * (H - T - B)
        svg.append(f'<svg viewBox="0 0 {W} {H}" width="{W}" height="{H}">')
        for tk in nice(y0, y1):
            if y0 <= tk <= y1:
                svg.append(f'<line x1="{L}" x2="{W-R}" y1="{sy(tk):.1f}" '
                           f'y2="{sy(tk):.1f}" class=grid /><text x="{L-5}" '
                           f'y="{sy(tk)+3:.1f}" text-anchor=end class=ax>'
                           f'{tk:.3g}</text>')
        for tk in nice(x0, x1):
            if x0 <= tk <= x1:
                svg.append(f'<text x="{sx(tk):.1f}" y="{H-B+13}" '
                           f'text-anchor=middle class=ax>{tk:g}</text>')
        svg.append(f'<text x="{(L+W-R)/2}" y="{H-3}" text-anchor=middle '
                   f'class=ax>step (hover a line for its min-max band)</text>')
        if y0 < 0 < y1:
            svg.append(f'<line x1="{L}" x2="{W-R}" y1="{sy(0):.1f}" '
                       f'y2="{sy(0):.1f}" class=mark />')
        for n, (k, pts) in enumerate(d["pts"].items()):
            if not pts:
                continue
            c = BIAS_COL[k]
            up = " ".join(f"{sx(s):.1f},{sy(v['max']):.1f}" for s, v in pts)
            dn = " ".join(f"{sx(s):.1f},{sy(v['min']):.1f}"
                          for s, v in reversed(pts))
            mean = " ".join(f"{sx(s):.1f},{sy(v['mean']):.1f}"
                            for s, v in pts)
            svg.append(
                f'<g class=rel><polygon class=band points="{up} {dn}" '
                f'fill="{c}" /><polyline points="{mean}" fill=none '
                f'stroke="{c}" stroke-width=2 /><polyline points="{mean}" '
                f'fill=none stroke="transparent" stroke-width=12 />'
                + "".join(
                    f'<circle cx="{sx(s):.1f}" cy="{sy(v["mean"]):.1f}" r=2.6 '
                    f'fill="{c}"><title>{names[k]} step {s}: mean '
                    f'{v["mean"]:.4g}, range {v["min"]:.4g} to '
                    f'{v["max"]:.4g}</title></circle>' for s, v in pts)
                + f'<text x="{L+6}" y="{T+12+n*12}" class=ax fill="{c}" '
                f'style="fill:{c}">{names[k]}</text></g>')
        svg.append("</svg>")
    lt = d["latest"] or {}
    keys = [k for k, _ in BIAS_PAIRS if k in lt]
    flat = [x for k in keys for r in lt[k] for x in r]
    lo, hi = (min(flat), max(flat)) if flat else (0, 1)
    tiles = "".join(
        f'<div class=tile4><div class=mut>{k} (step {lt.get("step")})</div>'
        f'{heat_svg(lt[k], k, lo, hi)}</div>' for k in keys)
    right = (f'<div><div class=tiles4>{tiles}</div><div class=mut>colour '
             f'scale {lo:.3g} to {hi:.3g}, shared; rows are layers, columns '
             f'are heads</div></div>' if tiles else "")
    return (f'<div class=biasrow><div class=card>{"".join(svg)}</div>'
            f'{right}{lr_h}</div>')


# ---------- page ----------
def fmt(v, n=4):
    return "" if v is None else (f"{v:.{n}f}" if isinstance(v, float) else
                                 html.escape(str(v)))


def dur(s):
    if s is None or s != s or s < 0:
        return "n/a"
    s = int(s)
    return f"{s//3600}h {s%3600//60:02d}m"


def build():
    now = time.time()
    LC = live_cfg()
    RUN = live_dir()
    logs = {c["key"]: run_log(c) for c in RUNS}
    sl = logs[LC["key"]]
    status = JC.get(str(RUN / "train/stream-status.json")) or {}
    hist = JC.get(str(RUN / "train/history.json")) or []
    step = sl[-1]["step"] if sl else None
    last_t = sl[-1]["t"] if sl else None
    rate = None
    recent = [r for r in sl if r["t"] >= (last_t or 0) - 1800]
    if len(recent) >= 2 and recent[-1]["t"] > recent[0]["t"]:
        rate = (recent[-1]["step"] - recent[0]["step"]) / (
            recent[-1]["t"] - recent[0]["t"])
    P = run_params(RUN, hist, step)
    prefixes = P["prefixes"]
    target = max(prefixes)
    # steps per simulation: the learning-rate schedule spans the final
    # prefix; without one, the last checkpoint's own ratio.
    sps = (P["lr_steps"] / target if P["lr_steps"] else next(
        (s["steps"] / s["n_outbreaks"] for _, s, _ in reversed(
            checkpoints(RUN)) if s.get("steps") and s.get("n_outbreaks")),
        1.0))
    saved = {n for n in prefixes if (RUN / f"train/n{n}/settings.json").exists()}
    scoring = [n for n in prefixes if n not in saved
               and (RUN / f"train/n{n}/finetuned-ckpt").exists()]
    nxt = next((n for n in prefixes if n not in saved and n not in scoring),
               None)
    tstep = lambda n: round(n * sps)
    eta = lambda n: ("n/a" if not (rate and step is not None and n) else
                     "due now" if tstep(n) <= step else
                     dur((tstep(n) - step) / rate))
    ntrained = step / sps if step is not None else None
    pidtxt, alive = pid_state(RUN)
    age = now - last_t if last_t else None
    stale = age is not None and age > 600
    cls = lambda ok: "ok" if ok else "bad"
    next_txt = (f"n{scoring[0]}: scoring since " + datetime.fromtimestamp(
        (RUN / f"train/n{scoring[0]}").stat().st_mtime).strftime("%H:%M")
        if scoring else f"n{nxt}" if nxt else "none")
    next_sub = ((f"then n{nxt} at step {tstep(nxt)}, ETA {eta(nxt)}" if nxt
                 else "") if scoring else
                f"step {tstep(nxt)}, ETA {eta(nxt)}" if nxt else "")
    tiles = [
        ("Steps / s (30 min)", f"{rate:.3f}" if rate else "n/a",
         f"{rate*60:.1f} steps/min" if rate else ""),
        (f"ETA n{target}", eta(target), f"step {tstep(target)}"),
        ("Learning rates",
         "\n".join(f"{g} {v:.2e}" for g, v in P["current"].items())
         or "n/a",
         (f"linear decay to 0 at step {P['lr_steps']}" if P["lr_steps"]
          else "constant") + (", from history" if P["logged"] else "")),
        ("Last log line",
         datetime.fromtimestamp(last_t).strftime("%H:%M:%S") if last_t
         else "n/a",
         ("STALE " if stale else "") + (dur(age) + " ago" if age is not None
                                        else "") if age is not None else ""),
    ]
    tiles_h = "".join(
        f'<div class="tile{" bad" if k == "Last log line" and stale else ""}">'
        f'<div class=k>{k}</div><div class="v{" vl" if "\n" in v else ""}">'
        f'{html.escape(v).replace(chr(10), "<br>")}</div>'
        f'<div class=s>{html.escape(s)}</div></div>'
        for k, v, s in tiles)
    bar_tiles = (
        f'<span class=bt><span class=k>step</span> <b>{step if step is not None else "n/a"}</b>'
        f' <span class=mut>n{ntrained:.0f}/{target}</span></span>'
        if ntrained is not None else "") + (
        f'<span class=bt><span class=k>next</span> <b>{html.escape(next_txt)}</b>'
        f' <span class=mut>{html.escape(next_sub)}</span></span>'
        f'<span class="bt {"ok" if alive else "bad" if alive is False else ""}">'
        f'<span class=k>trainer</span> <b>{html.escape(pidtxt)}</b></span>')
    stage = status.get("stage")
    sub = (f"stage: {html.escape(str(stage))}; stream status "
           f"{html.escape(str(status.get('time', '')))}" if status else "")

    cks = {c["key"]: checkpoints(Path(c["dir"])) for c in RUNS}
    marks = []
    for name, s, _ in cks[LC["key"]]:
        n = int(re.sub(r"\D", "", name) or 0)
        marks.append((s.get("steps") or tstep(n), name))
    marks += [(tstep(n), f"n{n}?") for n in scoring]
    train = [(h["step"], h["loss"]) for h in hist
             if isinstance(h, dict) and "loss" in h and "step" in h]
    if not train:
        train = [(r["step"], r["loss"]) for r in sl]
    smax = max([x for x, _ in train] + [1000])
    SRC_COL = {"weekly": PAL[0], "daily": PAL[1], "3day": PAL[2]}
    styles = [None, "dash", "dd", "dot"]
    info = {}
    for c in RUNS:
        d = Path(c["dir"])
        hh = JC.get(str(d / "train/history.json")) or []
        h0 = next((h for h in hh if isinstance(h, dict)
                   and h.get("step") == 0 and "held_out_loss" in h), {})
        ck = cks[c["key"]]
        full0 = h0.get("held_out_loss") or next(
            (s.get("held_out_loss_start") for _, s, _ in ck
             if s.get("held_out_loss_start")), None)
        if c is LC:
            tr = train
        else:
            tr = [(h["step"], h["loss"]) for h in hh if isinstance(h, dict)
                  and "loss" in h and "step" in h] or [
                (r["step"], r["loss"]) for r in logs[c["key"]]]
        info[c["key"]] = {"full0": full0, "src0": h0.get("by_source") or {},
                          "train": tr, "ck": ck,
                          "tr0": (sum(v for _, v in tr[:20]) / len(tr[:20])
                                  if tr else None)}
    for c in RUNS:  # base model: own step-0 unless the run starts with labels
        b = info.get(c.get("base_from") or c["key"], info[c["key"]])
        info[c["key"]]["bfull"], info[c["key"]]["bsrc"] = b["full0"], b["src0"]

    def make_plots(mode, zoom):
        rel = mode != "raw"
        suffix = {"raw": "", "start": ", relative to start",
                  "base": ", relative to base Chronos-2"}[mode]
        ref = (1.0,) if rel else ()
        tr_series, held, bysrc = [], [], []
        for i, c in enumerate(RUNS):
            k, col, I = c["key"], run_colour(c["key"]), info[c["key"]]
            rows = logs[k]
            tr = [q for q in I["train"] if c is LC or q[0] <= smax]
            if rel and I["tr0"]:
                tr = [(x, y / I["tr0"]) for x, y in tr]
            if c is LC:
                tr_series += [(f"{c['label']} raw", col, tr, "dots"),
                              (f"{c['label']} rolling mean", col,
                               rolling(tr, 20), "line")]
            else:
                tr_series.append((f"{c['label']} rolling mean", col,
                                  rolling(tr, 20), "faint"))
            if not rel:
                held.append((c["label"], col, [(r["step"], r["held"])
                                               for r in rows], "pts"))
            else:
                den = I["full0"] if mode == "start" else I["bfull"]
                if den and I["full0"]:
                    pts = [(0, I["full0"] / den)] + [
                        (s.get("steps") or tstep(int(re.sub(
                            r"\D", "", nm) or 0)),
                         s["held_out_loss"] / den)
                        for nm, s, _ in I["ck"] if s.get("held_out_loss")]
                    held.append((c["label"], col, pts, "pts"))
            for src in ("weekly", "daily", "3day"):
                pts = [(r["step"], r["src"][src]) for r in rows]
                if not any(v is not None for _, v in pts):
                    continue
                if rel:
                    den = (I["src0"] if mode == "start" else I["bsrc"]).get(src)
                    if not den:
                        continue
                    z = I["src0"].get(src)
                    pts = ([(0, z)] if z else []) + pts
                    pts = [(x, v / den) for x, v in pts if v is not None]
                s = (f"{src} ({c['label']})", SRC_COL[src], pts, "pts")
                bysrc.append(s + (styles[i % 4],) if styles[i % 4] else s)
        # Logged held-out losses are over the first 100 blocks. The pooled
        # relative charts use the checkpoints' full-set losses instead,
        # because the step-0 first-100 pooled loss is not logged.
        ttl = {"raw": "Held-out pooled loss by run (first 100 blocks)",
               "start": "Held-out pooled loss by run (checkpoints, full set)",
               "base": "Held-out pooled loss by run (checkpoints, full set)"}
        return [
            chart("Training loss (rolling mean of 20 logged points)"
                  + suffix, tr_series, marks, ylog=True, hlines=ref,
                  xmax=xlim if zoom else None, W=440, H=280),
            chart(ttl[mode] + suffix, held, marks, ylog=True, hlines=ref,
                  xmax=xlim if zoom else None, W=440, H=280),
            chart("Held-out loss by source (colour is the source, style the "
                  "run)" + suffix, bysrc, marks, ylog=True, hlines=ref,
                  xmax=xlim if zoom else None, W=440, H=280)]

    # x range: the live run's steps (to 1.2x the latest, at least the next
    # checkpoint), unless "full range" is chosen.
    next_ck = next((n for n in prefixes if n not in saved), None)
    xlim = max([1.2 * (step or 0), 300] + (
        [tstep(next_ck)] if next_ck else []))
    plots_h = "".join(
        f'<div class="mode m-{m} rng r-{z}"><div class=plots3>'
        + "".join(make_plots(m, z == "zoom")) + "</div></div>"
        for m in ("raw", "start", "base") for z in ("zoom", "full"))

    def cktable(rows, run, key, mode):
        if not rows:
            return f"<p class=mut>no checkpoints yet for {run}</p>"
        I = info[key]
        hd = {"raw": "held-out", "start": "held-out / start",
              "base": "held-out / base"}[mode]
        o = [f"<table><tr><th>run<th>ckpt<th>hash<th>{hd}<th>held-out "
             "start<th>real-val<th>weekly<th>daily<th>3day<th>saved</tr>"]
        for name, s, mt in rows:
            bs = s.get("held_out_loss_by_source") or {}
            ho = s.get("held_out_loss")
            den = {"raw": 1, "start": I["full0"], "base": I["bfull"]}[mode]
            hv = ho / den if ho and den else None
            o.append("<tr>" + "".join(f"<td>{c}" for c in [
                run, name, html.escape(str(s.get("hash", ""))),
                fmt(hv) or "n/a", fmt(s.get("held_out_loss_start")),
                fmt(s.get("real_validation_loss")),
                fmt(bs.get("weekly")) or "n/a", fmt(bs.get("daily")) or "n/a",
                fmt(bs.get("3day")) or "n/a",
                datetime.fromtimestamp(mt).strftime("%m-%d %H:%M") if mt
                else ""]) + "</tr>")
        return "".join(o) + "</table>"

    ck_h = "".join(
        f'<div class="mode m-{m}">' + "".join(
            cktable(cks[c["key"]], c["label"], c["key"], m) for c in RUNS)
        + "</div>" for m in ("raw", "start", "base"))
    cmp_h = "<p class=mut>output/finetune/slots-vs-noslots.csv not present</p>"
    if CSV.exists():
        try:
            rows = list(csv.reader(CSV.open()))
            cmp_h = ("<table>" + "".join(
                "<tr>" + "".join(
                    f"<{'th' if i == 0 else 'td'}>{html.escape(c)}"
                    for c in r) + "</tr>" for i, r in enumerate(rows))
                + "</table>")
        except Exception as e:
            cmp_h = f"<p class=mut>could not read csv: {html.escape(str(e))}</p>"
    mh = ""
    if MAILBOX:
        mail_kick()
        if MAIL["ok"] is None:
            mh = "<p class=mut>loading mailbox (refreshes in a minute)</p>"
        elif MAIL["ok"] is False and not MAIL["lines"]:
            mh = "<p class=mut>mailbox unavailable (ssh failed)</p>"
        elif not MAIL["lines"]:
            mh = "<p class=mut>no matching lines</p>"
        else:
            ln = MAIL["lines"]
            cut = lambda s: html.escape(s if len(s) <= 260 else s[:257] + "...")
            mh = ("<pre>" + "\n".join(cut(x) for x in ln[-5:]) + "</pre>"
                  + (f"<details><summary>show more ({len(ln) - 5})</summary>"
                     "<pre>" + html.escape("\n".join(ln[:-5])) + "</pre>"
                     "</details>" if len(ln) > 5 else ""))

    for k in ("scores", "md", "germany"):
        rem_kick(k)
    srows = read_scores()
    local = []
    for run, base in [(c["key"], Path(c["dir"])) for c in RUNS]:
        for d in base.glob("train/n*"):
            n = int(re.sub(r"\D", "", d.name) or 0)
            st = jload(d / "settings.json") if (d / "settings.json").exists() \
                else None
            if n and (st or (d / "finetuned-ckpt").exists()):
                local.append((run, n, str((st or {}).get("hash") or "")))
    sc_h = note("scores", "Scores")
    if srows:
        sc_h += scores_html(srows, local)
    elif not sc_h:
        sc_h = "<p class=mut>no checkpoint scores yet</p>"
    try:
        md_txt = (CACHE / "slots-vs-noslots.md").read_text()
    except Exception:
        md_txt = ""
    md_h = ("<details><summary>slots-vs-noslots.md (archie review)"
            "</summary><pre>" + html.escape(md_txt) + "</pre></details>"
            if md_txt else "")
    rv_h = germany_html(
        {str(s.get("hash")): s.get("n_outbreaks")
         for v in cks.values() for _, s, _ in v}, ONCE["on"])

    bias_h = bias_html(RUN, LC["label"], hist, step, lr_card(P, step))
    modebar = (
        '<div class="sctl modebar"><span class=mut>Loss scale</span>'
        '<button class=tog data-mode=raw>raw</button>'
        '<button class=tog data-mode=start title="divided by the run\'s own step-0 loss">relative to start</button>'
        '<button class=tog data-mode=base title="divided by base Chronos-2 on the same blocks (base_from in RUNS)">relative to base Chronos-2</button>'
        '<span class=sep></span><span class=mut>x range</span>'
        '<button class=tog data-rng=zoom>live run</button>'
        '<button class=tog data-rng=full>full range</button></div>')
    sc_pay = score_payload(srows, local) if srows else {}
    refs = [{"key": "zero", "label": "zero-shot", "color": REF_COL["zero"]},
            {"key": "complete", "label": "complete-only",
             "color": REF_COL["complete"]},
            {"key": "rw", "label": "randomwalknowcast",
             "color": REF_COL["rw"]}]
    for ds in DATASETS:
        gb = ger_build(ds)
        for s in (gb["sets"] if gb else []):
            if s.startswith("hub:") and not any(r["key"] == s for r in refs):
                refs.append({"key": s, "label": s[4:], "color": "#94a3b8"})
    seldata = json.dumps({
        "runs": [{"key": c["key"], "label": c["label"],
                  "color": run_colour(c["key"])} for c in RUNS],
        "refs": refs,
        "defaults": sc_pay.get("defaults") or [LC["key"], "zero", "rw"]},
        separators=(",", ":")).replace("</", "<\\/")
    sections = [("status", "Status"), ("loss", "Loss"),
                ("bias", "Attention bias"), ("performance", "Performance"),
                ("nowcasts", "Nowcasts")] + (
        [("mailbox", "Activity")] if mh else []) + [("tables", "Tables")]
    nav = "".join(f'<a href="#{i}">{n}</a>' for i, n in sections)
    css = """
:root{--bg:#fff;--fg:#1d2127;--mut:#6b7280;--card:#f6f7f9;--line:#d9dde3;
--c1:#2563eb;--c2:#ea580c;--c3:#16a34a;--c4:#9333ea;--bad:#dc2626;--ok:#15803d}
@media(prefers-color-scheme:dark){:root{--bg:#14171c;--fg:#e6e8eb;
--mut:#9aa3af;--card:#1d2128;--line:#343a44;--c1:#60a5fa;--c2:#fb923c;
--c3:#4ade80;--c4:#c084fc;--bad:#f87171;--ok:#4ade80}}
body{background:var(--bg);color:var(--fg);font:14px system-ui,sans-serif;
margin:0;padding:16px;max-width:1300px;margin-inline:auto}
h1{font-size:18px;margin:0 0 4px}h2{font-size:15px;margin:12px 0 6px}
h3{font-size:13px;margin:0 0 6px}.mut{color:var(--mut)}
.tiles{display:grid;grid-template-columns:repeat(auto-fill,minmax(170px,1fr));
gap:8px}.tile,.card{background:var(--card);border-radius:8px;padding:10px}
.vl{font-size:14px!important;line-height:1.3}.k{font-size:11px;color:var(--mut)}.v{font-size:18px;font-weight:600;
font-variant-numeric:tabular-nums}.s{font-size:11px;color:var(--mut)}
.tile.ok .v{color:var(--ok)}.tile.bad .v{color:var(--bad)}
.plots{display:grid;grid-template-columns:repeat(auto-fill,minmax(min(100%,440px),1fr));
gap:8px}svg{display:block}.grid{stroke:var(--line);stroke-width:1}
.mark{stroke:var(--mut);stroke-dasharray:3 3;stroke-width:1}
.ax{fill:var(--mut);font-size:10px}.leg{font-size:12px;display:flex;gap:12px;
flex-wrap:wrap}.leg i{display:inline-block;width:10px;height:10px;
border-radius:2px;margin-right:4px}
table{border-collapse:collapse;width:100%;margin-bottom:10px;
font-variant-numeric:tabular-nums}th,td{padding:3px 8px;text-align:left;
border-bottom:1px solid var(--line);font-size:12px}
.gctl{display:flex;flex-wrap:wrap;gap:12px;align-items:center;
margin-bottom:8px}.gwide input{width:200px;vertical-align:middle}
.gck{display:flex;flex-wrap:wrap;gap:6px 16px;margin-bottom:8px}
.mode{display:none}body.m-raw .m-raw:not(.rng),body.m-start .m-start:not(.rng),
body.m-base .m-base:not(.rng){display:block}
body.m-raw.r-zoom .m-raw.r-zoom,body.m-raw.r-full .m-raw.r-full,
body.m-start.r-zoom .m-start.r-zoom,body.m-start.r-full .m-start.r-full,
body.m-base.r-zoom .m-base.r-zoom,body.m-base.r-full .m-base.r-full{display:block}.sctl{display:flex;flex-wrap:wrap;gap:4px;align-items:center;
margin-bottom:8px}.sep{width:10px}.sum{font-size:13px;font-weight:600;
min-height:1.3em;margin-bottom:4px}.plots2{display:grid;
grid-template-columns:repeat(auto-fill,minmax(min(100%,440px),1fr));gap:8px;
align-items:start}.tog{border:1px solid var(--line);background:var(--card);
color:var(--fg);border-radius:6px;padding:2px 8px;cursor:pointer;
font-size:12px}
.tog.on{background:var(--c1);color:#fff;border-color:var(--c1)}
.sctl .mut{margin-left:10px}.lg{display:inline-flex;align-items:center;gap:6px;margin-right:14px}
.leg2{font-size:12px;display:flex;flex-wrap:wrap;gap:4px 18px}.glocs{margin:6px 0}.glocs .tog{margin:1px}.panels{display:grid;gap:8px}
.panels h4{margin:0 0 2px;font-size:12px}.plots3{display:grid;grid-template-columns:repeat(auto-fit,minmax(min(100%,290px),1fr));gap:8px}
.plots3 .leg{font-size:11px;gap:2px 8px}
.biasrow{display:flex;flex-wrap:wrap;gap:8px;align-items:flex-start}
.topbar{position:sticky;top:0;z-index:20;background:var(--bg);padding:6px 0 4px;
border-bottom:1px solid var(--line);margin-bottom:6px}
.navrow{display:flex;flex-wrap:wrap;gap:4px 16px;align-items:center}
.nav{display:flex;gap:10px}.nav a{color:var(--c1);text-decoration:none;font-size:13px}
.bt{font-size:13px}.bt.ok b{color:var(--ok)}.bt.bad b{color:var(--bad)}
.selbar{display:flex;flex-wrap:wrap;gap:4px;align-items:center;margin-top:4px}
.selbar .mut{font-size:11px;margin-left:6px}
details.sec{margin:0 0 6px}details.sec>summary{cursor:pointer;list-style:none;
display:flex;align-items:center}details.sec>summary::-webkit-details-marker{display:none}
details.sec>summary::before{content:"▸";margin-right:6px;color:var(--mut)}
details.sec[open]>summary::before{content:"▾"}
details.sec{scroll-margin-top:92px}details.sec h2{margin:10px 0 6px}
.gwrap{display:grid;grid-template-columns:minmax(0,720px) 260px;gap:8px;align-items:start}
.gwrap.wide{grid-template-columns:minmax(0,1fr) 260px}
@media(max-width:900px){.gwrap{grid-template-columns:1fr}}
@media(max-width:1250px){.tiles4{flex-wrap:wrap;width:292px}.biasrow>div:not(.card){max-width:292px}}
.biasrow .card{padding:4px;line-height:0}
.biasrow>.card:first-child{flex:1 1 300px;max-width:480px}.biasrow>.card:first-child svg{width:100%;height:auto}.tiles4{display:flex;gap:6px}
.tile4 .mut{font-size:11px;line-height:1.3}.tile4 svg{display:block}
.rel .band{opacity:0;pointer-events:none}.rel:hover .band{opacity:.22}
.rel{pointer-events:all}
.gck i{display:inline-block;width:10px;height:10px;border-radius:2px;
margin:0 4px}select,button{font:inherit}
pre{white-space:pre-wrap;background:var(--card);padding:8px;border-radius:8px;
font-size:12px;margin:4px 0}.wrap{overflow-x:auto}"""
    body = f"""<header class=topbar>
<div class=navrow><b>{html.escape(LC["label"])} run</b><nav class=nav>{nav}</nav>
{bar_tiles}<span class=mut>{datetime.now():%H:%M:%S}</span></div>
<div id=selbar class=selbar></div>
<script type="application/json" id=seldata>{seldata}</script>
<script>ftSel.init(JSON.parse(document.getElementById("seldata").textContent))</script>
</header>
<details class=sec id=status open><summary><h2>Status</h2></summary>
<div class=tiles>{tiles_h}</div>
<div class=mut>{sub}</div></details>
<details class=sec id=loss open><summary><h2>Loss</h2></summary>{modebar}{plots_h}</details>
<details class=sec id=bias open><summary><h2>Attention bias</h2></summary>{bias_h}</details>
<details class=sec id=performance open><summary><h2>Performance on real data</h2></summary>{sc_h}{md_h}</details>
<details class=sec id=nowcasts open><summary><h2>German nowcasts</h2></summary>{rv_h}</details>
{"" if not mh else f'<details class=sec id=mailbox open><summary><h2>Recent activity</h2></summary>{mh}</details>'}
<details class=sec id=tables><summary><h2>Tables</h2></summary>
<h3>Checkpoints</h3><div class=wrap>{ck_h}</div>
<h3>Slots vs no slots (matched checkpoints)</h3><div class=wrap>{cmp_h}</div>
</details>"""
    return (f'<!doctype html><html><head><meta charset=utf-8>'
            f'<meta name=viewport content="width=device-width,initial-scale=1">'
            f'<title>FT dashboard</title>'
            f'<style>{css}</style><script>{PLAYER_JS}\n{SEL_JS}</script></head>'
            f'<body>{body}<script>{MODE_JS}\n{NAV_JS}\n{REFRESH_JS}</script></body></html>')


LOCK = threading.Lock()


class H(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path.startswith("/data/"):
            from urllib.parse import unquote
            parts = unquote(self.path[6:].split("?")[0])[:-5].split("/", 1)
            ds = next((d for d in DATASETS if d["key"] == parts[0]), None)
            with LOCK:
                b = (ger_loc_json(ds, parts[1]) if ds and len(parts) == 2
                     else "{}").encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(b)))
            self.end_headers()
            self.wfile.write(b)
            return
        try:
            with LOCK:
                b = build().encode()
            self.send_response(200)
        except Exception as e:
            b = f"error: {e}".encode()
            self.send_response(500)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(b)))
        self.end_headers()
        self.wfile.write(b)

    def log_message(self, *a):
        pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--once", metavar="FILE")
    ap.add_argument("--scores-csv", help="local checkpoint-scores.csv (test)")
    ap.add_argument("--review-dir", help="local copy of ~/bdbv-ckpt-review")
    ap.add_argument("--runs-config", help="JSON list replacing RUNS (test)")
    ap.add_argument("--germany-dir", help="local mirror of germany/ (test)")
    a = ap.parse_args()
    OVERRIDE["scores"], OVERRIDE["review_dir"] = a.scores_csv, a.review_dir
    OVERRIDE["germany"] = a.germany_dir
    if a.runs_config:
        cfgs = json.loads(Path(a.runs_config).read_text())
        if isinstance(cfgs, dict):
            EXCLUDE.update(cfgs.get("exclude", {}))
            cfgs = cfgs["runs"]
        for c in cfgs:
            c["dir"] = Path(c["dir"])
        RUNS[:] = cfgs
    if a.once:
        ONCE["on"] = True
        mail_kick(wait=True)
        for k in ("scores", "md", "germany"):
            rem_kick(k, wait=True)
        Path(a.once).write_text(build())
        print("wrote", a.once)
        return
    srv = ThreadingHTTPServer(("127.0.0.1", a.port), H)
    print(f"http://127.0.0.1:{a.port}/")
    srv.serve_forever()


if __name__ == "__main__":
    main()
