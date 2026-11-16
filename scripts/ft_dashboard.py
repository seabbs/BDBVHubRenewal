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
CKPTS = [250, 500, 1000, 2000, 4000, 8000]
TARGET_SIMS, LR0, LR_STEPS = 8000, 1e-5, 36500
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


LOGS = {"slots": Log(RUN / "train.log"), "prev": Log(PREV / "train.log")}
JC = Cache()
MAIL = {"t": 0.0, "lines": [], "ok": None, "busy": False}


def mail_fetch():
    try:
        r = subprocess.run(
            ["ssh", "-o", "ConnectTimeout=10", "-o", "BatchMode=yes",
             "archie", "tail -n 30 ~/bdbv-archie-mailbox.md"],
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


def pid_state():
    pf = RUN / "trainer.pid"
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


def chart(title, series, marks=(), ylog=False, W=620, H=300,
          xticks=None, hlines=(), xlabel="step"):
    """series: list of (label, colour, points[(x,y)], kind) kind line|dots"""
    import math
    pts = [p for s in series for p in s[2] if p[1] is not None]
    if not pts:
        return f"<div class=card><h3>{title}</h3><p class=mut>no data</p></div>"
    f = (lambda y: math.log10(max(y, 1e-9))) if ylog else (lambda y: y)
    xs = [p[0] for p in pts]
    ys = [f(p[1]) for p in pts]
    x0, x1 = 0, max(max(xs), 1)
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
        dash = f' stroke-dasharray="5 3"' if len(s) > 4 and s[4] == "dash" \
            else ""
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
        f'{" (dashed)" if len(s) > 4 and s[4] == "dash" else ""}</span>'
        for s in series)
    return (f"<div class=card><h3>{title}</h3>{''.join(o)}"
            f"<div class=leg>{leg}</div></div>")


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


def score_charts(rows, slot_hashes, r4_hashes):
    """Relative-score charts against checkpoint (simulations, log2 axis)."""
    import math
    rows = [r for r in rows if r.get("rule") == RULE]
    xt = [(125, "zero-shot"), (250, "250"), (500, "500"), (1000, "1000"),
          (2000, "2000"), (4000, "4000"), (8000, "8000")]

    def run_of(r):
        h = re.sub(r"^chronos-(slots|ft)-", "", r["checkpoint"])
        if h in r4_hashes or r.get("grid") == "r4":
            return "no slots"
        return "slots"

    def xval(r):
        if r["checkpoint"] == "zero-shot":
            return 125
        n = num(r.get("outbreaks"))
        return n if n and n > 0 else None

    def lines(study, level_cols, model, col):
        ser = []
        for lvl, colour in level_cols:
            rs = [r for r in rows if r["study"] == study
                  and r["level"] == lvl and r["model"] == model
                  and num(r.get(col)) is not None and xval(r)]
            zs = [(125, num(r[col])) for r in rs
                  if r["checkpoint"] == "zero-shot"]
            for run, dash in (("slots", None), ("no slots", "dash")):
                pts = sorted(zs + [(xval(r), num(r[col])) for r in rs
                                   if r["checkpoint"] != "zero-shot"
                                   and run_of(r) == run])
                if len(pts) > 1 or (pts and run == "slots" and not zs):
                    s = (f"{lvl}, {run}", colour, pts, "pts")
                    ser.append(s + (dash,) if dash else s)
        return ser

    ger = [("national", PAL[0]), ("states", PAL[1]), ("age", PAL[2])]
    bd = [("national", PAL[0]), ("provinces", PAL[1])]
    out = []
    for study, cols, col, lab in (
            ("germany", ger, "rel_wis", "Germany first half, rel WIS"),
            ("ebola", bd, "rel_crps", "BDBV, rel CRPS")):
        for model, nm in (("fm_chronos_daily_ragged", "daily"),
                          ("fm_chronos_weekly_ragged", "weekly")):
            out.append(chart(
                f"{lab} vs randomwalknowcast, {nm}",
                lines(study, cols, model, col), ylog=True, xticks=xt,
                hlines=(1.0,), xlabel="simulations (log2)"))
    ho = []
    for lvl, colour in (("all", PAL[0]), ("real_validation", PAL[3])):
        rs = [r for r in rows if r["study"] == "held_out"
              and r["level"] == lvl and num(r.get("loss")) is not None]
        for run, dash in (("slots", None), ("no slots", "dash")):
            pts = sorted((xval(r), num(r["loss"])) for r in rs
                         if run_of(r) == run and xval(r))
            z = [(125, num(r["loss_zero_shot"])) for r in rs
                 if num(r.get("loss_zero_shot")) is not None][:1]
            if pts:
                s = (f"{lvl}, {run}", colour, z + pts, "pts")
                ho.append(s + (dash,) if dash else s)
    out.append(chart("Held-out loss (all blocks; real validation)", ho,
                     ylog=True, xticks=xt, xlabel="simulations (log2)"))
    return out


RULE = "own-quantile-wis_crps-1000-common"


# ---------- German nowcasts: mirrored per-checkpoint forecasts ----------
GER_REMOTE = "archie:ckpt-scores/cache/germany/"
QS = ["0.025", "0.1", "0.25", "0.5", "0.75", "0.9", "0.975"]
GER = {"files": {}, "sig": None, "built": None, "loc_json": {}}


def ger_dir():
    return Path(OVERRIDE["germany"]) if OVERRIDE["germany"] \
        else CACHE / "germany"


def job_germany():
    if OVERRIDE["germany"]:
        return
    d = ger_dir()
    d.mkdir(parents=True, exist_ok=True)
    r = subprocess.run(
        ["rsync", "-a", "-e", "ssh -o ConnectTimeout=10 -o BatchMode=yes",
         GER_REMOTE, str(d) + "/"], capture_output=True, text=True,
        timeout=300)
    if r.returncode != 0:
        raise RuntimeError(r.stderr[-200:].strip() or f"rsync {r.returncode}")


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
            srcs = ["daily", "weekly"] if mdl == "randomwalknowcast" else \
                ["daily"] if "daily" in mdl else \
                ["weekly"] if "weekly" in mdl else []
            q = num(r["value"])
            if r["quantile"] not in QS or q is None:
                continue
            for s in srcs:
                k = (r["location"], "rw" if mdl == "randomwalknowcast"
                     else None, s)
                e = out.setdefault(k, {}).setdefault(
                    r["target_end_date"], {"q": {}, "y": None, "v": ""})
                e["q"][r["quantile"]] = q
                y = num(r.get("truth"))
                if y is not None:
                    e["y"], e["v"] = y, r.get("truth_vintage") or ""
    return out


def ger_build():
    """Assemble per-location data from the mirrored files (cached)."""
    base = ger_dir()
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
    if sig == GER["sig"]:
        return GER["built"]
    cache = GER["files"]
    series, wis, truth, origins = {}, {}, {}, set()
    for setid, origin, f, mt in files:
        key = str(f)
        if key not in cache or cache[key][0] != mt:
            try:
                cache[key] = (mt, parse_ger_file(f))
            except Exception:
                continue
        origins.add(origin) if setid == "reference" else None
        sid = "zero-shot" if setid == "reference" else setid
        for (loc, rw, src), dates in cache[key][1].items():
            s = rw or sid
            ds = sorted(dates)
            qm, ts = [], []
            for dte in ds:
                e = dates[dte]
                if all(k in e["q"] for k in QS):
                    qm.append([round(e["q"][k], 1) for k in QS])
                    ts.append(dte)
                    if e["y"] is not None:
                        tv = truth.setdefault(loc, {})
                        if dte not in tv or e["v"] >= tv[dte][0]:
                            tv[dte] = (e["v"], e["y"])
            if not ts:
                continue
            series.setdefault(loc, {}).setdefault(s, {}).setdefault(
                src, {})[origin] = {"t": ts, "q": qm}
            ws = [wis_row(q, dates[t]["y"]) for q, t in zip(qm, ts)
                  if dates[t]["y"] is not None]
            if ws:
                wis.setdefault(loc, {}).setdefault(s, {}).setdefault(
                    src, {})[origin] = round(sum(ws) / len(ws), 2)
    sets = sorted({s for l in series.values() for s in l})
    locs = sorted(series, key=lambda l: (
        0 if l == "DE" else 1 if l.startswith("DE-") else 2, l))
    built = {"series": series, "wis": wis, "truth": truth, "sets": sets,
             "locs": locs, "origins": sorted(origins or {
                 o for l in series.values() for s in l.values()
                 for src in s.values() for o in src})}
    GER["sig"], GER["built"], GER["loc_json"] = sig, built, {}
    return built


def ger_loc_json(loc):
    b = ger_build()
    if not b or loc not in b["series"]:
        return "{}"
    if loc not in GER["loc_json"]:
        tv = sorted(b["truth"].get(loc, {}).items())
        GER["loc_json"][loc] = json.dumps({
            "series": b["series"][loc], "wis": b["wis"].get(loc, {}),
            "truth": {"t": [d for d, _ in tv], "y": [v[1] for _, v in tv]}},
            separators=(",", ":"))
    return GER["loc_json"][loc]


def ger_meta(b, slot_n, prev_n):
    """Label each folder by run, n and hash."""
    csv_n = {}
    for r in read_scores():
        h = re.sub(r"^chronos-(slots|ft)-", "", r["checkpoint"])
        n = num(r.get("outbreaks"))
        if n:
            csv_n.setdefault(h, int(n))
    meta = []
    for s in b["sets"]:
        if s == "zero-shot":
            meta.append(dict(id=s, label="zero-shot", run="ref", n=0))
        elif s == "rw":
            meta.append(dict(id=s, label="randomwalknowcast", run="ref", n=1))
        else:
            if s in slot_n:
                run, n = "slots", slot_n[s]
            elif s in prev_n or s in ("de129103cacc", "7962fd718a47"):
                run, n = "no slots", prev_n.get(s) or csv_n.get(s) or {
                    "de129103cacc": 1000, "7962fd718a47": 2000}.get(s)
            else:
                run, n = "older run", csv_n.get(s)
            meta.append(dict(id=s, run=run, n=n or 0, label=(
                f"{run} n{n} {s}" if n else f"{run} {s}")))
    order = {"ref": 0, "slots": 1, "no slots": 2, "older run": 3}
    meta.sort(key=lambda m: (order.get(m["run"], 4), m["n"]))
    return meta


GER_JS = r'''
(function(){
var D=JSON.parse(document.getElementById("gdata").textContent);
var META=D.meta,LOCS=D.locs,ORI=D.origins,PRE=D.preload||{};
var COL=["#2563eb","#ea580c","#16a34a","#9333ea","#db2777","#0891b2","#ca8a04","#64748b","#0d9488","#b91c1c"];
var $=function(i){return document.getElementById(i)};
var S={loc:"DE",src:"daily",oi:ORI.length-1,latest:true,sel:null};
try{var sv=JSON.parse(sessionStorage.getItem("ftg")||"null");
 if(sv){S.loc=LOCS.indexOf(sv.loc)>=0?sv.loc:"DE";S.src=sv.src||"daily";
  S.latest=sv.latest!==false;S.sel=sv.sel;
  if(!S.latest){var k=ORI.indexOf(sv.origin);if(k>=0)S.oi=k;else S.latest=true}}}catch(e){}
if(!S.sel){S.sel=["zero-shot"];["slots","no slots"].forEach(function(r){
 var m=META.filter(function(x){return x.run==r});if(m.length)S.sel.push(m[m.length-1].id)})}
function save(){try{sessionStorage.setItem("ftg",JSON.stringify({loc:S.loc,src:S.src,
 latest:S.latest,origin:ORI[S.oi],sel:S.sel}))}catch(e){}}
var cache={};
function load(loc){
 if(PRE[loc])return Promise.resolve(PRE[loc]);
 if(cache[loc])return Promise.resolve(cache[loc]);
 return fetch("/data/"+encodeURIComponent(loc)+".json").then(function(r){return r.json()})
  .then(function(j){cache[loc]=j;return j})}
function ms(s){return Date.parse(s+"T00:00:00Z")}
function colour(id){return COL[META.findIndex(function(m){return m.id==id})%COL.length]}
function el(h){return h}
function fmt(v){return v>=1000?Math.round(v).toLocaleString("en-GB"):(+v).toPrecision(3)}
function plot(d){
 var o=ORI[S.oi],t0=ms(o)-70*864e5,t1=ms(o)+35*864e5;
 var W=900,H=380,L=56,R=14,T=12,B=34,ymax=1,i;
 var tr=d.truth;var pts=[];
 for(i=0;i<tr.t.length;i++){var x=ms(tr.t[i]);if(x>=t0&&x<=t1){pts.push([x,tr.y[i]]);ymax=Math.max(ymax,tr.y[i])}}
 var fs=[];
 S.sel.forEach(function(id){
  var s=(d.series[id]||{})[S.src];var f=s&&s[o];
  if(f){fs.push({id:id,f:f});f.q.forEach(function(q){ymax=Math.max(ymax,q[6])})}});
 ymax*=1.06;
 var sx=function(x){return L+(x-t0)/(t1-t0)*(W-L-R)},sy=function(y){return T+(1-y/ymax)*(H-T-B)};
 var h='<svg viewBox="0 0 '+W+' '+H+'" width="100%">';
 var step=Math.pow(10,Math.floor(Math.log10(ymax/4)));
 var m=[1,2,2.5,5,10].find(function(k){return k*step>=ymax/5})*step;
 for(var v=0;v<=ymax;v+=m)h+='<line x1="'+L+'" x2="'+(W-R)+'" y1="'+sy(v)+'" y2="'+sy(v)+'" class=grid /><text x="'+(L-6)+'" y="'+(sy(v)+4)+'" text-anchor=end class=ax>'+fmt(v)+'</text>';
 for(var x=t0;x<=t1;x+=14*864e5){h+='<text x="'+sx(x)+'" y="'+(H-14)+'" text-anchor=middle class=ax>'+new Date(x).toISOString().slice(5,10)+'</text>'}
 h+='<line x1="'+sx(ms(o))+'" x2="'+sx(ms(o))+'" y1="'+T+'" y2="'+(H-B)+'" class=mark /><text x="'+(sx(ms(o))+3)+'" y="'+(T+10)+'" class=ax>nowcast date</text>';
 fs.forEach(function(e){
  var c=colour(e.id),n=e.f.t.length,P=function(k){return e.f.t.map(function(d,j){return sx(ms(d))+","+sy(e.f.q[j][k])})};
  function band(a,b,op){h+='<polygon points="'+P(a).concat(P(b).reverse()).join(" ")+'" fill="'+c+'" opacity="'+op+'" />'}
  band(0,6,.10);band(2,4,.22);
  h+='<polyline points="'+P(3).join(" ")+'" fill=none stroke="'+c+'" stroke-width=2 />';
  e.f.t.forEach(function(d,j){h+='<circle cx="'+sx(ms(d))+'" cy="'+sy(e.f.q[j][3])+'" r=3 fill="'+c+'"><title>'+id2lab(e.id)+' '+d+' median '+fmt(e.f.q[j][3])+' (50%: '+fmt(e.f.q[j][2])+' to '+fmt(e.f.q[j][4])+'; 95%: '+fmt(e.f.q[j][0])+' to '+fmt(e.f.q[j][6])+')</title></circle>'})});
 if(pts.length){h+='<polyline points="'+pts.map(function(p){return sx(p[0])+","+sy(p[1])}).join(" ")+'" fill=none stroke="var(--fg)" stroke-width=1.6 />';
  pts.forEach(function(p){h+='<circle cx="'+sx(p[0])+'" cy="'+sy(p[1])+'" r=2.4 fill="var(--fg)"><title>truth '+new Date(p[0]).toISOString().slice(0,10)+': '+fmt(p[1])+'</title></circle>'})}
 h+='</svg>';
 var miss=S.sel.filter(function(id){return !fs.some(function(e){return e.id==id})}).map(id2lab);
 $("gplot").innerHTML=h+'<div class=leg><span><i style="background:var(--fg)"></i>truth (latest vintage)</span>'+fs.map(function(e){return '<span><i style="background:'+colour(e.id)+'"></i>'+id2lab(e.id)+'</span>'}).join("")+'</div>'+(miss.length?'<p class=mut>No forecast for '+o+': '+miss.join(", ")+'</p>':"")+'<p class=mut>Weekly counts by target week. Line is the median, dark band 50%, light band 95%.</p>';
}
function id2lab(id){var m=META.find(function(x){return x.id==id});return m?m.label:id}
function wisChart(d){
 var sel=S.sel.filter(function(id){return d.wis[id]&&d.wis[id][S.src]});
 if(!sel.length){$("gwis").innerHTML='<p class=mut>No WIS available.</p>';return}
 var com=Object.keys(d.wis[sel[0]][S.src]).filter(function(o){return sel.every(function(id){return d.wis[id][S.src][o]!==undefined})});
 if(!com.length){$("gwis").innerHTML='<p class=mut>No nowcast dates common to the selected checkpoints.</p>';return}
 var vals=sel.map(function(id){var a=com.map(function(o){return d.wis[id][S.src][o]});return a.reduce(function(x,y){return x+y},0)/a.length});
 var W=520,H=60+sel.length*26,L=230,mx=Math.max.apply(null,vals)*1.05,h='<svg viewBox="0 0 '+W+' '+H+'" width="100%">';
 sel.forEach(function(id,i){var y=10+i*26,w=vals[i]/mx*(W-L-60);
  h+='<text x="'+(L-6)+'" y="'+(y+14)+'" text-anchor=end class=ax>'+id2lab(id)+'</text><rect x="'+L+'" y="'+y+'" width="'+w+'" height="18" fill="'+colour(id)+'" /><text x="'+(L+w+4)+'" y="'+(y+14)+'" class=ax>'+fmt(vals[i])+'</text>'});
 $("gwis").innerHTML=h+'</svg><p class=mut>Mean WIS for '+S.loc+' ('+S.src+' source) over '+com.length+' nowcast dates common to the selected checkpoints. Lower is better.</p>';
}
function draw(){
 $("gdate").textContent=ORI[S.oi]+(S.latest?" (latest)":"");
 $("gsl").value=S.oi;
 load(S.loc).then(function(d){plot(d);wisChart(d)}).catch(function(){$("gplot").innerHTML='<p class=mut>Could not load data for '+S.loc+'.</p>'});
 save()}
function build(){
 $("gloc").innerHTML=LOCS.map(function(l){return '<option'+(l==S.loc?' selected':'')+'>'+l+'</option>'}).join("");
 $("gsrc").value=S.src;
 var sl=$("gsl");sl.min=0;sl.max=ORI.length-1;
 $("gck").innerHTML=META.map(function(m){return '<label><input type=checkbox value="'+m.id+'"'+(S.sel.indexOf(m.id)>=0?' checked':'')+'><i style="background:'+colour(m.id)+'"></i>'+m.label+' <span class=mut>('+m.count+' dates)</span></label>'}).join("");
 $("gloc").onchange=function(){S.loc=this.value;draw()};
 $("gsrc").onchange=function(){S.src=this.value;draw()};
 sl.oninput=function(){S.oi=+this.value;S.latest=S.oi==ORI.length-1;draw()};
 $("gprev").onclick=function(){S.oi=Math.max(0,S.oi-1);S.latest=false;draw()};
 $("gnext").onclick=function(){S.oi=Math.min(ORI.length-1,S.oi+1);S.latest=S.oi==ORI.length-1;draw()};
 $("gck").onchange=function(){S.sel=[].slice.call(this.querySelectorAll("input:checked")).map(function(i){return i.value});draw()};
 draw()}
if(!ORI.length)$("gplot").innerHTML='<p class=mut>No German forecasts mirrored yet.</p>';else build();
})();
'''


def germany_html(slot_n, prev_n, once):
    b = ger_build()
    if not b or not b["origins"]:
        return note("germany", "German forecasts") or \
            "<p class=mut>No German forecasts mirrored yet.</p>"
    meta = ger_meta(b, slot_n, prev_n)
    for m in meta:
        m["count"] = max((len(src) for src in b["series"].get(
            "DE", {}).get(m["id"], {}).values()), default=0)
    pre = {l: json.loads(ger_loc_json(l)) for l in b["locs"]} if once \
        else {}
    payload = json.dumps({"meta": meta, "locs": b["locs"],
                          "origins": b["origins"], "preload": pre},
                         separators=(",", ":")).replace("</", "<\\/")
    return (note("germany", "German forecasts") + f"""
<div class=gctl>
<label>Location <select id=gloc></select></label>
<label>Source <select id=gsrc><option value=daily>daily</option>
<option value=weekly>weekly</option></select></label>
<label class=gwide>Nowcast date <button id=gprev>&lt;</button>
<input type=range id=gsl><button id=gnext>&gt;</button>
<b id=gdate></b></label></div>
<div id=gck class=gck></div>
<div class=plots><div class=card><h3>Nowcast, weekly counts</h3>
<div id=gplot></div></div>
<div class=card><h3>WIS by checkpoint</h3><div id=gwis></div></div></div>
<script type="application/json" id=gdata>{payload}</script>
<script>{GER_JS}</script>""")


ONCE = {"on": False}

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
    for lg in LOGS.values():
        lg.update()
    sl, pv = LOGS["slots"].sorted(), LOGS["prev"].sorted()
    status = JC.get(str(RUN / "train/stream-status.json")) or {}
    hist = JC.get(str(RUN / "train/history.json")) or []
    cons = RUN / "train/consumed.txt"
    nsims = 0
    try:
        for ln in cons.read_text().splitlines():
            parts = ln.split("\t", 1)
            if len(parts) == 2:
                nsims += len([x for x in parts[1].split(",") if x])
    except Exception:
        pass
    step = sl[-1]["step"] if sl else status.get("step")
    if isinstance(status.get("step"), int) and (
            step is None or status["step"] > step):
        step = status["step"]
    last_t = sl[-1]["t"] if sl else None
    rate = None
    recent = [r for r in sl if r["t"] >= (last_t or 0) - 1800]
    if len(recent) >= 2 and recent[-1]["t"] > recent[0]["t"]:
        rate = (recent[-1]["step"] - recent[0]["step"]) / (
            recent[-1]["t"] - recent[0]["t"])
    nxt = next((c for c in CKPTS if step is not None and c > step), None)
    eta = lambda tgt: dur((tgt - step) / rate) if rate and step is not None \
        and tgt else "n/a"
    lr = LR0 * max(0.0, 1 - step / LR_STEPS) if step is not None else None
    pidtxt, alive = pid_state()
    age = now - last_t if last_t else None
    stale = age is not None and age > 600
    cls = lambda ok: "ok" if ok else "bad"
    tiles = [
        ("Step", f"{step}" if step is not None else "n/a", ""),
        ("Simulations consumed", f"{nsims} / {TARGET_SIMS}", ""),
        ("Steps / s (30 min)", f"{rate:.3f}" if rate else "n/a",
         f"{rate*60:.1f} steps/min" if rate else ""),
        ("Next checkpoint", f"n{nxt}" if nxt else "none",
         f"ETA {eta(nxt)}" if nxt else ""),
        ("ETA n8000", eta(8000), ""),
        ("Learning rate", f"{lr:.3e}" if lr is not None else "n/a",
         f"linear decay, {LR_STEPS} steps"),
        ("Trainer", pidtxt, cls(alive) if alive is not None else ""),
        ("Last log line",
         datetime.fromtimestamp(last_t).strftime("%H:%M:%S") if last_t
         else "n/a",
         ("STALE " if stale else "") + (dur(age) + " ago" if age is not None
                                        else "") if age is not None else ""),
    ]
    tiles_h = "".join(
        f'<div class="tile {c if c else ""}{" bad" if t=="Last log line" and stale else ""}">'
        f'<div class=k>{k}</div><div class=v>{html.escape(v)}</div>'
        f'<div class=s>{html.escape(s) if c not in ("ok","bad") else ""}</div>'
        f'</div>' for k, v, s in tiles for t, c in [(k, s if k == "Trainer" else "")])
    stage = status.get("stage")
    sub = (f"stage: {html.escape(str(stage))}; stream status "
           f"{html.escape(str(status.get('time', '')))}" if status else "")

    cks_s = checkpoints(RUN)
    cks_p = checkpoints(PREV)
    marks = []
    for name, s, _ in cks_s:
        n = int(re.sub(r"\D", "", name) or 0)
        marks.append((n, name))
    train = [(h["step"], h["loss"]) for h in hist
             if isinstance(h, dict) and "loss" in h and "step" in h]
    if not train:
        train = [(r["step"], r["loss"]) for r in sl]
    ph = JC.get(str(PREV / "train/history.json"))
    ptrain = [(h["step"], h["loss"]) for h in ph or []
              if isinstance(h, dict) and "loss" in h and "step" in h]
    if not ptrain:
        ptrain = [(r["step"], r["loss"]) for r in pv]
    smax = max([x for x, _ in train] + [1])
    ptrain = [p for p in ptrain if p[0] <= smax]
    held_s = [(r["step"], r["held"]) for r in sl]
    held_p = [(r["step"], r["held"]) for r in pv]
    # Logged held-out losses are over the first 100 blocks; history's step-0
    # value is over all held-out blocks, so it is not on the same scale.
    plots = [
        chart("Training loss (rolling mean of 20 logged points)",
              [("slots raw", PAL[0], train, "dots"),
               ("slots rolling mean", PAL[1], rolling(train, 20), "line"),
               ("no slots (r4) rolling mean", PAL[2], rolling(ptrain, 20),
                "faint")],
              marks, ylog=True),
        chart("Held-out pooled loss, slots vs no slots",
              [("slots", PAL[0], held_s, "pts"),
               ("no slots (r4)", PAL[2], held_p, "pts")], marks, ylog=True),
        chart("Held-out loss by source, slots run",
              [(k, PAL[i], [(r["step"], r["src"][k]) for r in sl], "pts")
               for i, k in enumerate(["weekly", "daily", "3day"])], marks,
              ylog=True),
    ]

    def cktable(rows, run):
        if not rows:
            return f"<p class=mut>no checkpoints yet for {run}</p>"
        o = ["<table><tr><th>run<th>ckpt<th>hash<th>held-out<th>held-out "
             "start<th>real-val<th>weekly<th>daily<th>3day<th>saved</tr>"]
        for name, s, mt in rows:
            bs = s.get("held_out_loss_by_source") or {}
            o.append("<tr>" + "".join(f"<td>{c}" for c in [
                run, name, html.escape(str(s.get("hash", ""))),
                fmt(s.get("held_out_loss")), fmt(s.get("held_out_loss_start")),
                fmt(s.get("real_validation_loss")),
                fmt(bs.get("weekly")) or "n/a", fmt(bs.get("daily")) or "n/a",
                fmt(bs.get("3day")) or "n/a",
                datetime.fromtimestamp(mt).strftime("%m-%d %H:%M") if mt
                else ""]) + "</tr>")
        return "".join(o) + "</table>"

    ck_h = cktable(cks_s, "slots") + cktable(cks_p, "no slots (r4)")
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
    mail_kick()
    if MAIL["ok"] is None:
        mh = "<p class=mut>loading mailbox (refreshes in a minute)</p>"
    elif MAIL["ok"] is False and not MAIL["lines"]:
        mh = "<p class=mut>mailbox unavailable (ssh failed)</p>"
    elif not MAIL["lines"]:
        mh = "<p class=mut>no matching lines in last 30</p>"
    else:
        mh = "<pre>" + html.escape("\n".join(MAIL["lines"])) + "</pre>"

    for k in ("scores", "md", "germany"):
        rem_kick(k)
    slot_hashes = {str(s.get("hash")) for _, s, _ in cks_s}
    r4_hashes = {str(s.get("hash")) for _, s, _ in cks_p} | {
        "de129103cacc", "7962fd718a47"}
    srows = read_scores()
    sc_h = note("scores", "Scores")
    if srows:
        sc_h += ('<p class=mut>Relative to randomwalknowcast (below 1 is '
                 'better; dashed line is 1). Rule own-quantile-wis_crps-'
                 '1000-common. Solid is slots, dashed is no slots.</p>'
                 '<div class=plots>' + "".join(
                     score_charts(srows, slot_hashes, r4_hashes)) + "</div>")
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
        {str(s.get("hash")): s.get("n_outbreaks") for _, s, _ in cks_s},
        {str(s.get("hash")): s.get("n_outbreaks") for _, s, _ in cks_p},
        ONCE["on"])

    css = """
:root{--bg:#fff;--fg:#1d2127;--mut:#6b7280;--card:#f6f7f9;--line:#d9dde3;
--c1:#2563eb;--c2:#ea580c;--c3:#16a34a;--c4:#9333ea;--bad:#dc2626;--ok:#15803d}
@media(prefers-color-scheme:dark){:root{--bg:#14171c;--fg:#e6e8eb;
--mut:#9aa3af;--card:#1d2128;--line:#343a44;--c1:#60a5fa;--c2:#fb923c;
--c3:#4ade80;--c4:#c084fc;--bad:#f87171;--ok:#4ade80}}
body{background:var(--bg);color:var(--fg);font:14px system-ui,sans-serif;
margin:0;padding:16px;max-width:1300px;margin-inline:auto}
h1{font-size:18px;margin:0 0 4px}h2{font-size:15px;margin:24px 0 8px}
h3{font-size:13px;margin:0 0 6px}.mut{color:var(--mut)}
.tiles{display:grid;grid-template-columns:repeat(auto-fill,minmax(170px,1fr));
gap:8px}.tile,.card{background:var(--card);border-radius:8px;padding:10px}
.k{font-size:11px;color:var(--mut)}.v{font-size:18px;font-weight:600;
font-variant-numeric:tabular-nums}.s{font-size:11px;color:var(--mut)}
.tile.ok .v{color:var(--ok)}.tile.bad .v{color:var(--bad)}
.plots{display:grid;grid-template-columns:repeat(auto-fit,minmax(340px,1fr));
gap:8px}svg{display:block}.grid{stroke:var(--line);stroke-width:1}
.mark{stroke:var(--mut);stroke-dasharray:3 3;stroke-width:1}
.ax{fill:var(--mut);font-size:10px}.leg{font-size:12px;display:flex;gap:12px;
flex-wrap:wrap}.leg i{display:inline-block;width:10px;height:10px;
border-radius:2px;margin-right:4px}
table{border-collapse:collapse;width:100%;margin-bottom:10px;
font-variant-numeric:tabular-nums}th,td{padding:3px 8px;text-align:left;
border-bottom:1px solid var(--line);font-size:12px}
.gctl{display:flex;flex-wrap:wrap;gap:12px;align-items:center;
margin-bottom:8px}.gwide input{width:320px;vertical-align:middle}
.gck{display:flex;flex-wrap:wrap;gap:6px 16px;margin-bottom:8px}
.gck i{display:inline-block;width:10px;height:10px;border-radius:2px;
margin:0 4px}select,button{font:inherit}
pre{white-space:pre-wrap;background:var(--card);padding:10px;border-radius:8px;
font-size:12px}.wrap{overflow-x:auto}"""
    body = f"""<h1>Chronos-2 fine-tuning: slots run</h1>
<div class=mut>{datetime.now():%Y-%m-%d %H:%M:%S}; auto-refresh 60 s; {sub}</div>
<h2>Status</h2><div class=tiles>{tiles_h}</div>
<h2>Plots</h2><div class=plots>{''.join(plots)}</div>
<p class=mut>Dashed lines mark published checkpoints of the slots run.</p>
<h2>Checkpoints</h2><div class=wrap>{ck_h}</div>
<h2>Slots vs no slots (matched checkpoints)</h2><div class=wrap>{cmp_h}</div>
<h2>Performance on real data</h2>{sc_h}{md_h}
<h2>German nowcasts by checkpoint</h2>{rv_h}
<h2>Recent activity (archie mailbox)</h2>{mh}"""
    return (f'<!doctype html><html><head><meta charset=utf-8>'
            f'<meta name=viewport content="width=device-width,initial-scale=1">'
            f'<meta http-equiv=refresh content=60><title>FT dashboard</title>'
            f'<style>{css}</style></head><body>{body}</body></html>')


LOCK = threading.Lock()


class H(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path.startswith("/data/"):
            from urllib.parse import unquote
            loc = unquote(self.path[6:].split("?")[0])[:-5]
            with LOCK:
                b = ger_loc_json(loc).encode()
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
    ap.add_argument("--germany-dir", help="local mirror of germany/ (test)")
    a = ap.parse_args()
    OVERRIDE["scores"], OVERRIDE["review_dir"] = a.scores_csv, a.review_dir
    OVERRIDE["germany"] = a.germany_dir
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
