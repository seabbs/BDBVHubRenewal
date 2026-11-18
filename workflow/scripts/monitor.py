"""The workflow monitor: what runs where (`task monitor`).

    python3 workflow/scripts/monitor.py

Serves one page listing each machine's workflow run from the status files
the ticks write (`flowlib.runstatus`): its step, its running jobs (rule,
wildcards, resources, start) and its recent ones. On the rounds host the
files are in `<ROOT>/status/`; elsewhere (`REMOTE` set) they are copied
from the rounds host's `status/` before each refresh. A run that stops
writing its status (killed, or its machine down) is shown as stale.
Where, and how often it refreshes, come from config/local.env
(`MONITOR_HOST`, `MONITOR_PORT`, `MONITOR_REFRESH`). It is separate from
any fine-tuning dashboard. Work run outside the workflow does not appear.
Standard library only.
"""

from __future__ import annotations

import json
import subprocess
import sys
from collections.abc import Mapping
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "workflow"))

from flowlib import localconfig  # noqa: E402


def _folder(cfg: Mapping[str, str]) -> Path:
    """Return where the status files are, copying them from REMOTE."""
    if not cfg.get("REMOTE"):
        return Path(cfg["ROOT"]).expanduser() / "status"
    here = ROOT / "output" / "monitor" / "status"
    here.mkdir(parents=True, exist_ok=True)
    there = f"{cfg.get('REMOTE_ROOT') or cfg['ROOT']}/status/"
    subprocess.run(
        ["rsync", "-a", f"{cfg['REMOTE']}:{there}", f"{here}/"],
        capture_output=True,
        check=False,
    )
    return here


def collect(cfg: Mapping[str, str], now: str | None = None) -> list[dict]:
    """Return each machine's status, with its age and whether it is stale.

    A run is stale when it has not written for two of its intervals.
    """
    t = datetime.fromisoformat(now) if now else datetime.now()
    out = []
    for path in sorted(_folder(cfg).glob("*.json")):
        got = json.loads(path.read_text())
        age = int((t - datetime.fromisoformat(got["updated"])).total_seconds())
        got["age"] = age
        got["stale"] = bool(got["alive"]) and age > 2 * int(got["every"])
        out.append(got)
    return out


PAGE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Workflow monitor</title>
<style>
:root { color-scheme: light dark; font-family: system-ui, sans-serif; }
body { margin: 1rem; }
table { border-collapse: collapse; margin: 0.3rem 0 1.2rem; }
td, th { padding: 0.2rem 0.6rem; border-bottom: 1px solid #8884;
  text-align: left; font-size: 0.9rem; }
.stale { color: #c0392b; } .live { color: #1e8449; } .idle { color: #888; }
</style></head><body>
<h1>Workflow monitor</h1>
<p>Each machine's workflow run, refreshed every REFRESH s.</p>
<div id="hosts">Loading…</div>
<script>
const esc = s => String(s ?? "").replace(/[&<>"]/g,
  c => ({"&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;"}[c]));
const res = r => Object.entries(r || {}).map(([k, v]) => k + "=" + v)
  .join(", ");
function table(head, rows) {
  if (!rows.length) return "<p class='idle'>None.</p>";
  return "<table><tr>" + head.map(h => "<th>" + h + "</th>").join("") +
    "</tr>" + rows.map(r => "<tr>" + r.map(c => "<td>" + esc(c) + "</td>")
    .join("") + "</tr>").join("") + "</table>";
}
async function draw() {
  const hosts = await (await fetch("status.json")).json();
  document.getElementById("hosts").innerHTML = hosts.length ? hosts.map(h => {
    const state = h.stale ? "stale" : h.alive ? "live" : "idle";
    return "<h2>" + esc(h.host) + " <span class='" + state + "'>" + state +
      "</span></h2><p>" + esc(h.step) + ", updated " + h.age + " s ago</p>" +
      table(["Rule", "Wildcards", "Resources", "Started"], h.running.map(
        j => [j.rule, j.wildcards, res(j.resources), j.started])) +
      "<details><summary>Recent jobs</summary>" +
      table(["Rule", "Wildcards", "State", "Ended"], h.recent.map(
        j => [j.rule, j.wildcards, j.state, j.ended])) + "</details>";
  }).join("") : "<p class='idle'>No machine has written a status yet.</p>";
}
draw(); setInterval(draw, REFRESH * 1000);
</script></body></html>
"""


def server(cfg: Mapping[str, str]) -> ThreadingHTTPServer:
    """Return the monitor's server on MONITOR_HOST and MONITOR_PORT."""
    page = PAGE.replace("REFRESH", str(int(cfg.get("MONITOR_REFRESH") or 30)))

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            if self.path.split("?")[0] == "/status.json":
                body, kind = (
                    json.dumps(collect(cfg)).encode(),
                    "application/json",
                )
            else:
                body, kind = page.encode(), "text/html; charset=utf-8"
            self.send_response(200)
            self.send_header("Content-Type", kind)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *_args: object) -> None:
            return

    host = cfg.get("MONITOR_HOST") or "127.0.0.1"
    return ThreadingHTTPServer(
        (host, int(cfg.get("MONITOR_PORT") or 0)), Handler
    )


def main() -> None:
    """Serve the monitor until stopped."""
    cfg = localconfig.read()
    srv = server(cfg)
    host, port = srv.server_address[:2]
    print(f"Workflow monitor on http://{host}:{port}/", flush=True)
    srv.serve_forever()


if __name__ == "__main__":
    main()
