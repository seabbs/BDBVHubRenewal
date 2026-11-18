"""Serve the fine-tuning dashboard, or write it once as a static page.

    dashboard.py [--port 8765] [--run SPEC ...] [--ref SPEC ...]
    dashboard.py --once FILE [--run SPEC ...]

Runs and references come from workflow/config.yaml's `dashboard`,
`DASHBOARD_RUNS` in config/local.env, and `--run`/`--ref` (see
`flowlib.dashboard.runs`). With `REMOTE` set, the scores are mirrored
read-only before each page. Serves on 127.0.0.1 only.
"""

from __future__ import annotations

import argparse
import hashlib
import sys
import threading
import time
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlsplit

import yaml

sys.path.insert(0, str(Path(__file__).parents[1]))

from flowlib import localconfig
from flowlib.dashboard import context, nowcasts, page

CONFIG = Path(__file__).parents[1] / "config.yaml"
LOCK = threading.Lock()


def parser() -> argparse.ArgumentParser:
    """Return the command line's parser."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--port", type=int, default=8765)
    p.add_argument("--once", type=Path, metavar="FILE")
    p.add_argument("--run", action="append", default=[], metavar="SPEC")
    p.add_argument("--ref", action="append", default=[], metavar="SPEC")
    p.add_argument("--config", type=Path, default=CONFIG)
    return p


def handler(ctx: page.Context) -> type[BaseHTTPRequestHandler]:
    """Return the request handler: the page, its parts and location data."""

    class Handler(BaseHTTPRequestHandler):
        def _send(self, code: int, body: bytes, kind: str) -> None:
            self.send_response(code)
            self.send_header("Content-Type", kind)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self) -> None:
            path = unquote(urlsplit(self.path).path)
            try:
                if path == "/sections.json":
                    query = parse_qs(urlsplit(self.path).query)
                    self._sections(query.get("have", [""])[0])
                elif path.startswith("/data/"):
                    self._data(path.removeprefix("/data/"))
                elif path in ("/", "/index.html"):
                    with LOCK:
                        body = page.build(ctx).encode()
                    self._send(200, body, "text/html; charset=utf-8")
                else:
                    self._send(404, b"not found", "text/plain")
            except Exception:
                # The detail goes to the server's log, not the browser.
                traceback.print_exc(file=sys.stderr)
                self._send(500, b"error: see the server's log", "text/plain")

        def _sections(self, have: str) -> None:
            # Each part's hash, with its html only where `have` (key:hash,
            # comma separated) holds another hash or none.
            known = dict(
                item.split(":", 1) for item in have.split(",") if ":" in item
            )
            with LOCK:
                parts = page.sections(ctx)
            out = {}
            for key, content in parts.items():
                digest = hashlib.sha1(content.encode()).hexdigest()[:12]
                out[key] = (
                    {"hash": digest}
                    if known.get(key) == digest
                    else {"hash": digest, "html": content}
                )
            self._send(200, page._json(out).encode(), "application/json")

        def _data(self, name: str) -> None:
            # /data/<dataset>/<location>.json
            key, _, loc = name.removesuffix(".json").partition("/")
            if key not in {d["key"] for d in ctx.datasets} or not loc:
                self._send(404, b"{}", "application/json")
                return
            with LOCK:
                found = nowcasts.location(ctx.nowcasts(key), loc)
            if not found:
                self._send(404, b"{}", "application/json")
                return
            self._send(200, page._json(found).encode(), "application/json")

        def log_message(self, *_args: object) -> None:
            pass

    return Handler


def keep_mirrored(ctx: page.Context, every: float) -> None:
    """Refresh copies and summaries every `every` seconds, outside requests."""

    def loop() -> None:
        while True:
            status = ctx.source.refresh()
            ctx.summarise()
            with LOCK:
                ctx.mirror_status = status
                # Warm each dataset's nowcasts, so pages do not build them.
                for ds in ctx.datasets:
                    ctx.nowcasts(ds["key"])
            time.sleep(every)

    threading.Thread(target=loop, daemon=True).start()


def write_once(ctx: page.Context, out: Path) -> int:
    """Refresh the mirror and write the page to `out`.

    Returns 1 if the score mirror failed, so the page shows the failure
    and the caller fails too, else 0. A failed mailbox copy is shown on
    the page but does not fail it, as the mailbox is optional.
    """
    ctx.mirror_status = ctx.source.refresh()
    print(ctx.mirror_status)
    ctx.summarise()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(page.build(ctx))
    print("wrote", out)
    return int(ctx.mirror_status.startswith("mirror failed"))


def main() -> None:
    """Serve the dashboard or write it once."""
    a = parser().parse_args()
    config = yaml.safe_load(a.config.read_text())
    ctx = context.build(
        config, localconfig.read(), runs=a.run, refs=a.ref, once=bool(a.once)
    )
    if a.once:
        sys.exit(write_once(ctx, a.once))
    # Copies and summaries are kept fresh in the background, from the start
    # and also on a machine whose sources are all local, so no page waits.
    keep_mirrored(ctx, float(config["dashboard"].get("mirror_every", 300)))
    server = ThreadingHTTPServer(("127.0.0.1", a.port), handler(ctx))
    print(f"http://127.0.0.1:{a.port}/")
    server.serve_forever()


if __name__ == "__main__":
    main()
