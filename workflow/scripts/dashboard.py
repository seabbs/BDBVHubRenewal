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
import sys
import threading
import time
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlsplit

import yaml

sys.path.insert(0, str(Path(__file__).parents[1]))

from flowlib import localconfig
from flowlib.dashboard import context, germany, page

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
    """Return the request handler: the page, and one location's data."""

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
                if path.startswith("/data/"):
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

        def _data(self, name: str) -> None:
            loc = name.removesuffix(".json").split("/", 1)[-1]
            with LOCK:
                found = germany.location(ctx.germany(), loc)
            if not found:
                self._send(404, b"{}", "application/json")
                return
            self._send(200, page._json(found).encode(), "application/json")

        def log_message(self, *_args: object) -> None:
            pass

    return Handler


def keep_mirrored(ctx: page.Context, every: float) -> None:
    """Refresh the score mirror every `every` seconds, outside requests."""

    def loop() -> None:
        while True:
            status = ctx.source.refresh()
            with LOCK:
                ctx.mirror_status = status
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
    if ctx.source.command():
        keep_mirrored(ctx, float(config["dashboard"].get("mirror_every", 300)))
    server = ThreadingHTTPServer(("127.0.0.1", a.port), handler(ctx))
    print(f"http://127.0.0.1:{a.port}/")
    server.serve_forever()


if __name__ == "__main__":
    main()
