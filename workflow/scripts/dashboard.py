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
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote

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
            if self.path.startswith("/data/"):
                parts = unquote(self.path[6:].split("?")[0])
                loc = parts.removesuffix(".json").split("/", 1)[-1]
                with LOCK:
                    built = germany.build(
                        ctx.source.germany,
                        ctx.runs,
                        ctx.references,
                        levels=ctx.style["levels"],
                    )
                    data = page._json(germany.location(built, loc))
                self._send(200, data.encode(), "application/json")
                return
            try:
                with LOCK:
                    ctx.source.refresh()
                    body = page.build(ctx).encode()
                self._send(200, body, "text/html; charset=utf-8")
            except Exception as err:
                self._send(500, f"error: {err}".encode(), "text/plain")

        def log_message(self, *_args) -> None:
            pass

    return Handler


def main() -> None:
    """Serve the dashboard or write it once."""
    a = parser().parse_args()
    config = yaml.safe_load(a.config.read_text())
    ctx = context.build(
        config, localconfig.read(), runs=a.run, refs=a.ref, once=bool(a.once)
    )
    if a.once:
        print(ctx.source.refresh())
        a.once.parent.mkdir(parents=True, exist_ok=True)
        a.once.write_text(page.build(ctx))
        print("wrote", a.once)
        return
    server = ThreadingHTTPServer(("127.0.0.1", a.port), handler(ctx))
    print(f"http://127.0.0.1:{a.port}/")
    server.serve_forever()


if __name__ == "__main__":
    main()
