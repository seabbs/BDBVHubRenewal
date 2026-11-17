"""The dashboard server answers the page and one location's data."""

import json
import sys
import threading
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer
from pathlib import Path

import pytest

pytest.importorskip("matplotlib")

sys.path.insert(0, str(Path(__file__).parents[2] / "workflow" / "scripts"))
sys.path.insert(0, str(Path(__file__).parent))

import dashboard
from test_workflow_dashboard_page import _machine


@pytest.fixture
def server(tmp_path):
    ctx = _machine(tmp_path)
    ctx.once = False
    srv = ThreadingHTTPServer(("127.0.0.1", 0), dashboard.handler(ctx))
    thread = threading.Thread(target=srv.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{srv.server_address[1]}"
    srv.shutdown()


def get(url):
    with urllib.request.urlopen(url, timeout=30) as r:
        return r.status, r.read().decode()


def test_the_page_is_served(server):
    status, body = get(server + "/")
    assert status == 200
    assert "<h2>German nowcasts by checkpoint" in body


def test_one_location_s_data_is_served(server):
    status, body = get(server + "/data/germany/DE.json")
    assert status == 200
    data = json.loads(body)
    assert set(data) == {"series", "wis", "truth"}


def status_of(url):
    try:
        return get(url)[0]
    except urllib.error.HTTPError as err:
        return err.code


def test_unknown_locations_and_paths_are_not_found(server):
    assert status_of(server + "/data/germany/XX.json") == 404
    assert status_of(server + "/favicon.ico") == 404


def test_a_failing_data_request_is_a_server_error(server, monkeypatch):
    def broken(*_args):
        raise ValueError("bad cache")

    monkeypatch.setattr(dashboard.germany, "location", broken)
    assert status_of(server + "/data/germany/DE.json") == 500


def test_once_shows_and_reports_a_failed_mirror(tmp_path, monkeypatch):
    ctx = _machine(tmp_path)
    monkeypatch.setattr(
        ctx.source, "refresh", lambda: "mirror failed: host unreachable"
    )
    out = tmp_path / "page" / "index.html"
    assert dashboard.write_once(ctx, out) == 1
    assert "mirror failed: host unreachable" in out.read_text()


def test_once_shows_but_does_not_fail_on_a_failed_mailbox(
    tmp_path, monkeypatch
):
    ctx = _machine(tmp_path)
    status = "mirrored from h; mailbox failed: no such file"
    monkeypatch.setattr(ctx.source, "refresh", lambda: status)
    out = tmp_path / "page" / "index.html"
    assert dashboard.write_once(ctx, out) == 0
    assert "mailbox failed: no such file" in out.read_text()
