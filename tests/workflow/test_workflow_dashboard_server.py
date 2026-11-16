"""The dashboard server answers the page and one location's data."""

import json
import sys
import threading
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
    assert get(server + "/data/germany/XX.json")[1] == "{}"
