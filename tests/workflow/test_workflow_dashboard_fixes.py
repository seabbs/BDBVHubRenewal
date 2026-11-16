"""Fixes from #267: config values keep their commas, failures are reported."""

import io
import subprocess
import sys
import threading
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer
from pathlib import Path

import pytest
from flowlib.dashboard import mirror, rows, runs


def test_a_config_run_keeps_commas_and_equals_in_its_label():
    got = runs.load_runs(
        [
            {
                "key": "zi",
                "source": "slots-zeroinit-",
                "label": "slots, zero init = yes",
                "hashes": ["abc123"],
                "n": {"abc123": 250},
            }
        ],
        "",
        [],
    )
    assert got[0].label == "slots, zero init = yes"
    assert got[0].hashes == ("abc123",)
    assert got[0].n == {"abc123": 250}


def test_a_ref_regex_may_hold_commas():
    ref = runs.parse_reference(
        "multi=^fm_chronos_(daily|weekly){1,2}$,checkpoint=zero-shot,role=start"
    )
    assert ref.model == "^fm_chronos_(daily|weekly){1,2}$"
    assert ref.checkpoint == "zero-shot"
    assert ref.role == "start"


def test_a_run_spec_still_rejects_unknown_fields():
    with pytest.raises(ValueError, match="unknown"):
        runs.parse_run("a=b-,shade=3")
    assert runs.parse_run("a=b-,live,label=x").live


def test_config_and_spec_references_share_their_defaults():
    from_config = runs.load_references([{"key": "r", "model": "x"}], [])[0]
    from_spec = runs.parse_reference("r=x")
    assert from_config == from_spec


def test_a_failed_mailbox_copy_is_reported(tmp_path, monkeypatch):
    cfg = {
        "REMOTE": "h",
        "REMOTE_ROOT": "r",
        "REMOTE_MAILBOX": "box.md",
        "SCORE_CACHE": "/x",
    }
    src = mirror.Source(cfg, tmp_path)

    def run(cmd, **_):
        code = 23 if cmd[-1].endswith("mailbox.md") else 0
        return subprocess.CompletedProcess(cmd, code, "", "no such file")

    monkeypatch.setattr(mirror.subprocess, "run", run)
    status = src.refresh()
    assert status.startswith("mirrored from h")
    assert "mailbox failed" in status


def test_shared_row_helpers():
    assert rows.num("0.5") == 0.5
    assert rows.num("") is None
    assert rows.source("fm_chronos_daily_ragged_ft") == "daily"
    assert rows.source("randomwalknowcast") is None
    assert rows.sources("randomwalknowcast") == ["daily", "weekly"]


pytest.importorskip("matplotlib")

from flowlib.dashboard import page  # noqa: E402

sys.path.insert(0, str(Path(__file__).parents[2] / "workflow" / "scripts"))
sys.path.insert(0, str(Path(__file__).parent))

import dashboard  # noqa: E402
from test_workflow_dashboard_page import _machine  # noqa: E402


def test_a_page_with_no_runs_says_so(tmp_path):
    ctx = _machine(tmp_path)
    ctx.runs = []
    with pytest.raises(ValueError, match="no runs"):
        page.build(ctx)


def test_a_server_error_is_logged_not_sent(tmp_path, monkeypatch):
    ctx = _machine(tmp_path)

    def boom(_ctx):
        raise RuntimeError("secret detail")

    monkeypatch.setattr(dashboard.page, "build", boom)
    err = io.StringIO()
    monkeypatch.setattr(sys, "stderr", err)
    srv = ThreadingHTTPServer(("127.0.0.1", 0), dashboard.handler(ctx))
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        with pytest.raises(urllib.error.HTTPError) as got:
            urllib.request.urlopen(
                f"http://127.0.0.1:{srv.server_address[1]}/", timeout=30
            )
        body = got.value.read().decode()
    finally:
        srv.shutdown()
    assert got.value.code == 500
    assert "secret detail" not in body
    assert "secret detail" in err.getvalue()


def test_dashed_series_by_run():
    s = ("a", "#111", [(0, 1)], "pts")
    assert page._styled(s, 0) == s
    assert page._styled(s, 1) == (*s, "dash")
