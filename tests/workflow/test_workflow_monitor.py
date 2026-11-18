"""The workflow monitor: what runs where, from each machine's status."""

import json
import sys
import threading
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[2] / "workflow" / "scripts"))

import monitor


def _status(folder, host, updated, alive=True, every=60):
    folder.mkdir(parents=True, exist_ok=True)
    (folder / f"{host}.json").write_text(
        json.dumps(
            {
                "host": host,
                "step": "simulate finetune",
                "alive": alive,
                "updated": updated,
                "every": every,
                "running": [{"rule": "train_segment", "wildcards": "n=500"}],
                "recent": [],
            }
        )
    )


def test_every_machine_s_status_is_listed(tmp_path):
    _status(tmp_path / "status", "host-a", "2026-11-18T12:00:00")
    _status(tmp_path / "status", "host-b", "2026-11-18T12:00:30")
    got = monitor.collect({"ROOT": str(tmp_path)}, now="2026-11-18T12:01:00")
    assert [s["host"] for s in got] == ["host-a", "host-b"]
    assert got[0]["age"] == 60


def test_a_run_that_stops_writing_is_stale(tmp_path):
    # A killed run's log still lists its jobs as started; its status
    # stops being written, so the monitor says it is stale.
    _status(tmp_path / "status", "host-a", "2026-11-18T12:00:00")
    got = monitor.collect({"ROOT": str(tmp_path)}, now="2026-11-18T12:03:00")
    assert got[0]["stale"] is True
    fresh = monitor.collect({"ROOT": str(tmp_path)}, now="2026-11-18T12:01:30")
    assert fresh[0]["stale"] is False


def test_an_idle_machine_is_not_stale(tmp_path):
    _status(tmp_path / "status", "host-a", "2026-11-18T10:00:00", alive=False)
    got = monitor.collect({"ROOT": str(tmp_path)}, now="2026-11-18T12:00:00")
    assert got[0]["stale"] is False


def test_no_status_yet_is_an_empty_list(tmp_path):
    assert monitor.collect({"ROOT": str(tmp_path)}) == []


def test_the_page_and_its_data_are_served_from_local_config(tmp_path):
    _status(tmp_path / "status", "host-a", "2026-11-18T12:00:00")
    cfg = {
        "ROOT": str(tmp_path),
        "MONITOR_HOST": "127.0.0.1",
        "MONITOR_PORT": "0",
        "MONITOR_REFRESH": "17",
    }
    server = monitor.server(cfg)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        url = f"http://127.0.0.1:{server.server_address[1]}"
        page = urllib.request.urlopen(f"{url}/").read().decode()
        data = json.loads(urllib.request.urlopen(f"{url}/status.json").read())
    finally:
        server.shutdown()
    assert "17" in page
    assert "<title>Workflow monitor</title>" in page
    assert data[0]["host"] == "host-a"
