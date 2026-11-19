"""The workflow releases a series-bias checkpoint as chronos-slots."""

import importlib.util
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).parents[2]


@pytest.fixture
def publish_checkpoint():
    pytest.importorskip("pandas")
    path = ROOT / "workflow" / "scripts" / "publish_checkpoint.py"
    spec = importlib.util.spec_from_file_location("publish_checkpoint", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize(
    ("bias", "tag"),
    [(True, "chronos-slots-abc"), (False, "chronos-ft-abc")],
)
def test_the_tag_follows_the_series_bias(
    publish_checkpoint, monkeypatch, tmp_path, bias, tag
):
    settings = {"hash": "abc"}
    if bias:
        settings["series_bias"] = {"learning_rate": 1e-3}
    (tmp_path / "settings.json").write_text(json.dumps(settings))
    specs = []
    monkeypatch.setattr(
        publish_checkpoint.fs, "publish", lambda _, spec: specs.append(spec)
    )
    out = tmp_path / "published.json"
    monkeypatch.setattr(sys, "argv", ["x", str(tmp_path), str(out)])
    publish_checkpoint.main()
    assert json.loads(out.read_text())["tag"] == tag


@pytest.mark.parametrize("offloaded", [False, True])
def test_an_offloaded_checkpoint_s_losses_go_with_its_release(
    publish_checkpoint, monkeypatch, tmp_path, offloaded
):
    # WP10: the evaluate rules' sidecar is uploaded beside the settings.
    (tmp_path / "settings.json").write_text(json.dumps({"hash": "abc"}))
    if offloaded:
        (tmp_path / "held-out.json").write_text('{"held_out_loss": 0.2}')
    calls = []
    monkeypatch.setattr(publish_checkpoint.fs, "publish", lambda *_: None)
    monkeypatch.setattr(
        publish_checkpoint.subprocess, "run", lambda cmd, **_: calls.append(cmd)
    )
    out = tmp_path / "published.json"
    monkeypatch.setattr(sys, "argv", ["x", str(tmp_path), str(out)])
    publish_checkpoint.main()
    assert len(calls) == int(offloaded)
    if offloaded:
        assert calls[0][:4] == ["gh", "release", "upload", "chronos-ft-abc"]
        assert str(tmp_path / "held-out.json") in calls[0]
