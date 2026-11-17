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
