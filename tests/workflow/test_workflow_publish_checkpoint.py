"""Publishing a checkpoint (workflow/scripts/publish_checkpoint.py)."""

import importlib.util
import json
from pathlib import Path

import pytest

pytest.importorskip("torch")

ROOT = Path(__file__).parents[2]


def script():
    spec = importlib.util.spec_from_file_location(
        "publish_checkpoint", ROOT / "workflow/scripts/publish_checkpoint.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.mark.parametrize("offloaded", [False, True])
def test_an_offloaded_checkpoint_s_losses_go_with_its_release(
    tmp_path, monkeypatch, offloaded
):
    mod = script()
    where = tmp_path / "n500"
    where.mkdir()
    (where / "settings.json").write_text(
        json.dumps({"hash": "abc", "series_bias": {"module": "x"}})
    )
    if offloaded:
        (where / "held-out.json").write_text('{"held_out_loss": 0.2}')
    calls = []
    monkeypatch.setattr(mod.fs, "publish", lambda _w, spec: calls.append(spec))
    monkeypatch.setattr(
        mod.subprocess, "run", lambda cmd, **_k: calls.append(cmd)
    )
    mod.sys.argv = ["x", str(where), str(tmp_path / "out.json")]
    mod.main()
    assert calls[0] == "slots-abc"
    uploads = [c for c in calls[1:] if "upload" in c]
    assert len(uploads) == int(offloaded)
    if offloaded:
        assert "chronos-slots-abc" in uploads[0]
        assert str(where / "held-out.json") in uploads[0]
