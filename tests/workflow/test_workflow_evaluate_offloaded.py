"""The runner of the evaluation rules (evaluate_offloaded.py)."""

import importlib.util
import json
from pathlib import Path

import pytest

pytest.importorskip("torch")
pytest.importorskip("chronos")

ROOT = Path(__file__).parents[2]


def runner():
    spec = importlib.util.spec_from_file_location(
        "evaluate_offloaded", ROOT / "workflow/scripts/evaluate_offloaded.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def snapshot(train, step, kinds):
    where = train / "evals" / f"step-{step:07d}"
    (where / "adapter").mkdir(parents=True)
    (where / "request.json").write_text(
        json.dumps({"step": step, "kinds": kinds, "n_validation": 2})
    )


@pytest.fixture
def fake(monkeypatch):
    mod = runner()
    calls = {"loads": 0, "evals": []}

    def load_base(_device):
        calls["loads"] += 1
        return object()

    def evaluate(adapter, _base, _blocks, kinds, _batch, real=()):
        calls["evals"].append((Path(adapter), list(kinds), len(real)))
        if "start" in kinds:
            return {"held_out_loss": 0.4}
        if "checkpoint" in kinds:
            return {"held_out_loss": 0.25, "held_out_loss_by_level": {}}
        return {"eval_loss": 0.3}

    monkeypatch.setattr(mod.fe, "load_base", load_base)
    monkeypatch.setattr(mod.fe, "evaluate", evaluate)
    monkeypatch.setattr(mod.fe, "load_blocks", lambda *_a, **_k: [{}, {}])
    monkeypatch.setattr(mod, "fresh", lambda base, _device: base)
    monkeypatch.setattr(mod.series_bias, "attach_for", lambda *_a: None)
    return mod, calls


def checkpoint_at(train, n, steps):
    where = train / f"n{n}"
    (where / "finetuned-ckpt").mkdir(parents=True)
    (where / "settings.json").write_text(
        json.dumps({"n_validation": 2, "n_outbreaks": n, "steps": steps})
    )
    return where


def test_steps_scores_each_of_its_segment_s_snapshots_once(tmp_path, fake):
    mod, calls = fake
    train = tmp_path / "train"
    for step, kinds in ((0, ["start"]), (200, ["monitor"]), (400, ["monitor"])):
        snapshot(train, step, kinds)
    first = checkpoint_at(train, 250, 200)
    second = checkpoint_at(train, 500, 400)
    mod.main(["steps", str(first), "--device", "cpu"])
    assert calls["loads"] == 1
    assert [c[1] for c in calls["evals"]] == [["start"], ["monitor"]]
    got = json.loads((train / "evals" / "step-0000200.json").read_text())
    assert got["step"] == 200
    assert got["eval_loss"] == 0.3
    assert not (train / "evals" / "step-0000400.json").exists()
    mod.main(["steps", str(second), "--prev", str(first), "--device", "cpu"])
    assert len(calls["evals"]) == 3
    mod.main(["steps", str(second), "--prev", str(first), "--device", "cpu"])
    assert len(calls["evals"]) == 3


def test_checkpoint_writes_the_sidecar_with_the_start_loss(tmp_path, fake):
    mod, calls = fake
    train = tmp_path / "train"
    (train / "n500" / "finetuned-ckpt").mkdir(parents=True)
    (train / "n500" / "settings.json").write_text(
        json.dumps({"n_validation": 2, "n_outbreaks": 500, "steps": 31})
    )
    (train / "evals").mkdir()
    (train / "evals" / "step-0000000.json").write_text(
        json.dumps({"held_out_loss": 0.4})
    )
    out = train / "n500" / "held-out.json"
    mod.main(["checkpoint", str(train / "n500"), "--device", "cpu"])
    got = json.loads(out.read_text())
    assert got["held_out_loss"] == 0.25
    assert got["held_out_loss_start"] == 0.4
    assert calls["evals"][-1][1] == ["checkpoint"]
    assert calls["evals"][-1][0] == train / "n500" / "finetuned-ckpt"


def test_a_checkpoint_with_no_base_loss_fails_clearly(tmp_path, fake):
    mod, _calls = fake
    where = checkpoint_at(tmp_path / "train", 500, 31)
    with pytest.raises(SystemExit, match="base Chronos-2 loss"):
        mod.main(["checkpoint", str(where), "--device", "cpu"])
