"""Attention bias by relation, from checkpoints and the run's history."""

import json
import struct

from flowlib.dashboard import bias, runs


def _safetensors(path, tensors):
    """Write 2-D float32 tensors as a safetensors file (standard library)."""
    header, data = {}, b""
    for name, rows in tensors.items():
        raw = struct.pack(
            f"<{sum(len(r) for r in rows)}f", *[x for r in rows for x in r]
        )
        header[name] = {
            "dtype": "F32",
            "shape": [len(rows), len(rows[0])],
            "data_offsets": [len(data), len(data) + len(raw)],
        }
        data += raw
    head = json.dumps(header).encode()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(len(head).to_bytes(8, "little") + head + data)


NAMES = {
    k: f"base.model.series_bias.modules_to_save.default.{k}"
    for k in ("self_", "same", "up", "down")
}


def test_relations_are_read_from_a_safetensors_adapter(tmp_path):
    f = tmp_path / "a.safetensors"
    _safetensors(
        f,
        {
            NAMES["same"]: [[1.0, 2.0], [3.0, 4.0]],
            NAMES["up"]: [[0.5, 0.5], [0.5, 0.5]],
            "base.model.lora_A.weight": [[9.0, 9.0]],
        },
    )
    got = bias.read_bias(f)
    assert set(got) == {"same", "up"}
    assert got["same"] == [[1.0, 2.0], [3.0, 4.0]]


def test_stats_are_mean_min_and_max_over_layers_and_heads():
    assert bias.summary([[1.0, 2.0], [3.0, 6.0]]) == {
        "mean": 3.0,
        "min": 1.0,
        "max": 6.0,
    }


def test_a_run_s_bias_from_its_checkpoints_and_history(tmp_path):
    folder = tmp_path / "run"
    for n, step, v in ((250, 1140, 1.0), (500, 2280, 2.0)):
        d = folder / "train" / f"n{n}"
        _safetensors(
            d / "finetuned-ckpt" / "adapter_model.safetensors",
            {NAMES["same"]: [[v, v], [v, v]]},
        )
        (d / "settings.json").write_text(json.dumps({"steps": step}))
    (folder / "train" / "history.json").write_text(
        json.dumps(
            [
                {
                    "step": 600,
                    "bias": {"same": {"values": [[1.5, 1.5], [1.5, 1.5]]}},
                }
            ]
        )
    )
    run = runs.make_run("r", str(folder), live=True)
    got = bias.run_bias(run, step=None)
    assert [p[0] for p in got["pts"]["same"]] == [600, 1140, 2280]
    assert got["pts"]["same"][0][1] == 1.5
    assert got["latest"]["step"] == 2280
    assert got["latest"]["same"] == [[2.0, 2.0], [2.0, 2.0]]


def test_a_run_without_bias_has_none(tmp_path):
    run = runs.make_run("r", str(tmp_path), live=True)
    assert bias.run_bias(run, step=None) is None


def test_a_torch_file_without_torch_is_skipped(tmp_path, monkeypatch):
    f = tmp_path / "adapter.pt"
    f.write_bytes(b"not a real torch file")
    monkeypatch.setattr(bias, "_torch", lambda: None)
    assert bias.read_bias(f) == {}


def test_the_page_payload_lists_each_run_with_bias(tmp_path):
    folder = tmp_path / "run"
    d = folder / "train" / "n250"
    _safetensors(
        d / "finetuned-ckpt" / "adapter_model.safetensors",
        {NAMES["down"]: [[1.0]]},
    )
    (d / "settings.json").write_text(json.dumps({"steps": 10}))
    rs = runs.load_runs([], f"r={folder},live;x=x-", [])
    got = bias.payload(rs, live_step=None)
    assert [r["key"] for r in got["runs"]] == ["r"]
    assert [p["key"] for p in got["pairs"]] == ["self", "same", "up", "down"]
    assert got["runs"][0]["pts"]["down"] == [[10, 1.0, 1.0, 1.0]]
