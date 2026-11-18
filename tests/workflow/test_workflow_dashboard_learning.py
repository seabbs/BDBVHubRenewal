"""What fine-tuning learns: held-out loss by source and level, LoRA size."""

import json
import math
import struct

import pytest
from flowlib.dashboard import learning, runs

HISTORY = [
    {
        "step": 0,
        "held_out_loss": 0.4,
        "by_level": {"leaf": 0.2, "national": 0.4},
        "by_source": {"daily": 0.2, "weekly": 0.8},
    },
    {"step": 10, "loss": 0.5},
    {"step": 100, "eval_loss": 0.39},
    {
        "step": 200,
        "eval_loss": 0.38,
        "by_source": {"daily": 0.1, "weekly": 0.6},
    },
]


def test_loss_by_level_is_relative_to_the_start_at_each_checkpoint():
    cks = [
        ("n250", {"steps": 1500, "held_out_loss_by_level": {"leaf": 0.1}}),
        ("n500", {"steps": 3000, "held_out_loss_by_level": {"leaf": 0.05}}),
    ]
    assert learning.by_level(HISTORY, cks) == {
        "leaf": [(0, 1.0), (1500, 0.5), (3000, 0.25)],
        "national": [(0, 1.0)],
    }


def test_no_start_gives_no_curves():
    assert learning.by_level([{"step": 10, "loss": 0.5}], []) == {}
    assert learning.by_level([], []) == {}


def _safetensors(path, tensors):
    header, data = {}, b""
    for name, rows in tensors.items():
        flat = [x for r in rows for x in r]
        raw = struct.pack(f"<{len(flat)}f", *flat)
        header[name] = {
            "dtype": "F32",
            "shape": [len(rows), len(rows[0])],
            "data_offsets": [len(data), len(data) + len(raw)],
        }
        data += raw
    head = json.dumps(header).encode()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(len(head).to_bytes(8, "little") + head + data)


def _lora(layer, part, module):
    stem = (
        f"base_model.model.encoder.block.{layer}.layer.{part}."
        f"self_attention.{module}"
    )
    return f"{stem}.lora_A.weight", f"{stem}.lora_B.weight"


def test_lora_update_size_by_layer(tmp_path):
    ckpt = tmp_path / "finetuned-ckpt"
    a0, b0 = _lora(0, 0, "q")
    a1, b1 = _lora(0, 1, "v")
    a2, b2 = _lora(1, 0, "q")
    _safetensors(
        ckpt / "adapter_model.safetensors",
        {
            # B @ A = [[1, 0], [0, 0]]: Frobenius norm 1.
            a0: [[1.0, 0.0]],
            b0: [[1.0], [0.0]],
            # B @ A = [[0, 2], [0, 0]]: norm 2.
            a1: [[0.0, 1.0]],
            b1: [[2.0], [0.0]],
            # Zero update.
            a2: [[0.0, 0.0]],
            b2: [[1.0], [1.0]],
        },
    )
    (ckpt / "adapter_config.json").write_text('{"r": 8, "lora_alpha": 16}')
    # Scaled by alpha / r = 2; a layer's modules add in quadrature.
    got = learning.lora_size(ckpt)
    assert got[0] == (0, pytest.approx(2 * math.sqrt(5)))
    assert got[1] == (1, 0.0)


def test_lora_size_of_a_run_by_checkpoint(tmp_path):
    folder = tmp_path / "run"
    a, b = _lora(0, 0, "q")
    for name in ("n250", "n500"):
        d = folder / "train" / name
        _safetensors(
            d / "finetuned-ckpt" / "adapter_model.safetensors",
            {a: [[1.0, 0.0]], b: [[1.0], [0.0]]},
        )
        (d / "finetuned-ckpt" / "adapter_config.json").write_text(
            '{"r": 1, "lora_alpha": 1}'
        )
        (d / "settings.json").write_text('{"steps": 10}')
    run = runs.Run("r5", "r5", folder=folder)
    assert learning.run_lora(run) == {
        "n250": [(0, 1.0)],
        "n500": [(0, 1.0)],
    }


def test_section_has_three_panels(tmp_path):
    folder = tmp_path / "run"
    (folder / "train").mkdir(parents=True)
    (folder / "train" / "history.json").write_text(json.dumps(HISTORY))
    run = runs.Run("r5", "r5", folder=folder, live=True)
    out = learning.section(run)
    assert out.count("<div class=card>") == 3
    assert "by encoder layer" in out


def test_lora_update_size_by_kind(tmp_path):
    folder = tmp_path / "run"
    t_a, t_b = _lora(0, 0, "q")
    g_a, g_b = _lora(0, 1, "q")
    out = "base_model.model.output_patch_embedding.output_layer"
    for name, k in (("n250", 1.0), ("n500", 2.0)):
        d = folder / "train" / name
        _safetensors(
            d / "finetuned-ckpt" / "adapter_model.safetensors",
            {
                t_a: [[k, 0.0]],
                t_b: [[1.0], [0.0]],
                g_a: [[0.0, 3.0]],
                g_b: [[1.0], [0.0]],
                f"{out}.lora_A.weight": [[4.0, 0.0]],
                f"{out}.lora_B.weight": [[1.0], [0.0]],
            },
        )
        (d / "finetuned-ckpt" / "adapter_config.json").write_text(
            '{"r": 1, "lora_alpha": 1}'
        )
        (d / "settings.json").write_text(
            json.dumps({"steps": 10, "n_outbreaks": int(name[1:])})
        )
    run = runs.Run("r5", "r5", folder=folder)
    assert learning.run_kinds(run) == {
        "time attention": [(250, 1.0), (500, 2.0)],
        "group attention": [(250, 3.0), (500, 3.0)],
        "output layer": [(250, 4.0), (500, 4.0)],
    }
