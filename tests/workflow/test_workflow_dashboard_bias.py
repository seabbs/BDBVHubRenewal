"""Attention bias by relation, from checkpoints and the run's history."""

import json
import struct

import pytest
from flowlib.dashboard import bias, logs, runs, training


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
        (d / "settings.json").write_text(
            json.dumps({"steps": step, "n_outbreaks": n})
        )
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
    pairs = training.anchors(logs.checkpoints(folder), {})
    got = bias.run_bias(
        run, step=None, sims=lambda s: training.sims_at(pairs, s)
    )
    # Against simulations trained: step 600 is 600 / 1140 of n250.
    assert [p[0] for p in got["pts"]["same"]] == [131.58, 250, 500]
    assert got["pts"]["same"][0][1]["mean"] == 1.5
    assert got["latest"]["n"] == 500
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
    assert got["runs"][0]["pts"]["down"] == [[250, 1.0, 1.0, 1.0]]


LINE = (
    "2026-11-16 16:50:01 [train] step {step}: loss 0.5, held-out (x) 0.4, "
    "bias self_ {m:+.3f} [{lo:+.3f}, {hi:+.3f}], same {m:+.3f} "
    "[{lo:+.3f}, {hi:+.3f}], 9 s\n"
)
CHUNK = (
    "2026-11-16 16:40:00 [train] outbreaks {a}-{b}: 9 blocks, 99 series, "
    "{n} steps\n"
)


def _logged_run(folder, launcher, segment, steps_a_sim):
    """A run with an old launcher log and a segment log, as r5 has."""
    (folder / "train").mkdir(parents=True)
    text = CHUNK.format(a=1, b=100, n=100 * steps_a_sim)
    for step, m in launcher:
        text += LINE.format(step=step, m=m, lo=m - 0.1, hi=m + 0.1)
    (folder / "train-launcher.log").write_text(text)
    text = CHUNK.format(a=101, b=200, n=100 * steps_a_sim)
    for step, m in segment:
        text += LINE.format(step=step, m=m, lo=m - 0.1, hi=m + 0.1)
    (folder / "train" / "n200.log").write_text(text)


def test_each_run_s_bias_is_its_own_logged_values_over_simulations(tmp_path):
    from flowlib.dashboard import page

    # r5-like: 4 steps a simulation; r6-like: 10; slots: only a
    # published checkpoint's settings, no logs and no adapter.
    _logged_run(tmp_path / "r5", [(200, 0.1), (400, 0.2)], [(600, 0.3)], 4)
    _logged_run(tmp_path / "r6", [(500, -0.5)], [(1500, -0.7)], 10)
    slots = tmp_path / "slots" / "train" / "n250"
    slots.mkdir(parents=True)
    (slots / "settings.json").write_text(
        json.dumps({"steps": 900, "n_outbreaks": 250})
    )
    rs = runs.load_runs(
        [],
        f"r5={tmp_path / 'r5'};r6={tmp_path / 'r6'},live;"
        f"slots={tmp_path / 'slots'}",
        [],
    )
    ctx = page.Context(
        runs=rs,
        references=[],
        source=None,
        rule="",
        outbreaks=200,
        learning_rate=1e-5,
        prefixes=[200],
    )
    got = bias.payload(
        rs,
        None,
        {r.key: ctx.log(r) for r in rs},
        {r.key: ctx.sims(r) for r in rs},
    )
    by = {r["key"]: r["pts"] for r in got["runs"]}
    # The slots run has no bias of its own, so it is missing, not borrowed.
    assert set(by) == {"r5", "r6"}
    want = {
        "r5": [[50, 0.1], [100, 0.2], [150, 0.3]],
        "r6": [[50, -0.5], [150, -0.7]],
    }
    for key, pts in want.items():
        for rel in ("self", "same"):
            assert [p[0] for p in by[key][rel]] == [x for x, _ in pts]
            assert [p[1] for p in by[key][rel]] == pytest.approx(
                [m for _, m in pts]
            )
            assert [p[2] for p in by[key][rel]] == pytest.approx(
                [m - 0.1 for _, m in pts]
            )
