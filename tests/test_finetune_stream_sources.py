"""Held-out loss of the fine-tune by source."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))
import finetune_stream as fs


def test_held_out_loss_is_reported_by_source(monkeypatch):
    blocks = [
        {"source": s, "value": v}
        for s, v in [
            ("weekly", 1.0),
            ("daily", 2.0),
            ("weekly", 3.0),
            ("3day", 4.0),
            ("weekly", 5.0),
        ]
    ]

    def mean_value(_model, blocks, _batch_size):
        return sum(b["value"] for b in blocks) / len(blocks)

    monkeypatch.setattr(fs, "held_out_loss", mean_value)
    assert fs.held_out_by_source(None, blocks, 8) == {
        "3day": 4.0,
        "daily": 2.0,
        "weekly": 3.0,
    }
    # The first `n` blocks of each source.
    assert fs.held_out_by_source(None, blocks, 8, n=2)["weekly"] == 2.0


def test_compare_keeps_the_logged_curve_of_the_sources_compared(tmp_path):
    import json

    import finetune_compare_runs as cr

    (tmp_path / "train").mkdir()
    (tmp_path / "train" / "history.json").write_text(
        json.dumps(
            [
                {"step": 0, "by_source": {"weekly": 0.5, "3day": 0.9}},
                {"step": 10, "loss": 0.3},
                {"step": 100, "by_source": {"weekly": 0.4, "daily": 0.2}},
            ]
        )
    )
    rows = cr.curve_rows("slots", tmp_path, ("weekly", "daily"))
    assert [(r["step"], r["source"], r["loss"]) for r in rows] == [
        (0, "weekly", 0.5),
        (100, "weekly", 0.4),
        (100, "daily", 0.2),
    ]
