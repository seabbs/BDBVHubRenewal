"""Held-out loss of the fine-tune by source."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))
import finetune_stream as fs  # noqa: E402


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

    def mean_value(model, blocks, batch_size):
        return sum(b["value"] for b in blocks) / len(blocks)

    monkeypatch.setattr(fs, "held_out_loss", mean_value)
    assert fs.held_out_by_source(None, blocks, 8) == {
        "3day": 4.0,
        "daily": 2.0,
        "weekly": 3.0,
    }
    # The first `n` blocks of each source.
    assert fs.held_out_by_source(None, blocks, 8, n=2)["weekly"] == 2.0
