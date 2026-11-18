"""Inference precision: fp32, fp16, bf16 or auto, by device."""

from __future__ import annotations

import importlib.util

import numpy as np
import pytest

from bdbvhubrenewal.forecasters import foundation

pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("torch") is None, reason="needs torch"
)


@pytest.mark.parametrize(
    ("device", "capability", "want"),
    [
        ("cuda", (7, 5), "fp16"),  # Turing: fp16 tensor cores, no bf16
        ("cuda", (8, 6), "bf16"),  # Ampere and later
        ("cuda:1", (9, 0), "bf16"),
        ("mps", None, "bf16"),
        ("cpu", None, "fp32"),
    ],
)
def test_auto_takes_each_device_s_half_precision(
    monkeypatch: pytest.MonkeyPatch,
    device: str,
    capability: tuple[int, int] | None,
    want: str,
) -> None:
    import torch

    if capability is not None:
        monkeypatch.setattr(
            torch.cuda, "get_device_capability", lambda *_: capability
        )
    assert foundation.inference_precision(device, "auto") == want


def test_precision_comes_from_the_environment_fp32_by_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("PRECISION", raising=False)
    assert foundation.inference_precision("cuda") == "fp32"
    monkeypatch.setenv("PRECISION", "bf16")
    assert foundation.inference_precision("cpu") == "bf16"
    monkeypatch.setenv("PRECISION", "fp8")
    with pytest.raises(ValueError, match="fp8"):
        foundation.inference_precision("cpu")


class _Pipeline:
    """Records whether each pass ran under autocast, and its dtype."""

    quantiles = (0.1, 0.5, 0.9)

    def __init__(self) -> None:
        self.seen: list = []

    def predict_quantiles(
        self, inputs, prediction_length, quantile_levels, **_kw
    ):
        import torch

        on = torch.is_autocast_enabled("cpu")
        self.seen.append(torch.get_autocast_dtype("cpu") if on else None)
        q = np.zeros((1, prediction_length, len(quantile_levels)))
        return [torch.tensor(q, dtype=torch.float32) for _ in inputs], None


def _group() -> dict:
    return {"target": np.ones((1, 5))}


@pytest.mark.parametrize(
    ("precision", "want"), [("fp32", None), ("bf16", "bfloat16")]
)
def test_predict_runs_under_the_precision(
    monkeypatch: pytest.MonkeyPatch, precision: str, want: str | None
) -> None:
    import torch

    monkeypatch.setenv("DEVICE", "cpu")
    monkeypatch.setenv("PRECISION", precision)
    pipe = _Pipeline()
    foundation._predict(pipe, [_group()], 3, np.asarray(pipe.quantiles))
    assert pipe.seen == [None if want is None else getattr(torch, want)]
