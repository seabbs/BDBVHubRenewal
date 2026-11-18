"""Chronos-2 loads on the device the workflow resolves (`DEVICE`)."""

from __future__ import annotations

import pytest

pytest.importorskip("chronos")


@pytest.fixture
def loaded(monkeypatch):
    import chronos

    from bdbvhubrenewal.forecasters import foundation

    seen: dict[str, str] = {}

    class Pipeline:
        @classmethod
        def from_pretrained(cls, *_args, device_map=None, **_kw):
            seen["device"] = device_map
            return "pipeline"

    monkeypatch.setattr(chronos, "Chronos2Pipeline", Pipeline)
    foundation._chronos.cache_clear()
    yield foundation, seen
    foundation._chronos.cache_clear()


def test_chronos_loads_on_the_device_setting(loaded, monkeypatch):
    foundation, seen = loaded
    # A device auto-detection never gives, so the setting must be read.
    monkeypatch.setenv("DEVICE", "cuda:1")
    assert foundation._chronos(None) == "pipeline"
    assert seen["device"] == "cuda:1"


def test_auto_or_unset_detects_the_device(loaded, monkeypatch):
    import torch

    foundation, seen = loaded
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)
    for value in ("auto", None):
        if value is None:
            monkeypatch.delenv("DEVICE", raising=False)
        else:
            monkeypatch.setenv("DEVICE", value)
        foundation._chronos.cache_clear()
        foundation._chronos(None)
        assert seen["device"] == "cpu"


def test_one_device_resolver_for_scoring_and_training():
    from bdbvhubrenewal.forecasters import foundation

    assert foundation.torch_device("cuda:1") == "cuda:1"
    scripts = __import__("pathlib").Path(__file__).parents[2] / "scripts"
    for name in ("finetune_stream.py", "finetune_speed.py"):
        assert "def pick_device" not in (scripts / name).read_text()


def test_chronos_batch_and_threads_are_machine_settings(monkeypatch):
    import torch

    from bdbvhubrenewal.forecasters import foundation

    seen = {}

    class Pipeline:
        def predict_quantiles(self, inputs, batch_size=None, **_kw):
            seen["batch"] = batch_size
            return [None] * len(inputs), None

    monkeypatch.setenv("CHRONOS_BATCH", "64")
    monkeypatch.setenv("TORCH_THREADS", "3")
    monkeypatch.setattr(torch, "set_num_threads", lambda n: seen.update(n=n))
    foundation._predict(
        Pipeline(), [{"target": [[1.0]]}], 2, __import__("numpy").array([0.5])
    )
    assert seen["batch"] == 64
    foundation.apply_threads()
    assert seen["n"] == 3
