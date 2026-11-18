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
