"""Tests of the one torch device, threads and batch setting."""

from __future__ import annotations

import sys
import types

import pytest

from bdbvhubrenewal import torchdevice as devices


def _torch(cuda: bool, mps: bool, calls: list | None = None):
    """A stand-in torch reporting which devices it has."""
    return types.SimpleNamespace(
        cuda=types.SimpleNamespace(is_available=lambda: cuda),
        backends=types.SimpleNamespace(
            mps=types.SimpleNamespace(is_available=lambda: mps)
        ),
        set_num_threads=lambda n: (calls if calls is not None else []).append(
            n
        ),
    )


@pytest.mark.parametrize(
    ("cuda", "mps", "want"),
    [(True, True, "cuda"), (False, True, "mps"), (False, False, "cpu")],
)
def test_auto_takes_the_fastest_device_there_is(monkeypatch, cuda, mps, want):
    monkeypatch.setitem(sys.modules, "torch", _torch(cuda, mps))
    monkeypatch.delenv("DEVICE", raising=False)
    assert devices.pick_device() == want
    assert devices.pick_device("auto") == want


def test_the_device_setting_wins_over_auto(monkeypatch):
    monkeypatch.setitem(sys.modules, "torch", _torch(True, False))
    monkeypatch.setenv("DEVICE", "cpu")
    assert devices.pick_device() == "cpu"
    # An explicit device wins over the setting.
    assert devices.pick_device("cuda") == "cuda"


def test_torch_threads_come_from_the_setting(monkeypatch):
    calls: list[int] = []
    monkeypatch.setitem(sys.modules, "torch", _torch(False, False, calls))
    monkeypatch.delenv("TORCH_THREADS", raising=False)
    assert devices.torch_threads() is None
    devices.apply_threads()
    assert calls == []
    monkeypatch.setenv("TORCH_THREADS", "4")
    assert devices.torch_threads() == 4
    devices.apply_threads()
    assert calls == [4]


def test_the_chronos_batch_comes_from_the_setting(monkeypatch):
    monkeypatch.delenv("CHRONOS_BATCH", raising=False)
    assert devices.chronos_batch() == devices.CHRONOS_BATCH
    monkeypatch.setenv("CHRONOS_BATCH", "64")
    assert devices.chronos_batch() == 64


@pytest.mark.parametrize("name", ["TORCH_THREADS", "CHRONOS_BATCH"])
def test_a_bad_count_is_refused(monkeypatch, name):
    monkeypatch.setenv(name, "0")
    get = (
        devices.torch_threads
        if name == "TORCH_THREADS"
        else devices.chronos_batch
    )
    with pytest.raises(ValueError, match=name):
        get()


def test_chronos_loads_on_the_resolved_device(monkeypatch):
    pytest.importorskip("chronos")
    from bdbvhubrenewal.forecasters import foundation

    seen = {}

    class Pipeline:
        @classmethod
        def from_pretrained(cls, *_args, device_map=None, **_kw):
            seen["device"] = device_map
            return "pipeline"

    import chronos

    monkeypatch.setattr(chronos, "Chronos2Pipeline", Pipeline)
    monkeypatch.setattr(devices, "pick_device", lambda *_: "cpu")
    foundation._chronos.cache_clear()
    try:
        assert foundation._chronos(None) == "pipeline"
    finally:
        foundation._chronos.cache_clear()
    assert seen["device"] == "cpu"


def test_the_local_config_knows_the_settings():
    sys.path.insert(0, "workflow")
    from flowlib import localconfig

    keys = dict(localconfig.DEFAULTS)
    assert keys["DEVICE"] == "auto"
    assert "TORCH_THREADS" in keys
    assert keys["CHRONOS_BATCH"] == str(devices.CHRONOS_BATCH)
