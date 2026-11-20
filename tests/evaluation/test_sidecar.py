"""An offloaded checkpoint's held-out sidecar merged into its settings."""

from nowcastfoundation.sidecar import with_sidecar


def test_the_sidecar_adds_losses_and_never_overrides_settings():
    settings = {"hash": "h", "steps": 10}
    sidecar = {"held_out_loss": 0.2, "steps": 99, "device": "cuda", "kinds": []}
    assert with_sidecar(settings, sidecar) == {
        "hash": "h",
        "steps": 10,
        "held_out_loss": 0.2,
    }
    assert with_sidecar(settings, None) == settings
