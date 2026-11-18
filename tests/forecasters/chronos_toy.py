"""A tiny Chronos-2 and a toy ragged block for the series bias tests."""

import numpy as np
import torch

N_PLACES, LAG, T, H = 3, 2, 48, 16


HIER = ("CD", "R2", "R1", "R2-P01", "R1-P03", "A1", "P05", "P05-A1")
"""Places of a training block over several levels (the trainer's names)."""


def tiny_model(seed: int = 0):
    """A small, randomly initialised Chronos-2 with no dropout."""
    from chronos.chronos2.config import Chronos2CoreConfig
    from chronos.chronos2.model import Chronos2Model

    torch.manual_seed(seed)
    config = Chronos2CoreConfig(
        d_model=32,
        d_kv=8,
        d_ff=64,
        num_layers=2,
        num_heads=4,
        dropout_rate=0.0,
        rope_theta=10000.0,
        dense_act_fn="relu",
        feed_forward_proj="relu",
        chronos_config={
            "context_length": 256,
            "input_patch_size": 16,
            "input_patch_stride": 16,
            "max_output_patches": 4,
            "output_patch_size": 16,
            "quantiles": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
            "use_arcsinh": True,
            "use_reg_token": True,
        },
    )
    return Chronos2Model(config).eval()


def toy_block(seed: int = 0) -> dict:
    """A ragged block: three places' counts, then each place's two lags."""
    rng = np.random.default_rng(seed)
    level = np.array([5.0, 50.0, 500.0])[:, None]
    target = rng.poisson(level, (N_PLACES, T + H)).astype(np.float32)
    lags = np.stack(
        [target[p] * share for p in range(N_PLACES) for share in (0.4, 0.8)]
    ).astype(np.float32)
    lags[:, T + 4 :] = np.nan
    context = np.concatenate([target, lags])
    return {
        "context": context,
        "n_targets": N_PLACES,
        "n_covariates": N_PLACES * LAG,
        "n_future_covariates": N_PLACES * LAG,
        "prediction_length": H,
    }


def swapped(block: dict) -> dict:
    """The block with the lag series of places 0 and 2 swapped."""
    x = block["context"].copy()
    rows = x[N_PLACES:].reshape(N_PLACES, LAG, -1)
    x[N_PLACES:] = rows[[2, 1, 0]].reshape(N_PLACES * LAG, -1)
    return {**block, "context": x}


def batch_of(block: dict) -> dict:
    """Chronos-2's validation batch of a block, as the trainer builds it."""
    from chronos.chronos2.dataset import Chronos2Dataset, DatasetMode

    prepared = {
        **block,
        "context": torch.from_numpy(block["context"]),
        "future_covariates": torch.from_numpy(block["context"][:, T:]),
    }
    data = Chronos2Dataset(
        inputs=[prepared],
        context_length=T,
        prediction_length=H,
        batch_size=1000,
        output_patch_size=16,
        mode=DatasetMode.VALIDATION,
    )
    return next(iter(data))
