"""Build a renewal-family model at an origin and initialise it (renewal-speed)."""
import os
from dataclasses import replace

import jax
from numpyro.infer import init_to_median
from numpyro.infer.util import initialize_model

from bdbvhubrenewal.data import load_daily, load_delays, load_target
from bdbvhubrenewal.evaluation.backtest import load_config
from bdbvhubrenewal.evaluation.store import origin_tasks
from bdbvhubrenewal.forecasters import renewal as rn

ORIGIN = os.environ.get("ORIGIN", "2026-11-01")


def setup(name, data_dir="data"):
    cfg = load_config("config/backtest.toml")
    tasks = origin_tasks(load_target(data_dir), load_delays(data_dir), cfg,
                         load_daily(data_dir))
    task = tasks[ORIGIN]
    variant_name, _, loc = name.partition(":")
    variant = rn.VARIANTS[variant_name]
    if variant.per_province:
        task = replace(task, locations=(loc,), mask=())
    pr = rn.engine()
    config = rn.model_config(variant)
    rows = None
    if config.bvd:
        from bdbvhubrenewal.data import bvd
        release = rn.bvd_release(task)
        if release is not None:
            rows = bvd.model_rows(release)
    data = rn.vintage_inputs(task, config, bvd=rows)
    model = pr.build_model(data, config)
    args = pr.model_args(model, data)
    info = initialize_model(
        jax.random.key(0), model.sample, model_kwargs=args,
        init_strategy=init_to_median(num_samples=15),
    )
    return info, data, model, args
