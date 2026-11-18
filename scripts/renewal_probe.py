r"""Profile one renewal fit by stage and dump its sampler geometry.

Runs a stored model's fit at one origin as the fit store does, with the
sampler table of `ci/policy.toml` and any overrides, and writes to
`--out`:

- `timing.json`: wall seconds by stage (imports, data preparation,
  model build, gradient compile and cost, warmup, sampling, forecast,
  store files and writing), and JAX compile seconds.
- `diagnostics.json`: per-chain sampler diagnostics and per-parameter
  R-hat and ESS, as the fit store reports them.
- `geometry.npz`: unconstrained draws by chain, the adapted inverse
  mass matrix and step size of each chain, and `num_steps`.

`--province` fits one province of a per-province variant (`renewalsp`)
on its own, as `RenewalModel.forecast_fit` does, and skips the store
files.

Usage::

    XLA_FLAGS=--xla_force_host_platform_device_count=2 \\
      uv run python scripts/renewal_probe.py --model renewalbvd \\
      --origin 2026-11-15 --out output/probe/bvd-base
"""

from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
from dataclasses import replace
from pathlib import Path
from typing import Any

T0 = time.perf_counter()
STAGES: dict[str, float] = defaultdict(float)
COMPILE: dict[str, float] = defaultdict(float)


def _since(start: float) -> float:
    return time.perf_counter() - start


def _log_rss(path: Path, every: float = 30.0) -> None:
    """Append this process's resident memory (GB) to `path` every `every` s."""
    import threading

    def loop() -> None:
        while True:
            rss = 0
            for line in Path("/proc/self/status").read_text().splitlines():
                if line.startswith("VmRSS:"):
                    rss = int(line.split()[1])
            with path.open("a") as fh:
                fh.write(f"{_since(T0):.0f} {rss / 1e6:.2f}\n")
            time.sleep(every)

    threading.Thread(target=loop, daemon=True).start()


def main() -> int:
    """Run the probe."""
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--model", default="renewalbvd")
    p.add_argument("--origin", default="2026-11-15")
    p.add_argument("--province", default=None)
    p.add_argument("--sampler", default="{}", help="JSON overrides")
    p.add_argument(
        "--model-config", default="{}", help="JSON `ModelConfig` overrides"
    )
    p.add_argument("--data", default="data")
    p.add_argument("--config", default="config/backtest.toml")
    p.add_argument("--out", required=True)
    args = p.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    _log_rss(out / "rss.log")

    import jax
    import jax.monitoring
    import numpy as np

    def on_duration(event: str, seconds: float, **_: Any) -> None:
        if "compile" in event or "lower" in event or "trace" in event:
            COMPILE[event] += seconds

    jax.monitoring.register_event_duration_secs_listener(on_duration)

    import bdbvhubrenewal.model as pr
    from bdbvhubrenewal.evaluation import store
    from bdbvhubrenewal.evaluation.backtest import load_config
    from bdbvhubrenewal.forecasters import renewal

    STAGES["imports"] = _since(T0)
    start = time.perf_counter()
    sampler = {**store.samplers()[args.model], **json.loads(args.sampler)}
    from bdbvhubrenewal.data import load_daily, load_delays, load_target
    from bdbvhubrenewal.data.zones import load_zone_data

    config = load_config(args.config)
    tasks = store.origin_tasks(
        load_target(args.data),
        load_delays(args.data),
        config,
        load_daily(args.data),
        load_zone_data(args.data),
    )
    task = tasks[args.origin]
    STAGES["origin_tasks"] = _since(start)

    geometry: dict[str, Any] = {}

    def run_nuts(model: Any, data: Any, s: dict[str, Any]) -> Any:
        """`renewal.run_nuts`, timed by stage, with extra NUTS options."""
        from jax import random
        from numpyro.infer import MCMC, NUTS, init_to_sample
        from numpyro.infer.util import initialize_model

        STAGES["model_build"] += _since(build_start[0])
        jax.clear_caches()  # type: ignore[no-untyped-call]
        margs = pr.model_args(model, data)
        model.validate_data(
            **{k: v for k, v in margs.items() if k != "population_size"}
        )
        # One gradient: compile, then the cost of each.
        t = time.perf_counter()
        init = initialize_model(
            random.key(0),
            model.sample,
            model_kwargs=margs,
            init_strategy=init_to_sample,
        )
        vg = jax.jit(jax.value_and_grad(init.potential_fn))
        z0 = init.param_info.z
        jax.block_until_ready(vg(z0))
        STAGES["gradient_compile"] += _since(t)
        t = time.perf_counter()
        n_eval = 20
        for _ in range(n_eval):
            jax.block_until_ready(vg(z0))
        geometry["gradient_seconds"] = _since(t) / n_eval
        STAGES["gradient_timing"] += _since(t)
        transforms = {
            k: site["fn"].support
            for k, site in init.model_trace.items()
            if site["type"] == "sample" and not site["is_observed"]
        }

        dense = s.get("dense_mass", False)
        if isinstance(dense, list):
            dense = [tuple(x) for x in dense]
        kernel = NUTS(
            model.sample,
            target_accept_prob=float(s["target_accept"]),
            max_tree_depth=int(s["max_tree_depth"]),
            init_strategy=init_to_sample,
            dense_mass=dense,
            regularize_mass_matrix=bool(s.get("regularize_mass", True)),
        )
        model.mcmc = MCMC(
            kernel,
            num_warmup=int(s["num_warmup"]),
            num_samples=int(s["num_samples"]),
            num_chains=int(s["num_chains"]),
            chain_method=str(s.get("chain_method", "vectorized")),
            progress_bar=False,
        )
        fields = (*pr.model.EXTRA_FIELDS,)
        t = time.perf_counter()
        model.mcmc.warmup(
            random.key(int(s["seed"])),
            extra_fields=fields,
            collect_warmup=True,
            **margs,
        )
        jax.block_until_ready(model.mcmc.post_warmup_state)
        STAGES["warmup"] += _since(t)
        warm = model.mcmc.get_extra_fields(group_by_chain=True)
        geometry["warmup_num_steps"] = np.asarray(warm["num_steps"])
        t = time.perf_counter()
        model.mcmc.run(
            model.mcmc.post_warmup_state.rng_key,
            extra_fields=fields,
            **margs,
        )
        np.asarray(next(iter(model.mcmc.get_samples().values())))
        STAGES["sampling"] += _since(t)
        last = model.mcmc.last_state
        imm = last.adapt_state.inverse_mass_matrix
        geometry["imm"] = {"|".join(k): np.asarray(v) for k, v in imm.items()}
        geometry["step_size"] = np.asarray(last.adapt_state.step_size)
        from numpyro.distributions.transforms import biject_to

        by_chain = model.mcmc.get_samples(group_by_chain=True)
        geometry["unconstrained"] = {
            k: np.asarray(biject_to(transforms[k]).inv(by_chain[k]))
            for k in last.z
            if k in by_chain and k in transforms
        }
        geometry["num_steps"] = np.asarray(
            model.mcmc.get_extra_fields(group_by_chain=True)["num_steps"]
        )
        return model

    build_start = [0.0]
    orig_build = pr.build_model

    def build_model(*a: Any, **k: Any) -> Any:
        STAGES["vintage_inputs"] += _since(run_start[0])
        build_start[0] = time.perf_counter()
        return orig_build(*a, **k)

    orig_forecast = pr.forecast_trajectories

    def forecast_trajectories(*a: Any, **k: Any) -> Any:
        t = time.perf_counter()
        res = orig_forecast(*a, **k)
        STAGES["forecast_trajectories"] += _since(t)
        return res

    overrides = json.loads(args.model_config)
    if overrides:
        orig_config = renewal.model_config

        def model_config(variant: Any) -> Any:
            return replace(orig_config(variant), **overrides)

        renewal.model_config = model_config  # type: ignore[assignment]
    renewal.run_nuts = run_nuts  # type: ignore[assignment]
    pr.build_model = build_model  # type: ignore[assignment]
    pr.forecast_trajectories = forecast_trajectories  # type: ignore[assignment]
    run_start = [0.0]
    orig_run = renewal.RenewalModel.run

    def run(self: Any, t: Any) -> Any:
        run_start[0] = time.perf_counter()
        start = time.perf_counter()
        res = orig_run(self, t)
        STAGES["run_total"] += _since(start)
        return res

    renewal.RenewalModel.run = run  # type: ignore[method-assign]

    start = time.perf_counter()
    if args.province:
        model = store.build_model(args.model, {args.model: sampler})
        one = replace(task, locations=(args.province,), mask=())
        res = model.run(one)
        trace, extra = res.trace, res.extra
        STAGES["fit_total"] = _since(start)
    else:
        code = store.code_digest(args.model)
        stored = store.fit(args.model, task, {args.model: sampler}, code)
        STAGES["fit_total"] = _since(start)
        if stored.meta["status"] != "ok":
            print(stored.meta.get("error"))
            return 1
        t = time.perf_counter()
        store.write_fit(out / "store", stored)
        STAGES["write_fit"] = _since(t)
        diag = json.loads(
            __import__("base64").b64decode(stored.files["diagnostics.json"])
        )
        (out / "store-diagnostics.json").write_text(json.dumps(diag))
        trace, extra = None, None
    depth = int(sampler["max_tree_depth"])
    if trace is not None:
        chains = renewal.chain_diagnostics(extra, depth)
        params = renewal.parameter_diagnostics(trace)
    else:
        chains, params = diag["chains"], diag["parameters"]
    STAGES["total"] = _since(T0)
    timing = {
        "stages": dict(STAGES),
        "jax_compile_events": dict(COMPILE),
        "gradient_seconds": geometry.get("gradient_seconds"),
        "sampler": sampler,
        "model": args.model,
        "origin": args.origin,
        "province": args.province,
        "model_config": json.loads(args.model_config),
    }
    (out / "timing.json").write_text(json.dumps(timing, indent=1))
    (out / "diagnostics.json").write_text(
        json.dumps({"chains": chains, "parameters": params}, indent=1)
    )
    flat: dict[str, Any] = {
        "step_size": geometry["step_size"],
        "num_steps": geometry["num_steps"],
        "warmup_num_steps": geometry["warmup_num_steps"],
    }
    for k, v in geometry["imm"].items():
        flat[f"imm::{k}"] = v
    for k, v in geometry["unconstrained"].items():
        flat[f"z::{k}"] = v
    np.savez_compressed(out / "geometry.npz", **flat)
    ess_tail = [q["ess_tail"] for q in params if q["ess_tail"] is not None]
    hits = sum(c.get("tree_depth_hits", 0) for c in chains)
    draws = sum(c["draws"] for c in chains)
    print(
        json.dumps(
            {
                "min_ess_tail": min(ess_tail),
                "depth_share": hits / draws,
                "divergences": sum(c["divergences"] for c in chains),
                "step": [c.get("step_size") for c in chains],
                "stages": {k: round(v) for k, v in STAGES.items()},
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
