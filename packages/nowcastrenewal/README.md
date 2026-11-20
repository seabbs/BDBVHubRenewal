# nowcastrenewal

Renewal-model nowcasts and forecasts.

- `model`: the PyRenew renewal model (JAX, NumPyro), with provinces or health zones as patches.
- `forecaster`: the renewal variants as forecasters, their fitting, forecasts and diagnostics.
- `zones`: the renewal variants fitted on health zones.
- `bvd`: the BVDOutbreakSize releases the `bvd` variants read.

```sh
uv add "nowcastrenewal[model]"
```
