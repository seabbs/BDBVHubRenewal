# nowcastutils

Shared pieces the nowcasting packages build on.

- `releases`: release cut-offs, cut-off lags and release pairs.
- `snapshots`: daily release snapshots as arrays on the onset clock.
- `hierarchy`: location hierarchies read from ISO 3166-2 style codes.
- `dates`: epidemiological weeks and dates.
- `base`: the forecast task, the sample table and the model registry.
- `forecast`: a model's forecast record.
- `files`: table writing.
- `devices`: compute device settings.
- `trajectories`: coherent sample trajectories from quantile forecasts.

```sh
uv add nowcastutils
```
