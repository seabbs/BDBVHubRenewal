# nowcastbaselines

Baseline nowcasts and forecasts.

- `randomwalk`: the random walk forecaster and its nowcasting variant.
- `chain`, `walk`, `obs`: the chain-ladder delay fit, random walks on the log scale, and the observation model.
- `baselinenowcast`: a JAX port of the baselinenowcast R package.
- `pooling`, `introductions`, `copula`, `zones`: pooling over a hierarchy, introductions into silent places, a copula pairing places, and splitting places into zones.

```sh
uv add nowcastbaselines
```
