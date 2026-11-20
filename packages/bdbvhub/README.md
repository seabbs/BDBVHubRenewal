# bdbvhub

Nowcasts and forecasts for the BDBV Modeling Hub, built on the `nowcast*` packages.

- `data`: download, tidy and load the hub's data streams, its delays and health zones.
- `models`: the model registry; importing it registers every model.
- `store`: the fit store, its archive and the backtest digest.
- `compare`, `zones`: comparison with the hub's forecasts and health-zone forecasts.
- `submission`: the hub file writer and checks, model metadata and the CI fit configurations.
- `reporting`: the docs pages, tables and explorer.
- `cli`: the `bdbvhub` command (also installed as `bdbvhubrenewal`).

```sh
uv add bdbvhub
```
