# nowcastgermany

The German hospitalisation nowcasting hub as a case study.

- `data`: the hub's reporting triangles and forecasts as tasks.
- `evaluation`: the benchmark, our models scored beside the hub's teams.
- `store`: the benchmark's stored forecasts and scores.
- `reporting`: the benchmark's docs page.

It imports `bdbvhub` for the model registry, the download helper and the site's explorer and model list. These are documented exceptions to the one-way order until they move down.

```sh
uv add nowcastgermany
```
