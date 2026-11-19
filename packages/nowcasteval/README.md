# nowcasteval

Evaluation of nowcasts and forecasts.

- `scoring`: sample CRPS, WIS and its components, and the energy score, by level.
- `backtest`: forecasts over a run of releases, scored against later data.
- `nowcast`: nowcast tasks and their truth.
- `marginals`, `ppc`, `checks`: marginal summaries, predictive checks and model checks.
- `pairing`: how trajectories are paired across places.
- `labels`, `plots`, `tables`, `scorechart`, `scoresummary`, `nowcast_plots`, `pairing_plots`: display names, figures and tables of scores.

```sh
uv add nowcasteval
```
