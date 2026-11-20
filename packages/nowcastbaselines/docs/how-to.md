# How to

The examples continue from [Getting started](index.md).

## Score the baselines

The forecasts are sample tables, so nowcasteval scores them ([nowcasteval](https://samabbott.co.uk/bdbv-hub-modelling/nowcasteval/how-to/#score-a-table-of-forecasts)).

## Run them over many releases

A backtest is one task per release, each with the data up to that release.
`nowcasteval.backtest.make_tasks` builds them from a target table and the daily releases, and `nowcasteval.backtest.run_forecasts` runs a list of models over them, skipping a model where it cannot forecast.

Next: [Reference](reference.md).
