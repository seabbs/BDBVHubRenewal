Manuscript and supporting information for the paper planned in [#206](https://github.com/seabbs/BDBVHubRenewal/issues/206).

## Layout

- `main.qmd`, `si.qmd`, `reference.bib` and `figures/`: the manuscript and its SI.
- `schema.md`: the long table every study writes its forecasts to, the observations table, the rolling truth and the scores.
- `paperfigs/`: the schema checks, scores (checked against scoringutils), rolling truth, factors, periods, plots and dummy data.
- `paperdata/`: study loaders. `paperdata.ebola` reads the BDBV Modeling Hub.
- `paperpipe/` and `workflow/`: the Snakemake evaluation workflow and the steps it calls.
- `papertables/` and `tables/`: the synthetic scenario grid tables.
- `si/`: SI notes on the epinowcast model and the held-out synthetic set.
- `make_figures.py`: every figure from dummy data.
- `tests/`: the tests of all of the above.

## Running

Run from `paper/` with `task`, listed in `Taskfile.yml`.
`task test` runs the tests and `task pipeline:dry` checks both workflow targets.
`task pipeline:dev` runs the development target, which never reads the held-out data.
`task pipeline:paper` runs the paper target, with the held-out data.

The synthetic and German studies run on dummy data until their models are merged (#192).
Ebola reads the hub clone set in `workflow/config.yaml`, and runs on dummy data without one.
