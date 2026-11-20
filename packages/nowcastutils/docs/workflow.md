# How it is used in the workflow

Every model reads its data through these functions, so they all see the same releases, cut-offs and places.
Each model takes a `ForecastTask` from `nowcastutils.base`, returns a sample table, and is found by name in its registry.
