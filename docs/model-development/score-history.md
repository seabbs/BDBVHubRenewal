# Score history

Backtest scores at each deploy of this site from main, by model.
{ .page-subtitle }

Each point is one deploy, named by its time and the main commit it was built from.
A commit can deploy more than once, as the fit store reruns the build when it stores new fits and each build refreshes the hub data.
Scores can therefore move without a code change.
Relative WIS, as on the [Backtest](../backtest.md) page, also moves when the baseline changes.
Read it beside WIS, where randomwalk's own line shows the baseline.
"Relative to" picks another model as the baseline, such as one whose forecasts did not change at a deploy where randomwalk's did.
The table is relative to randomwalk.

A star and a vertical line mark a deploy where what a model is scored on changed: the origins, the releases or the number of forecasts.
Points either side of one are not comparable.
Deploys before 2026-11-09 have one summary, scored on each later release as reported, shown as the latest release target.
The last point of a preview build is that build's own scores.
`task score-history` writes the same history to `results/score-history.csv`.

--8<-- "score-history.md"
