# Score history

Backtest scores at each deploy of this site from main, by model.
{ .page-subtitle }

Each point is one deploy, named by its time and the main commit it was built from.
A commit can deploy more than once, as the fit store reruns the build when it stores new fits and each build refreshes the hub data.
Scores can therefore move without a code change.
Relative WIS, as on the [Backtest](../backtest.md) page, also moves when the baseline changes.
Read it beside WIS, where randomwalk's own line shows the baseline.
"Relative to" picks randomwalknowcast instead, whose forecasts did not change at deploys where randomwalk's did.
The table is relative to randomwalk, over all forecasts.

Deploys score different forecasts as origins and releases are added, so a mean over all of them mixes target weeks and places.
Each target week is its own series: a deploy's point is the mean over the forecasts of that week it scored.
"Split by" lists the weeks newest first, then all forecasts, and opens on the newest week scored at the most deploys.
"Common forecasts only" keeps, for each model, baseline and week, the forecasts (origin, release and place) scored at every deploy of a window that runs back from the newest deploy.
The window is as long as keeps the most common forecasts times deploys, so it ends before a deploy that would drop most of them.

A star and a vertical line mark a deploy where what a model is scored on changed: the origins, the releases or the number of forecasts.
On common forecasts only the first deploy of the window is marked.
Points either side of one are not comparable.
Deploys before 2026-11-09 have one summary, scored on each later release as reported, shown as the latest release target.
The last point of a preview build is that build's own scores.
`task score-history` writes the same history to `results/score-history.csv`.

--8<-- "score-history.md"
