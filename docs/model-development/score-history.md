# Score history

Backtest scores by target week at each main commit.
{ .page-subtitle }

Each point is a main commit, named by its short sha and the date of its latest deploy of this site.
A commit that was submitted for a round is labelled with the round.
Scores can move without a code change, as the fit store reruns the build when it stores new fits and each build refreshes the hub data.
A commit's point is its latest deploy.

Each target week is scored on its own: a point is the mean over the forecasts of that week scored at that commit.
There is no score over all weeks, as later commits score more weeks, origins and releases.
"Common forecasts only", ticked at first, keeps for each model, baseline and week the forecasts (origin, release and place) scored at every commit of a window back from the newest.
The window is as long as keeps the most common forecasts times commits.

Relative WIS, as on the [Backtest](../backtest.md) page, also moves when the baseline changes.
Read it beside WIS, where randomwalk's own line shows the baseline, or pick randomwalknowcast under "Relative to".
A star and a vertical line mark a commit where what a line is scored on changed.
Points either side of one are not comparable.
The last point of a preview build is that build's own scores.
`task score-history` writes the same history to `results/score-history.csv`.

--8<-- "score-history.md"
