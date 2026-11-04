# Reporting lag comparison

Whether the renewal model needs a lag between a case's report date and the release that first includes it.
{ .page-subtitle }

`renewal` assumes a case reported on day $t$ is in the release cut off at $T$ with probability $F_\ell(T - t)$, with the lag $\ell$ uniform on 0 to 3 days ([Renewal models](methods/model.md#known-onsets)).
The lag is fixed, not fitted, and the same lag in `randomwalknowcast` was later dropped.
`renewalnolag` is `renewal` without it, so a case reported by the cut-off is in that release.
Both are fitted with the backtest settings at every release and scored on what later releases reported.

## Do reports miss the next release?

Each delay file lists known-onset cases by onset week and delay.
A row in a later file but not the earlier one, whose report day is on or before the earlier file's cut-off, was reported in time but published late.
The report day is the end of the onset week plus the delay, the latest it can be.

![Share of delay rows reported by a cut-off that missed that release, by days from report to cut-off.](figures/lag-late.svg#only-light)
![Share of delay rows reported by a cut-off that missed that release, by days from report to cut-off.](figures/lag-late-dark.svg#only-dark)

--8<-- "lag-findings-late.md"

??? note "Tables"

    --8<-- "lag-late.md"

## Forecasts against later releases

Each arm's median, 50% and 90% intervals at each origin, with the counts of the origin's release and every later one.
The last origin has no later release yet.

![Forecasts of renewal and renewalnolag at each origin, with each later release.](figures/lag-fans.svg#only-light)
![Forecasts of renewal and renewalnolag at each origin, with each later release.](figures/lag-fans-dark.svg#only-dark)

## Scores by release

Mean WIS over every place and week each origin forecast, relative to `randomwalk` on the same forecasts, against the counts each later release reported.

![Relative WIS and interval coverage of renewal and renewalnolag by origin and truth release.](figures/lag-scores.svg#only-light)
![Relative WIS and interval coverage of renewal and renewalnolag by origin and truth release.](figures/lag-scores-dark.svg#only-dark)

--8<-- "lag-scores.md"

## Are the newest weeks too precise?

Nowcasts of the newest weeks, 0 to 3 weeks back from the reference week, against what later releases reported.
Coverage is the share of truths in each interval.
Width is the interval's width over its median.
"Added" is the count a later release added after the origin, realised and at the nowcast median.
The figure pools Congo, Ituri and North Kivu, where the counts are.
Small provinces with few cases cover trivially and are in the tables only.
`renewalsp` and `renewalbvd` are the stored forecasts of 2026-11-03, before the model changes of 2026-11-04.

![Coverage, misses above the 90% interval and width of the newest weeks, by weeks back.](figures/lag-calibration.svg#only-light)
![Coverage, misses above the 90% interval and width of the newest weeks, by weeks back.](figures/lag-calibration-dark.svg#only-dark)

=== "Main places by weeks back"

    --8<-- "lag-calibration-places-weeks.md"

=== "Main places by place"

    --8<-- "lag-calibration-places.md"

=== "All provinces by weeks back"

    --8<-- "lag-calibration-weeks.md"

=== "All provinces by origin"

    --8<-- "lag-calibration-origin.md"

=== "All provinces"

    --8<-- "lag-calibration-overall.md"

--8<-- "lag-findings-calibration.md"

## Convergence

Two chains of 500 warm-up and 500 draws, target acceptance 0.8.

--8<-- "lag-convergence.md"

## Result

--8<-- "lag-findings.md"
