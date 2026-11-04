# Reporting lag comparison

Whether the renewal model's reporting part is right: the lag to publication and the onset-to-report delay.
{ .page-subtitle }

`renewal` assumes a case reported on day $t$ is in the release cut off at $T$ with probability $F_\ell(T - t)$, with the lag $\ell$ uniform on 0 to 3 days ([Renewal models](methods/model.md#known-onsets)).
The lag is fixed, not fitted, and the same lag in `randomwalknowcast` was later dropped.
Its delay is shared by every province and shifts with onset week through a weekly walk $\gamma(s)$, and neither choice has been tested.
Each version below changes one part of `renewal`, is fitted with the backtest settings at every release, and is scored on what later releases reported.

| Version | Change from `renewal` |
|---|---|
| `renewalnolag` | No lag, so a case reported by the cut-off is in that release |
| `renewalplace` | A delay for each province, $\operatorname{logit} h_{p,s}(d) = \beta(d) + \gamma(s) + u_p$ with $u_p \sim \mathcal{N}(0, \sigma_u^2)$ and $\sigma_u \sim \mathcal{N}^+(0, 0.5^2)$, and the delay counts scored by province rather than pooled ([#22](https://github.com/seabbs/BDBVHubRenewal/issues/22)) |
| `renewalfixdelay` | A delay fixed over time, $\gamma = 0$ |

`renewalplace` changes two things, as the province shifts are identified by the delay counts of each province.
Zones are not in it.
The renewal models fit provinces, and the delay files give provinces only, so a zone delay cannot be identified and a zone takes its province's delay through the split of province samples.
So #22's zone part is not met.
`renewalbvd` takes the province delay from the same option.
`renewalsp` would take it by scoring, in each province's fit, the other provinces' delay counts for the shared delay and its own for its shift.
That breaks its definition that nothing is pooled across provinces, and with one province per fit $\sigma_u$ comes from its prior alone.
Both are fitted only if the province delay helps `renewal`.

## Do reports miss the next release?

Each delay file lists known-onset cases by onset week and delay.
A row in a later file but not the earlier one, whose report day is on or before the earlier file's cut-off, was reported in time but published late.
The report day is the end of the onset week plus the delay, the latest it can be.

![Share of delay rows reported by a cut-off that missed that release, by days from report to cut-off.](figures/lag-late.svg#only-light)
![Share of delay rows reported by a cut-off that missed that release, by days from report to cut-off.](figures/lag-late-dark.svg#only-dark)

Over the three pairs of delay files, rows reported 0 to 2 days before a cut-off missed that release 11.7% of the time (53 of 452), against 6.0% for rows 4 to 13 days before (100 of 1,655).
So some reports reach the next release a few days late, as the lag assumes.
The pairs differ, from 4% to 24% at 0 days, and late rows also come from onsets recorded later for cases first counted as imputed.

??? note "Tables"

    --8<-- "lag-late.md"

## Forecasts against later releases

Each version's median, 50% and 90% intervals at each origin beside `renewal`'s, with the counts of the origin's release and every later one.
The last origin has no later release yet.

=== "renewalnolag"

    ![Forecasts of renewal and renewalnolag at each origin, with each later release.](figures/lag-fans-renewalnolag.svg#only-light)
    ![Forecasts of renewal and renewalnolag at each origin, with each later release.](figures/lag-fans-renewalnolag-dark.svg#only-dark)

=== "renewalplace"

    ![Forecasts of renewal and renewalplace at each origin, with each later release.](figures/lag-fans-renewalplace.svg#only-light)
    ![Forecasts of renewal and renewalplace at each origin, with each later release.](figures/lag-fans-renewalplace-dark.svg#only-dark)

=== "renewalfixdelay"

    ![Forecasts of renewal and renewalfixdelay at each origin, with each later release.](figures/lag-fans-renewalfixdelay.svg#only-light)
    ![Forecasts of renewal and renewalfixdelay at each origin, with each later release.](figures/lag-fans-renewalfixdelay-dark.svg#only-dark)

## Scores by release

Mean WIS over every place and week each origin forecast, relative to `randomwalk` on the same forecasts, against the counts each later release reported.

![Relative WIS and interval coverage of each version by origin and truth release.](figures/lag-scores.svg#only-light)
![Relative WIS and interval coverage of each version by origin and truth release.](figures/lag-scores-dark.svg#only-dark)

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
