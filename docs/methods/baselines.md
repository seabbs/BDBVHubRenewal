# Baseline models

The random walk nowcast and the random walk.
{ .page-subtitle }

We nowcast the most recent weeks by estimating the reporting delay from past releases, and then forecast each nowcast draw forward with a random walk on the log scale.
This is the random walk nowcast (`randomwalknowcast`).
The same walk started from the latest complete week, with no nowcast, is the random walk (`randomwalk`).
The nowcast uses our port of the baselinenowcast method[^bnc-paper][^bnc-package].
Both are described in full, with their equations and the differences between case studies, in the [paper's supporting information (Baselines)](../paper/si.html#baselines).
The data and notation are on the [Data methods page](data.md#target-and-notation), and the renewal models on the [Renewal models](model.md) page.

Both baselines forecast each province and each health zone.
The country is the per-sample sum of the provinces.
Both pair their province draws within each week by a Gaussian copula ([Correlated trajectories](evaluation.md#correlated-trajectories)).

## Daily release snapshots

Both baselines use the daily onsets of each hub release ([Data](data.md#release-snapshots)).
Only consecutive releases at most $D = 42$ days apart inform the delay, and a pair outside that window keeps its counts in the data.

## Delay

The delay is our port of baselinenowcast's chain ladder, fitted on the daily snapshots, by non-negative least squares over release pairs of any gap up to $D$ ([SI](../paper/si.html#baselines)).

## Nowcast

The random walk nowcast fills each recent week to its nowcast mean as baselinenowcast does.
The final count is the count reported plus negative binomial new reports, whose size is fitted to retrospective nowcasts from earlier releases ([SI](../paper/si.html#baselines)).
A place with fewer than 10 cases a week takes its parent's delay, size and walk spread.

## Forecast

Each sample walks on $\log(1 + \lambda)$ with Poisson counts, with a spread from the last 8 weekly changes net of Poisson noise.
Places with no case in the 9 weeks to the cut-off stay at 0 until an introduction.
The random walk starts from the latest week at least 95% reported, and the random walk nowcast from each sample's own nowcast of the cut-off week ([SI](../paper/si.html#baselines)).

## Health zones

Both baselines run on each zone's own release snapshots as for provinces, and the zone samples are these draws.
No province sample is split over the zones, so zones need not sum to their province.
The hub does not require them to.

## Limitations

While the release history in the delay window is shorter than $D$, no chain to the final count can be checked against a final count.
Each delay factor rests on one onset day per age per pair, and the youngest ages rest on few cases even nationally.
The cut-off is the latest onset day, which can be earlier than the true last report day.
The chain ladder treats reporting speed as fixed by age and step, but speed and backlogs vary between releases.
The observation model is learnt from the earlier releases, which may be few.
Counts include the hub's imputed onsets, which are re-drawn as cases gain onsets, so changes between releases mix reporting with imputation.
Places are sampled independently, so a shock to a whole release is not shared, and the national band is a sum of independent provincial bands.
So it pairs its province draws within each week by a copula fitted to its own errors ([Correlated trajectories](evaluation.md#correlated-trajectories)).
The walk has no drift, and its spread comes from 8 weekly changes, net of a delta-method Poisson noise term that assumes no extra dispersion in the reported counts.

[^bnc-paper]: Johnson KE, Tang M, Tyszka E, Nemcova B, Wolffram D, Ergas R, Reich NG, Funk S, Mellor J, Bracher J, Abbott S (2025). Baseline nowcasting methods for handling delays in epidemiological data. Wellcome Open Research. [doi:10.12688/wellcomeopenres.25027.2](https://doi.org/10.12688/wellcomeopenres.25027.2).
[^bnc-package]: Johnson KE, Tyszka E, Bracher J, Funk S, Abbott S (2025). baselinenowcast: Methods for baseline nowcasting right-truncated epidemiological data. <https://github.com/epinowcast/baselinenowcast/>.
