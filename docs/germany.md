# German nowcast benchmark

Our models and the hub teams on the German COVID-19 hospitalisation nowcasting hub's data, scored as the hub scored them.
{ .page-subtitle }

Nowcasts of the 7-day hospitalisation incidence by Meldedatum, made each day from 2021-11-22 to 2022-02-09 as in Wolffram et al. (2023)[^wolffram], for the sums ending 0, 7, 14, 21 and 28 days before the nowcast date.
The paper's later dates, from 2022-02-10, are held out.
Every nowcast is scored on its quantiles at the hub's seven levels, by WIS and the coverage of the 50% and 95% intervals, on the data reported 40 days after the sum ends ([how the benchmark is set up](methods/evaluation.md#german-nowcast-benchmark)).
Relative scores are against `randomwalknowcast` on the nowcasts both made.

## Overall

Scores at each of the paper's levels: Germany, its states and its age groups.

--8<-- "germany-scores.md"

## By horizon

By days from the nowcast date to the end of the 7-day sum, at the level chosen.
The round picker is the nowcast date.

--8<-- "germany-horizon.md"

## By place

Scores at each state or age group.
Pick a place to narrow the table.

--8<-- "germany-place.md"

## Data

Data from the Robert Koch Institute (CC BY 4.0), processed by the German COVID-19 hospitalisation nowcasting hub, and the teams' nowcasts as the paper gathered them:

--8<-- "germany-sources.md"

[^wolffram]: Wolffram D, Abbott S, an der Heiden M, Funk S, Günther F, Hailer D, et al. (2023). Collaborative nowcasting of COVID-19 hospitalization incidences in Germany. PLOS Computational Biology 19(8): e1011394. [doi:10.1371/journal.pcbi.1011394](https://doi.org/10.1371/journal.pcbi.1011394).
