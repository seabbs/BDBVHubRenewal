# German nowcast benchmark

Our models on the German COVID-19 hospitalisation nowcasting hub's data, beside the hub teams, scored as in Wolffram et al. (2023)[^wolffram].
{ .page-subtitle }

Nowcasts of the 7-day hospitalisation incidence by Meldedatum, made each day from 2021-11-22 to 2022-04-29, for the sums ending 0, 7, 14, 21 and 28 days before the nowcast date, scored against the data of 2022-08-08 ([how the benchmark is set up](methods/evaluation.md#german-nowcast-benchmark)).
The renewal models use the BDBV hub's line list and outbreak estimates, so they are not run.
Hub teams are in grey.

## Overall

Mean WIS and coverage of the 50% and 95% intervals at each of the paper's levels.
The national level is Germany at all ages, the states are at all ages and the age groups are for Germany.

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

Data from the Robert Koch Institute (CC BY 4.0), processed by the German COVID-19 hospitalisation nowcasting hub, with the hub teams' nowcasts and the paper's scores from its evaluation repository (MIT), at these commits:

--8<-- "germany-sources.md"

[^wolffram]: Wolffram D, Abbott S, an der Heiden M, Funk S, Günther F, Hailer D, et al. (2023). Collaborative nowcasting of COVID-19 hospitalization incidences in Germany. PLOS Computational Biology 19(8): e1011394. [doi:10.1371/journal.pcbi.1011394](https://doi.org/10.1371/journal.pcbi.1011394).
