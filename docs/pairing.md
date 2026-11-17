# Pairing sensitivity

How much the rolling copula matters for `randomwalk` and `randomwalknowcast` ([Correlated trajectories](methods/evaluation.md#correlated-trajectories)).
{ .page-subtitle }

Each model forecasts each origin once, and its draws are scored with no pairing, paired by the prior, and paired by the copula estimated from its own errors (rolling), which is what it uses.
Pairing only reorders draws, so each place's weekly forecast is unchanged; only sums and joint scores move.
The country is the sum of the provinces (or Germany of its states), so its weekly score moves too.
Values are mean sample CRPS (energy score for "all places and weeks") relative to no pairing, so below 1 is better, with the 90% interval coverage of the same targets.

--8<-- "pairing.md"
