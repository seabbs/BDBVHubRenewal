# Pairing sensitivity

How much pairing province draws within each week matters for `randomwalk` and `randomwalknowcast` ([Correlated trajectories](methods/evaluation.md#correlated-trajectories)).
{ .page-subtitle }

Each model forecasts each origin once, and its draws are scored with no pairing and paired by the rolling $\rho_p$ estimated from its own errors, which is what it uses.
Zones are not paired, so they are not compared here.
Each week is paired on its own, so a place's sum over its weeks is not kept as one trajectory; it is shown, but nothing we submit or score uses it.
Pairing only reorders draws, so each place's weekly forecast is unchanged; only sums and joint scores move.
The country is the sum of the provinces (or Germany of its states), so its weekly score moves too.
Values are mean sample CRPS (energy score for "all places and weeks") relative to no pairing, so below 1 is better, with the 90% interval coverage of the same targets.

--8<-- "pairing.md"
