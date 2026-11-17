# Grid r5 pilot and QC (bdbv-sims)

Simulator at seabbs/BDBVHubRenewal bdbv-sims-r5 147218b1f (grid r5), the commit the rounds run from round 9.

- `pilot/sims/`: 20 pilot simulations (seeds 90001-90004, 5 each), all 52 weeks (`ENW_WEEKS=52`), `ENW_DRAWS=1`, in the rounds' format (`counts-<seed>.csv.gz`: outbreak, timestep, location, reference_date, report_date, count; `settings-<seed>.csv`: one row per simulation). Run with `scripts/sim_pilot.sh OUT 4 5 52 14G`.
- `pilot/pilot.csv`: peak RSS, wall time and peak TMPDIR per batch of 5.
- `pilot/worst_case_probe.*`: one places x ages simulation of 32 strata (the r5 cap at 52 weeks), 84-day max delay and a walking delay.
- `qc/`: trajectories per new level (bunching, slowdown, uneven multi_wave, 52-week places x ages), family shares, case sizes, summary table (`qc/sim_qc_r5.py pilot/sims qc`).
- `r1_r8_simulation_families.csv`: the observation family of every r1-r8 simulation (exact, mixed, or unrecorded for independent strata whose own cell drew families).
