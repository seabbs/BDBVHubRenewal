# The held-out synthetic set

The held-out set is simulated on grid `r3` with seeds 8016 to 8025, 25 simulations each.
These cover the complete balanced blocks 1670 to 1671, and simulations outside them are not used.
We evaluate draw 1 of each simulation, so the outbreaks are independent.
The seeds are disjoint from the fine-tune's training and held-out seeds (`config/finetune/r3.env`).
The simulations are written to `output/finetune/heldout`, and `config/paper/synthetic.toml` records them.
The protocol is in `paper/synthetic.py`, and `paper/run_synthetic.py` runs the models on the set.

The held-out set was reserved for the final evaluation.
No model was scored on it during development.
We checked the pipeline on the r3 pilot seeds 9501 to 9504, whose forecasts are not results.

## Protocol

Each outbreak is released weekly, on one weekday from Wednesday to Saturday drawn from its id, and a release's cut-off is its origin.
Each outbreak has four origins, spread evenly from its 6th release to the last release at least its longest delay plus 5 weeks before its last report.
Each forecast covers weeks -4 to 3 from the week of the cut-off, as weekly totals by onset week for the country and each stratum.
Scores use rolling truth, the count reported D days after the end of a target week, with D the outbreak's own 98% reporting delay in whole days, and are relative to `randomwalknowcast`.
