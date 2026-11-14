# The held-out synthetic set

The held-out set is simulated on grid `r3` with seeds 8016 to 8025, 25 simulations each.
These cover the complete balanced blocks 1670 to 1671, and simulations outside them are not used.
We evaluate draw 1 of each simulation, so the outbreaks are independent.
The seeds are disjoint from the fine-tune's training and held-out seeds (`config/finetune/r3.env`).
The simulations are written to `output/finetune/heldout`, and `config/paper/synthetic.toml` records them.
The protocol is in `paper/synthetic.py`, and `paper/run_synthetic.py` runs the models on the set.

The held-out set was reserved for the final evaluation.
No model was scored on it during development.
We checked the pipeline on the development seeds 7001 to 7001 and the pilot simulations, whose forecasts are not results.
