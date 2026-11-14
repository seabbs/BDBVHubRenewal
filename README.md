# Paper evidence

Rows the paper workflow (`paper/workflow/config.yaml`, block `evidence`) reads, pinned by commit.
Results of code not yet on main; nothing here is held out.

- `germany/chronos-paper-forecasts.csv.gz`: the German benchmark's paper table for the development half (2021-11-22 to 2022-02-09), with the Chronos-2 models zero-shot and round 1 fine-tuned (germany-chronos2).
- `germany/n1000-ft-forecasts.csv.gz`: the models fine-tuned on 1,000 outbreaks (release chronos-ft-0cea0acef1ce) on the development half, as `germany-backtest` writes them.
