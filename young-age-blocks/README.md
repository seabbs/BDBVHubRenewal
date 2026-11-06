# Evidence for the young-age block PR (#69)

Arms, all on top of #63 (63f0b209):

- `A`: #63.
- `B`: #63 with #65's cut-offs (asof-one-cutoff merged).
- `C`: B with ages pooled into blocks of at least 10 cases (this PR).
- `D`: #63 with blocks, without #65.
- `C30`, `C100`: C with blocks of 30 and 100 cases (`variants.py`, `blk30`, `blk100`), randomwalknowcast only, no zone run.

Each folder holds the recovery and release scores and quantiles, and the zone scores where run.
- `Cz`, `Dz`: C and D with ages that have no case left out of the blocks (`variants.py`, `skipzero`).
