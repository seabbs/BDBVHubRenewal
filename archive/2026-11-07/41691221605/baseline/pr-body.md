<!-- bdbv-submission:seabbs-rwnowcast:2026-11-07 -->
Automated submission of `seabbs-rwnowcast` for 2026-11-07, from data as of 2026-11-01, for 187 locations.

All checks passed, with warnings on bounds.

![Nowcast and forecast for 2026-11-07](https://raw.githubusercontent.com/seabbs-bot/BDBV-Modeling-Hub/62f7678c368fba846ca0488dfed4851357e78e58/plots/seabbs-rwnowcast/2026-11-07/41691221605.png)

This round in our docs: https://samabbott.co.uk/BDBVHubRenewal/submission/rounds/2026-11-07/

<details><summary>Checks</summary>

| check | status |
| --- | --- |
| Convergence | N/A (no sampler) |
| Posterior predictive checks | N/A (the model has no sampler) |
| Forecast checks | WARN (bounds flagged) |
| Hub validation | PASS |
| Data coherence | PASS (national (CD) equals the sum of the provinces in every week) |

</details>

<details><summary>Forecast checks</summary>

- WARN bounds: CD-HU-02: 95th centile 172 is 8.61 times the bound base 20 (and 3 more)
- Passed: schema, values, samples, coverage, zones, national_sum, plausible, growth, observed.

</details>

<details><summary>National summary</summary>

Submitted `CD` samples.

| week ending | median | 90% interval |
| --- | --- | --- |
| 2026-10-17 | 480 | 432 to 607 |
| 2026-10-24 | 550 | 412 to 821 |
| 2026-10-31 | 576 | 329 to 1.07e+03 |
| 2026-11-07 | 575 | 331 to 1.06e+03 |
| 2026-11-14 | 600 | 321 to 1.06e+03 |
| 2026-11-21 | 606 | 312 to 1086 |
| 2026-11-28 | 624 | 319 to 1.05e+03 |

</details>

<details><summary>Provenance</summary>

|  |  |
| --- | --- |
| File | `model-output/seabbs-rwnowcast/2026-11-07-seabbs-rwnowcast.csv` |
| Contents | 187 locations, 187000 rows |
| Model | randomwalknowcast (`randomwalknowcast`), version 0.1.0, code [`16340ee355`](https://github.com/seabbs/BDBVHubRenewal/commit/16340ee3556f578b7482d79f98657e7611b60b7a) |
| Data vintage (`as_of`) | 2026-11-01 |
| Hub data commit | [`d67f1b74e0`](https://github.com/InsightNet-US/BDBV-Modeling-Hub/commit/d67f1b74e0ecd1b32bb3d57554d871502acb2eca) |
| Hub config | [`b66c18270b`](https://github.com/InsightNet-US/BDBV-Modeling-Hub/tree/b66c18270b3cf90bb6668f1a2a6ec759867e0916) |
| Built | 2026-11-03 18:16 UTC, [workflow run](https://github.com/seabbs/BDBVHubRenewal/actions/runs/41691221605) |

</details>

This was opened by a bot. Please ping @seabbs for any questions.
