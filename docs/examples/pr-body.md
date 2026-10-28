<!-- bdbv-submission:epiforecasts-persistence:2026-10-31 -->
Automated submission of `epiforecasts-persistence` for 2026-10-31, from data as of 2026-10-24, for 9 locations.

All checks passed, with warnings on bounds. Stand-in: persistence runs as the `main` model. Dry run: nothing was pushed.

![Nowcast and forecast for 2026-10-31](../assets/example-submission-nowcast.png)

This round in our docs: https://samabbott.co.uk/BDBVHubRenewal/submission/rounds/2026-10-31/

<details><summary>Checks</summary>

| check | status |
| --- | --- |
| Convergence | N/A (no sampler) |
| Posterior predictive checks | N/A (persistence has no sampler and no posterior) |
| Forecast checks | WARN (bounds flagged) |
| Hub validation | PASS |
| Data coherence | PASS (national (CD) equals the sum of the provinces in every week) |

</details>

<details><summary>Forecast checks</summary>

- WARN bounds: CD-TO: 95th centile 108 is 5.4 times the bound base 20
- Passed: schema, values, samples, coverage, national_sum, growth, observed, baseline.

</details>

<details><summary>National summary</summary>

Submitted `CD` samples.

| week ending | median | 90% interval |
| --- | --- | --- |
| 2026-10-10 | 413 | 413 to 413 |
| 2026-10-17 | 427 | 339 to 528 |
| 2026-10-24 | 430 | 333 to 580 |
| 2026-10-31 | 432 | 311 to 683 |
| 2026-11-07 | 436 | 302 to 782 |
| 2026-11-14 | 452 | 308 to 828 |
| 2026-11-21 | 460 | 314 to 881 |

</details>

<details><summary>Provenance</summary>

|  |  |
| --- | --- |
| File | `model-output/epiforecasts-persistence/2026-10-31-epiforecasts-persistence.csv` |
| Contents | 9 locations, 9000 rows |
| Model | persistence (`main`), version 0.1.0, code [`fd9d26eaed`](https://github.com/seabbs/BDBVHubRenewal/commit/fd9d26eaedecf543c851271d365d52db09fff85c) |
| Data vintage (`as_of`) | 2026-10-24 |
| Hub data commit | [`187015937f`](https://github.com/InsightNet-US/BDBV-Modeling-Hub/commit/187015937fc7c0fc28cc769fb938ed89c9e18ae8) |
| Hub config | [`88478a33b6`](https://github.com/InsightNet-US/BDBV-Modeling-Hub/tree/88478a33b6f81dc717eec16d3ec0417c8f0d5588) |
| Built | 2026-10-27 18:36 UTC, [workflow run](https://github.com/seabbs/BDBVHubRenewal/actions/runs/40556037584) |

</details>

This was opened by a bot. Please ping @seabbs for any questions.
