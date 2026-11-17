<!-- bdbv-submission:seabbs-rwnowcast:2026-11-21 -->
Submission of `seabbs-rwnowcast` for 2026-11-21, from data as of 2026-11-15, for 546 locations.

Signed off by Sam Abbott (@seabbs).

Model: random walk on a nowcast (`randomwalknowcast`).

![Nowcast and forecast for 2026-11-21](https://raw.githubusercontent.com/seabbs-bot/BDBV-Modeling-Hub/f5937f066e16b2e522fa9d06267764320d54456e/plots/seabbs-rwnowcast/2026-11-21/44169712191.png)

|  | reported, mean of last 2 complete weeks | nowcast, week ending 2026-11-14 | forecast, week ending 2026-11-21 | forecast / reported |
| --- | --- | --- | --- | --- |
| DRC | 448 | 406 (275 to 649) | 409 (252 to 720) | 0.91 |
| Bas Uele | 0 | 1 (0 to 3) | 1 (0 to 3) | n/a |
| Haut Uele | 27 | 32 (13 to 68.1) | 31.5 (10.9 to 73.1) | 1.17 |
| Ituri | 231 | 223 (148 to 366) | 217 (132 to 361) | 0.94 |
| Nord Kivu | 178 | 104 (56 to 307) | 126 (50.9 to 325) | 0.71 |
| Nord Ubangi | 0 | 0 (0 to 1) | 0 (0 to 1.05) | n/a |
| Sud Kivu | 0 | 0 (0 to 0) | 0 (0 to 0) | n/a |
| Sud Ubangi | 0 | 0 (0 to 1) | 0 (0 to 1) | n/a |
| Tshopo | 12.5 | 11 (5.95 to 16) | 11 (4.95 to 18) | 0.88 |

| check | status |
| --- | --- |
| Convergence | N/A |
| Posterior predictive checks | N/A |
| Forecast checks | WARN |
| Hub validation | PASS |
| Data coherence | PASS |

This round in our docs: https://samabbott.co.uk/BDBVHubRenewal/submission/rounds/2026-11-21/

<details><summary>Details</summary>

**National (`CD`) by week**

| week ending | median | 90% interval |
| --- | --- | --- |
| 2026-10-31 | 468 | 428 to 551 |
| 2026-11-07 | 394 | 324 to 624 |
| 2026-11-14 | 406 | 275 to 649 |
| 2026-11-21 | 409 | 252 to 720 |
| 2026-11-28 | 406 | 249 to 747 |
| 2026-12-05 | 410 | 234 to 792 |
| 2026-12-12 | 408 | 253 to 850 |

**Forecast checks**

- WARN zones: 519 zones; 6115 province sample-weeks differ from the sum of their zones
- WARN bounds: CD-IT-21: 95th centile 146 is 5.41 times the bound base 27 (and 1 more)
- WARN previous: CD-HU-12: median 46 against last week's submission 101 over 2026-10-10 to 2026-12-05 (ratio 0.481) (and 6 more)
- Passed: schema, values, samples, coverage, national_sum, plausible, growth, observed.

</details>

This was opened by a bot. Please ping @seabbs for any questions.
