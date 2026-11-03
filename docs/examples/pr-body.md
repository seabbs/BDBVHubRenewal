<!-- bdbv-submission:seabbs-renewal:2026-11-07 -->
Submission of `seabbs-renewal` for 2026-11-07, from data as of 2026-11-01, for 187 locations.

Signed off by Sam Abbott (@seabbs).

Model: BVDOutbreakSize-informed renewal model (`renewalbvd`), the first model in our chain and the first to clear its checks.

![Nowcast and forecast for 2026-11-07](https://raw.githubusercontent.com/seabbs-bot/BDBV-Modeling-Hub/4a8fca762876fcda5b674afbacae90a11db77ec1/plots/seabbs-renewal/2026-11-07/41691221605.png)

National (`CD`) submitted samples:

| week ending | median | 90% interval |
| --- | --- | --- |
| 2026-10-17 | 467 | 440 to 504 |
| 2026-10-24 | 541 | 495 to 598 |
| 2026-10-31 | 565 | 477 to 663 |
| 2026-11-07 | 562 | 403 to 744 |
| 2026-11-14 | 582 | 383 to 861 |
| 2026-11-21 | 612 | 352 to 925 |
| 2026-11-28 | 566 | 308 to 936 |

| check | status |
| --- | --- |
| Convergence | WARN |
| Posterior predictive checks | PASS |
| Forecast checks | WARN |
| Hub validation | PASS |
| Data coherence | PASS |

This round in our docs: https://samabbott.co.uk/BDBVHubRenewal/submission/rounds/2026-11-07/

<details><summary>Details</summary>

**Convergence**

WARN (max R-hat 1.024, min bulk ESS 132, 14 divergences in 1000 draws).
- min tail ESS is 88.7, past the warning threshold 100
- divergent fraction is 0.014, past the warning threshold 0.01

The model used BVDOutbreakSize release `results-3306` (data to 2026-10-25, joint R-hat 1.042, limit 1.1).

**Posterior predictive checks**

Share of observed weekly all streams counts inside the in-sample predictive intervals.

|  | 50% | 90% | 50%, last 4 weeks | 90%, last 4 weeks |
| --- | --- | --- | --- | --- |
| all provinces | n/a | n/a | n/a | n/a |
| CD | 0.81 | 0.98 | 0.75 | 1 |
| CD-BU | 0.93 | 1 | 0.62 | 1 |
| CD-HU | 0.91 | 1 | 1 | 1 |
| CD-IT | 0.79 | 0.98 | 0.75 | 1 |
| CD-NK | 0.72 | 0.97 | 0.38 | 1 |
| CD-NU | 1 | 1 | 1 | 1 |
| CD-SK | 0.98 | 0.98 | 1 | 1 |
| CD-SU | 0.97 | 1 | 0.75 | 1 |
| CD-TO | 0.9 | 0.98 | 0.75 | 0.88 |

PIT: KS distance 0.062 from uniform.

**Forecast checks**

- WARN baseline: CD-IT-05: median 28.5 against random walk baseline 1 over 2026-11-14 to 2026-11-28 (ratio 5.58) (and 4 more)
- Passed: schema, values, samples, coverage, zones, national_sum, bounds, plausible, growth, observed.

</details>

This was opened by a bot. Please ping @seabbs for any questions.
