Settings drawn for each simulated outbreak. Realised values are over the 2,000 training outbreaks.

| Factor | Range or levels | Distribution | Enters through | Realised, median [min, max] |
|---|---|---|---|---|
| Weeks of cut-offs | 16 to 40 | Uniform on integers | `enw_preprocess_data()` grid, with 12 more weeks of reports | 26 [16, 40] |
| Start day offset (days) | 0 to 6 | Uniform on integers | `enw_preprocess_data()` grid | 3 [0, 6] |
| Longest delay (days) | 56 | Fixed | `enw_preprocess_data(max_delay)` |  |
| Places | 1, or 2 to 26 | 1 with probability 0.4, else uniform on integers | `enw_preprocess_data(by = 'location')` | 4 [1, 26]; 45% one place |
| Weight of the shared log Rt path | 0 to 1 | Uniform | `enw_simulate(growth_rate)` | 0.522 [0.00217, 0.999] |
| Generation time mean (days) | 6 to 20 | Uniform | `enw_expectation(generation_time)`, discretised gamma on 1 to 45 days | 13.6 [6, 20] |
| Generation time coefficient of variation | 0.3 to 0.7 | Uniform | `enw_expectation(generation_time)` | 0.496 [0.3, 0.7] |
| Initial log Rt | Unbounded | Normal(0.1, 0.2) | `enw_simulate(growth_rate)`, with `enw_expectation(r = ~1)` | 0.0697 [-0.715, 0.758] |
| Step changes in log Rt | 0 or more | Poisson(1.2), at uniform days | `enw_simulate(growth_rate)` | 1 [0, 6] |
| Step size sd (mean -0.2) | 0.1 to 0.5 | Uniform | `enw_simulate(growth_rate)` | 0.302 [0.1, 0.5] |
| Daily random walk sd of log Rt (clamped to -1.2 to 1.2) | 0 to 0.03 | Uniform | `enw_simulate(growth_rate)` | 0.0143 [0.0000134, 0.03] |
| Initial expected cases a day, over all places | 0.5 to 20 | Log-uniform | `expr_lelatent_int` of `enw_expectation()` | 2.89 [0.502, 20] |
| Sd of the places' log seed weights | 0.5 to 1.5 | Uniform | `expr_lelatent_int` of `enw_expectation()` | 1.01 [0.502, 1.5] |
| Delay median, exp of the lognormal log mean (days) | 2 to 21 | Log-uniform | `refp_mean_int` of `enw_reference(parametric = ~1 + week)` | 7.89 [2.01, 21] |
| Delay lognormal log sd | 0.3 to 1.1 | Uniform | `refp_sd_int` of `enw_reference()` | 0.83 [0.302, 1.1] |
| Daily change in the delay log mean (applied weekly) | Unbounded | Normal(0, 0.02) | `week` effect of `enw_reference(parametric)` | 0.000271 [-0.0497, 0.0453] |
| Sd of the places' delay log means | 0 to 0.4 | Uniform | `(1 \| location)` of `enw_reference(parametric)` | 0.199 [0.000287, 0.4] |
| Sd of the delay hazard effects | 0 to 0.4 | Uniform | `enw_reference(non_parametric = ~ 0 + (1 \| delay))` | 0.198 [0.0000284, 0.4] |
| Days a week with reports | 7, or 1 to 3 weekdays | 7 with probability 0.5, else uniform on 1 to 3 days of Monday to Friday | `enw_dayofweek_structural_reporting()` in `enw_report(structural)` | 52% every day |
| Sd of the report day-of-week effects | 0 to 0.8 | Uniform | `enw_report(non_parametric = ~ 0 + (1 \| day_of_week))` | 0.394 [0.000673, 0.799] |
| Negative binomial overdispersion, 1 / sqrt(phi) | 0.05 to 1 | Uniform | `enw_obs(family = 'negbin')` | 0.521 [0.0504, 0.999] |
| Reported cases in an outbreak | 50 to 30,000 | Outbreaks outside the range are drawn again | `enw_simulate()` | 1,160 [51, 29,970] |
