Settings of each simulated outbreak (grid r2-real). Realised values follow once the r2-real settings exist.

| Factor | Range or levels | Distribution | Enters through | Realised |
|---|---|---|---|---|
| Strata | One place; 2 to 26 places; 2 to 8 age groups in one place; 6 to 16 places by 3 to 7 age groups | Probabilities 0.3, 0.4, 0.2 and 0.1 | `enw_preprocess_data(by = c('place', 'age'))` |  |
| Places | 1 to 26 | Uniform on the strata's range | `enw_preprocess_data()` |  |
| Age groups | 1 to 8 | Uniform on the strata's range | `enw_preprocess_data()` |  |
| Generative process | Renewal; renewal with incubation; weekly growth walk; log incidence walk; growth steps; smooth growth | Uniform | `enw_expectation()` and `enw_simulate(growth_rate)` |  |
| Reference date model | Parametric; non-parametric; both | Probabilities 0.3, 0.2 and 0.5 | `enw_reference()` |  |
| Parametric delay | Lognormal; gamma; exponential | Uniform, if there is a parametric delay | `enw_reference(distribution)` |  |
| Delay hazard effects | None; by delay; by week; both | None with a parametric delay alone, by delay if stable, else uniform | `enw_reference(non_parametric)` |  |
| Stable delay | Yes or no | Yes with probability 0.2, which turns off every delay and report change | All delay changes |  |
| Drift in the delay's location | Yes or no | Probability 0.6 if not stable | `week` in `enw_reference(parametric)` |  |
| Drift in the delay's spread | Yes or no | Probability 0.5 if not stable | `week` in `enw_reference(parametric)` |  |
| Step changes in the delay | 0 to 3 | Probabilities 0.3, 0.3, 0.2 and 0.2 if not stable | Step covariates in `enw_reference(parametric)` |  |
| Weekly random walk in the delay | Yes or no | Probability 0.3 if not stable | `rw(week)` in `enw_reference(parametric)` |  |
| Day-of-week report effects | Yes or no | Probability 0.7 | `(1 \| day_of_week)` in `enw_report()` |  |
| Holidays | Yes or no | Probability 0.3 | `holiday` in `enw_report()` |  |
| Report week effects | None; `(1 \| week)`; `rw(week)` | Probabilities 0.6, 0.2 and 0.2 | `enw_report()` |  |
| Days with reports | Every day, or 1 to 3 of Monday to Friday | Every day with probability 0.5, else uniform on 1 to 3 days | `enw_report(structural)` |  |
| Days with reports after a change | Every day, or 1 to 3 of Monday to Friday | As the days before | `enw_report(structural)` |  |
| Change of report days | Yes or no | Probability 0.3 if not stable | `enw_report(structural)` |  |
| Observation model | NB2; NB1; Poisson | Probabilities 0.4, 0.4 and 0.2 | `enw_obs(family)` |  |
| Longest delay (days) | 56 or 84 | Probabilities 0.75 and 0.25 | `enw_preprocess_data(max_delay)` |  |
| Weeks of cut-offs | 16 to 40, or 16 to 24 for places by ages | Uniform on integers | `enw_preprocess_data()` grid, with the longest delay and 4 more weeks of reports |  |
| Start day offset (days) | 0 to 6 | Uniform on integers | `enw_preprocess_data()` grid |  |
| Probability a place is quiet | 0 to 0.9 | Uniform | `expr_lelatent_int` of `enw_expectation()` |  |
| Weight of the shared growth path | 0 to 1 | Uniform | `enw_simulate(growth_rate)` |  |
| Generation time mean (days) and coefficient of variation | 6 to 20; 0.3 to 0.7 | Uniform | `enw_expectation(generation_time)`, gamma on 1 to 45 days |  |
| Incubation period mean (days) and coefficient of variation | 3 to 12; 0.3 to 0.7 | Uniform | `enw_expectation(latent_reporting_delay)`, gamma on 1 to 30 days |  |
| Initial log Rt | Unbounded, paths held to -1.2 to 1.2 | Normal(0.1, 0.2) | `enw_simulate(growth_rate)` |  |
| Step changes in log Rt, and their size sd (mean -0.2) | 0 or more; 0.1 to 0.5 | Poisson(1.2); uniform | `enw_simulate(growth_rate)` |  |
| Daily random walk sd of log Rt | 0 to 0.03 | Uniform | `enw_simulate(growth_rate)` |  |
| Initial daily growth rate | Unbounded, paths held to -0.06 to 0.06 | Normal(0.01, 0.02) | `enw_simulate(growth_rate)` |  |
| Weekly growth walk sd | 0.005 to 0.015 | Uniform | `enw_simulate(growth_rate)` |  |
| Daily growth noise of the log incidence walk | 0.05 to 0.3 | Uniform | `enw_simulate(growth_rate)` |  |
| Step changes in growth, and their size sd (mean -0.01) | 1 to 3; 0.02 to 0.06 | Uniform | `enw_simulate(growth_rate)` |  |
| Smooth growth amplitude, period (days) and phase | 0.01 to 0.06; 40 to 200; 0 to 2 pi | Uniform | `enw_simulate(growth_rate)` |  |
| Initial expected cases a day, over all strata | 0.2 to 200 | Log-uniform | `expr_lelatent_int` of `enw_expectation()` |  |
| Sd of the places' log seed weights (half for age groups) | 0.5 to 2 | Uniform | `expr_lelatent_int` of `enw_expectation()` |  |
| Delay mean (days) | 2 to 21 | Log-uniform | `refp_mean_int`, or the hazard intercept with no parametric delay |  |
| Delay coefficient of variation | 0.3 to 1.2 (lognormal); 0.7 to 1.2 (gamma); 1 (exponential) | Uniform | `refp_sd_int` of `enw_reference()` |  |
| Log change in the delay's location and spread over the outbreak | Unbounded | Normal(0, 0.5); Normal(0, 0.3) | `week` effects of `enw_reference(parametric)` |  |
| When each delay step happens, as a share of the outbreak | 0.1 to 0.9 | Uniform | Step covariates in `enw_reference(parametric)` |  |
| Delay steps in log location and spread | Unbounded | Normal(0, 0.4); Normal(0, 0.3) | Step effects of `enw_reference(parametric)` |  |
| Weekly walk sd of the delay's location and spread | 0 to 0.1 | Uniform | `rw(week)` of `enw_reference(parametric)` |  |
| Sd of the strata's delay effects | 0 to 0.4 | Uniform | `(1 \| place)` and `(1 \| age)` of `enw_reference()` |  |
| Sd of the hazard effects by delay and by week | 0 to 0.4 | Uniform | `enw_reference(non_parametric)` |  |
| Sd of the day-of-week report effects | 0 to 0.8 | Uniform | `(1 \| day_of_week)` of `enw_report()` |  |
| Holidays, and their effect on the logit hazard | 2 to 6; -2.5 to -0.5 | Uniform | `holiday` of `enw_report()` |  |
| Sd of the report week effects | 0 to 0.3 | Uniform | `enw_report()` |  |
| When report days change, as a share of the outbreak | 0.2 to 0.8 | Uniform | `enw_report(structural)` |  |
| Overdispersion, 1 / sqrt(phi), on the NB2 scale | 0.05 to 1.5 | Uniform; NB1 scaled to match | `enw_obs()` |  |
| Reported cases in an outbreak | 10 to 300,000, more with many strata | Drawn again up to 5 times, else the design is skipped | `enw_simulate()` |  |
