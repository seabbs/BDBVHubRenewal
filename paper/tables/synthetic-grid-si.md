Realised values follow once the simulations exist.

| Setting | Drawn as | Enters through | Realised, median [min, max] or shares |
|---|---|---|---|
| Generative process | One of `growth`, `renewal_3`, `renewal_7`, `renewal_15` | `enw_expectation()` |  |
| Generation time shape | One of `gamma_0.3`, `gamma_0.6`, `gamma_1`, `lognormal_0.6` | `enw_expectation(generation_time)` |  |
| Trajectory of the growth rate | One of `growth`, `decline`, `flat`, `rise_fall_sharp`, `rise_fall_gradual`, `fall_rise`, `oscillating`, `multi_wave` | `enw_simulate(growth_rate)` |  |
| Parametric delay family | One of `lognormal`, `gamma`, `exponential` | `enw_reference(distribution)` |  |
| Delay form | One of `parametric`, `non_parametric`, `both` | `enw_reference(parametric, non_parametric)` |  |
| Delay over time | One of `static`, `drift`, `step`, `periodic`, `walk` | `enw_reference()` |  |
| Report days | One of `daily`, `weekdays` | `enw_report(structural)` |  |
| Day-of-week effect on expected counts | One of `no`, `yes` | `enw_expectation(observation)` |  |
| Observation model | One of `negbin`, `negbin1d`, `poisson` | `enw_obs(family)` |  |
| Strata | One of `single`, `places`, `ages`, `places_ages` | `enw_preprocess_data(by)` |  |
| Places | `switch(cell$strata, single = 1L, ages = 1L, places = sample(2:26, 1), places_ages = sample(6:16, 1) )` | `enw_preprocess_data(by)` |  |
| Age groups | `switch(cell$strata, single = 1L, places = 1L, ages = sample(2:8, 1), places_ages = sample(3:min(7L, 60L %/% n_places), 1) )` | `enw_preprocess_data(by)` |  |
| Weekdays with reports | `if (cell$reporting == "weekdays") { paste(sort(sample(days, sample(1:3, 1))), collapse = ";") } else { "all" }` | `enw_report(structural)` |  |
| Weeks of cut-offs | `if (places_ages) sample(16:24, 1) else sample(16:32, 1)` | `enw_preprocess_data()` grid |  |
| Start day offset (days) | `sample(0:6, 1)` | `enw_preprocess_data()` grid |  |
| Probability a place is quiet | `runif(1, 0, 0.9)` | `expr_lelatent_int` |  |
| Weight of the shared growth path | `runif(1, 0.5, 1)` | `enw_simulate(growth_rate)` |  |
| Sd of the strata's log seed weights | `runif(1, 0.5, 2)` | `expr_lelatent_int` |  |
| Generation time mean (days) | `if (renewal) { as.numeric(sub("renewal_", "", design$process)) * exp(runif(1, -0.2, 0.2)) } else { NA_real_ }` | `enw_expectation(generation_time)` |  |
| Steady daily growth or decline rate | `runif(1, log(2) / 28, log(2) / 7)` | `enw_simulate(growth_rate)` |  |
| Daily growth rate of a flat trajectory | `rnorm(1, 0, 0.005)` | `enw_simulate(growth_rate)` |  |
| Time of the peak or trough, as a share | `runif(1, 0.3, 0.7)` | `enw_simulate(growth_rate)` |  |
| Amplitude of oscillations and waves | `runif(1, 0.02, 0.08)` | `enw_simulate(growth_rate)` |  |
| Period of oscillations (days) | `runif(1, 28, 120)` | `enw_simulate(growth_rate)` |  |
| Phase of oscillations and waves | `runif(1, 0, 2 * pi)` | `enw_simulate(growth_rate)` |  |
| Waves of a multi-wave trajectory | `sample(2:3, 1)` | `enw_simulate(growth_rate)` |  |
| Sd of the weekly AR(1) growth noise | `runif(1, 0.002, 0.01)` | `enw_simulate(growth_rate)` |  |
| Expected outbreak size, which scales the seed | `log_uniform(30, 1e5) * max(1, design$n_places * design$n_ages / 26)` | `expr_lelatent_int` |  |
| Lognormal delay median (days) | `log_uniform(1, 10)` | `refp_mean_int` |  |
| Lognormal delay log sd | `runif(1, 0.5, 2.5)` | `refp_sd_int` |  |
| Gamma, exponential or geometric delay mean (days) | `log_uniform(2, 21)` | `refp_mean_int` or the hazard intercept |  |
| Gamma delay coefficient of variation | `runif(1, 0.8, 1.6)` | `refp_sd_int` |  |
| Sd of the places' delay effects | `runif(1, 0.1, 0.5)` | `(1 \| place)` of `enw_reference()` |  |
| Sd of the age groups' delay effects | `runif(1, 0.05, 0.25)` | `(1 \| age)` of `enw_reference()` |  |
| Sd of the smooth walk over delays | `runif(1, 0.05, 0.3)` | `enw_reference(non_parametric)` |  |
| Sd of the iid hazard effects by delay | `runif(1, 0, 0.15)` | `enw_reference(non_parametric)` |  |
| Delay drift over the outbreak | `sample(c(-1, 1), 1) * runif(1, 0.1, 0.4)` | `trend` covariate of `enw_reference()` |  |
| Delay step size | `sample(c(-1, 1), 1) * runif(1, 0.2, 0.5)` | `step` covariate of `enw_reference()` |  |
| Time of the delay step, as a share | `runif(1, 0.2, 0.8)` | Step covariate of `enw_reference()` |  |
| Amplitude of periodic delay change | `runif(1, 0.1, 0.4)` | `periodic` covariate of `enw_reference()` |  |
| Period of periodic delay change (weeks) | `runif(1, 4, 12)` | `enw_reference()` |  |
| Sd of the weekly delay walk | `runif(1, 0.03, 0.1)` | `rw(week)` of `enw_reference()` |  |
| Sd of the day-of-week effect on expected counts | `runif(1, 0.1, 0.5)` | `enw_expectation(observation)` |  |
| Sd of the report day-of-week effects | `runif(1, 0.1, 0.6)` | `(1 \| day_of_week)` of `enw_report()` |  |
| Overdispersion on the NB2 scale | `log_uniform(0.1, 1)` | `enw_obs()` |  |
| Longest delay (days), from the delay's 99th centile | `max_delay_for(delay_parameters(family, v)$q99)` | `enw_preprocess_data(max_delay)` |  |
