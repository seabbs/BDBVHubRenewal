Realised values follow once the simulations exist.

| Factor | Levels | Enters through | Realised |
|---|---|---|---|
| Generative process | `growth`, `renewal_3`, `renewal_7`, `renewal_15`; balanced, 1/4 each | `enw_expectation()` |  |
| Generation time shape | `gamma_0.3`, `gamma_0.6`, `gamma_1`, `lognormal_0.6`; balanced, 1/4 each | `enw_expectation(generation_time)` |  |
| Trajectory of the growth rate | `growth`, `decline`, `flat`, `rise_fall_sharp`, `rise_fall_gradual`, `fall_rise`, `oscillating`, `multi_wave`; balanced, 1/8 each | `enw_simulate(growth_rate)` |  |
| Parametric delay family | `lognormal`, `gamma`, `exponential`; balanced, 1/3 each | `enw_reference(distribution)` |  |
| Delay form | `parametric`, `non_parametric`, `both`; balanced, 1/3 each | `enw_reference(parametric, non_parametric)` |  |
| Delay over time | `static`, `drift`, `step`, `periodic`, `walk`; balanced, 1/5 each | `enw_reference()` |  |
| Report days | `daily`, `weekdays`; balanced, 1/2 each | `enw_report(structural)` |  |
| Day-of-week effect on expected counts | `no`, `yes`; balanced, 1/2 each | `enw_expectation(observation)` |  |
| Observation model | `negbin`, `negbin1d`, `poisson`; balanced, 1/3 each | `enw_obs(family)` |  |
| Strata | `single`, `places`, `ages`, `places_ages`; balanced, 1/4 each | `enw_preprocess_data(by)` |  |
