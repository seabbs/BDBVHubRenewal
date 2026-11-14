| Design | Levels and shares | Enters through | Realised |
|---|---|---|---|
| Strata | One place 30%; 2 to 26 places 40%; 2 to 8 age groups in one place 20%; 6 to 16 places by 3 to 7 age groups 10% | `enw_preprocess_data(by)` |  |
| Generative process | Renewal, renewal with incubation, weekly growth walk, log incidence walk, growth steps or smooth growth, one sixth each | `enw_expectation()`, `enw_simulate(growth_rate)` |  |
| Delay model | Parametric 30%; non-parametric hazard 20%; both 50% | `enw_reference()` |  |
| Parametric delay | Lognormal, gamma or exponential, one third each | `enw_reference(distribution)` |  |
| Hazard effects | By delay, by week or both, one third each; by delay if the delay is stable | `enw_reference(non_parametric)` |  |
| Delay changes | Stable 20%; otherwise drift in location 60%, drift in spread 50%, 0 to 3 steps (30%, 30%, 20%, 20%) and a weekly walk 30% | `enw_reference(parametric)` |  |
| Report date effects | Day of week 70%; holidays 30%; report week effects none 60%, `(1 \| week)` 20%, `rw(week)` 20% | `enw_report()` |  |
| Report days | Every day 50%; 1 to 3 weekdays 50%; a change of days 30% if not stable | `enw_report(structural)` |  |
| Observation model | NB2 40%; NB1 40%; Poisson 20% | `enw_obs(family)` |  |
| Longest delay | 8 weeks 75%; 12 weeks 25% | `enw_preprocess_data(max_delay)` |  |
