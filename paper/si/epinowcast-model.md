# The epinowcast model used to simulate outbreaks

We simulated each synthetic outbreak from the generative model of epinowcast @placeholder.
We used `enw_simulate()`, which runs the epinowcast Stan model with fixed parameters and no likelihood, and draws reported counts from its generated quantities.
`enw_simulate()` is on the `feature/829-simulate-forecast` branch of epinowcast, at commit `e084b080`.
The simulation script is `scripts/finetune_simulate.R`, with grid `r3`.
Each simulation is one set of fixed parameters with several observation draws, and we evaluate one draw of each.
Simulations follow a crossed design, in which every level of every factor is balanced within each block of simulations.
Continuous settings are drawn within a simulation's cell, as the SI table lists.
A parameter we did not set is drawn from its epinowcast prior.

## Design factors

- Generative process (`process`, through `enw_expectation()`): `growth`, `renewal_3`, `renewal_7`, `renewal_15`.
- Generation time shape (`gt_shape`, through `enw_expectation(generation_time)`): `gamma_0.3`, `gamma_0.6`, `gamma_1`, `lognormal_0.6`.
- Trajectory of the growth rate (`trajectory`, through `enw_simulate(growth_rate)`): `growth`, `decline`, `flat`, `rise_fall_sharp`, `rise_fall_gradual`, `fall_rise`, `oscillating`, `multi_wave`.
- Parametric delay family (`delay_family`, through `enw_reference(distribution)`): `lognormal`, `gamma`, `exponential`.
- Delay form (`delay_form`, through `enw_reference(parametric, non_parametric)`): `parametric`, `non_parametric`, `both`.
- Delay over time (`delay_change`, through `enw_reference()`): `static`, `drift`, `step`, `periodic`, `walk`.
- Report days (`reporting`, through `enw_report(structural)`): `daily`, `weekdays`.
- Day-of-week effect on expected counts (`reference_dow`, through `enw_expectation(observation)`): `no`, `yes`.
- Observation model (`observation`, through `enw_obs(family)`): `negbin`, `negbin1d`, `poisson`.
- Strata (`strata`, through `enw_preprocess_data(by)`): `single`, `places`, `ages`, `places_ages`.

## Model

Expected onsets come from `enw_expectation()`, driven by a daily growth rate given to `enw_simulate()` as `growth_rate`.
In a renewal process the growth rate is turned into log $R_t$ under the generation time, and expected latent cases follow $\lambda_t = R_t \sum_s g_s \lambda_{t-s}$.
The seed is scaled so the expected outbreak size is a drawn target.
The reporting delay is set by `enw_reference()`, as a parametric delay, a non-parametric hazard over delays, or both, and it may change over time.
Report date effects and report days are set by `enw_report()`.
Counts by onset day and delay are negative binomial (NB2 or NB1) or Poisson, set by `enw_obs()`, with the NB1 dispersion matched to the NB2 scale (epinowcast issue 937).
The longest delay is the shortest of 3, 4, 6, 8 and 12 weeks at or above the delay's 99th centile.
A draw that fails or is out of size is drawn again in its cell, up to five times, after which the cell is skipped.
