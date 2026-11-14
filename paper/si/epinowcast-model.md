# The epinowcast model used to simulate outbreaks

We simulated each synthetic outbreak from the generative model of epinowcast @placeholder.
We used `enw_simulate()`, which runs the epinowcast Stan model with fixed parameters and no likelihood.
It draws reported counts from the model's generated quantities, so the simulated data come from the same model epinowcast fits.
`enw_simulate()` is on the `feature/829-simulate-forecast` branch of epinowcast, at commit `e084b08089796ca7def8dac43fd62d29ae34258b`.
The simulation script is `scripts/finetune_simulate.R`, with the grid named r2-real, and the settings drawn for each outbreak are in the synthetic scenario grid table.
Each outbreak first draws its design, the categorical choices below, and then its continuous settings.
Outbreaks are simulated on a daily time step by stratum, with a longest reporting delay of 8 weeks, or 12 weeks in a quarter of outbreaks.
A parameter we did not set is drawn from its epinowcast prior.

## Strata

An outbreak has one stratum, 2 to 26 places, 2 to 8 age groups in one place, or 6 to 16 places by 3 to 7 age groups.
Strata share part of their growth through a weighted sum of a shared path and a path of their own, and places by ages also share a path by place.
Some places are quiet, seeded with almost no cases.
The initial latent cases (`expr_lelatent_int`) are a drawn total split over strata by log-normal weights.

## Expectation model

Expected cases come from `enw_expectation()`, driven by a daily growth path given to `enw_simulate()` as `growth_rate`.
In the two renewal processes the path is log $R_t$, and expected latent cases are $\lambda_t = R_t \sum_{s=1}^{45} g_s \lambda_{t-s}$ with a discretised gamma generation time $g_s$.
A log $R_t$ path starts at a drawn value, has a Poisson number of step changes and a daily random walk, and is held between -1.2 and 1.2.
One renewal process adds an incubation period, a discretised gamma delay from latent cases to onsets (`latent_reporting_delay`).
The other four processes have no generation time, and the path is the daily growth rate of log incidence.
It is a weekly random walk, white noise (a random walk on log incidence), one to three step changes, or a smooth sine wave, held between -0.06 and 0.06 a day.

## Reference date model

The reporting delay of an onset is set by `enw_reference()`.
It is a parametric delay alone, a non-parametric hazard alone, or both.
The parametric delay is a lognormal, gamma or exponential, with a drawn mean and coefficient of variation.
The non-parametric hazard has random effects by delay, by onset week, or both, on the logit hazard of being reported.
With no parametric delay, the hazard's intercept gives a geometric delay with the drawn mean.
Strata have random effects on the delay.
One outbreak in five has a stable delay.
In the others the delay may drift linearly in its location and spread, step up to three times at random weeks, and follow a weekly random walk (`rw(week)`).

## Report date model

Effects of the report date are set by `enw_report()`.
They may include random effects by day of the week of report, a `holiday` covariate that lowers reporting on a few dates, and report week effects, `(1 | week)` or `rw(week)`.
In half the outbreaks reports are made every day.
In the rest, a structural table from `enw_structural_reporting_metadata()`, given to `enw_report(structural)`, limits reports to one to three weekdays, drawn from Monday to Friday.
The report days may change once, at a drawn date.

## Observation model

The count of cases with onset on day $t$ reported with delay $d$ is negative binomial (NB2 or NB1) or Poisson, set by `enw_obs(family)`.
Its mean is the expected onsets on day $t$ times the probability of a delay of $d$ given the reference and report date models.
The overdispersion `sqrt_phi` is drawn on the NB2 scale, and the NB1 dispersion is scaled to match it (epinowcast issue 937).
A draw that fails or gives an outbreak outside its size range is drawn again with the same design, up to five times.
