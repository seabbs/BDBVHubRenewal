# Simulate outbreaks with epinowcast to fine-tune Chronos-2
# (src/bdbvhubrenewal/forecasters/finetune.py).
#
# Each simulation is one set of fixed parameters of epinowcast's
# generative model (`enw_simulate`) with `draws` observation draws, each
# an outbreak. Simulations follow a crossed factorial design (`factors`):
# each factor's levels are spread evenly over every block of `block`
# simulations, independently shuffled per factor, so every factor and
# (nearly) every pair of levels is balanced (a Latin hypercube over the
# factors). Each simulation's cell is in its settings, one column per
# factor; continuous values are drawn within the cell (`draw_values`).
#
# - process: daily growth r(t), or a renewal process whose Rt(t) gives
#   r(t) under a generation time of mean about 3, 7 or 15 days;
# - generation-time shape: gamma with CV 0.3, 0.6 or 1, or lognormal with
#   CV 0.6 (renewal only);
# - trajectory of r(t): steady growth, steady decline, flat, rise then
#   fall (sharp or gradual), fall then rise, oscillating, or two to three
#   waves, with small weekly noise, within -0.1 to 0.1 a day;
# - delay family: lognormal, gamma or exponential;
# - delay form: parametric, non-parametric (a smooth walk over delays and
#   an iid part, with distribution "none"; the family is not used), or
#   both (the walk and iid part as a deviation from the family);
# - delay over time: static, drift, a step, periodic, or a weekly walk
#   (iid weekly hazard effects when the delay is non-parametric only);
# - reporting: every day with day-of-week effects, or on one to three
#   weekdays;
# - reference day-of-week effect on expected counts: no or yes;
# - observation: "negbin" (NB2), "negbin1d" (NB1) or "poisson";
# - strata: one, 2 to 26 places, 2 to 8 ages, or places by ages (at most
#   60), sharing their trajectory and delay through random effects.
#
# The seed is scaled so the expected size is a drawn target, and the
# longest delay is the shortest of 3, 4, 6, 8 and 12 weeks at or above the
# delay's 99th centile. The NB1 dispersion matches the NB2 scale
# (epinowcast #937): sqrt_phi * sqrt(lambda_hat * sum(p_d^2)), with
# lambda_hat the median stratum-day with at least one expected onset.
# A draw that fails or is out of size is drawn again in the same cell, up
# to five times; then the cell is logged and skipped.
#
# The output is each outbreak's daily reporting triangle: a count by
# stratum (`location`), onset day and report date.
#
# `enw_simulate` is on the feature/829-simulate-forecast branch of
# epinowcast, pinned below. Install it into a local library first, from
# a clone of epinowcast:
#
#   git -C <epinowcast clone> archive e084b080 | tar -x -C src
#   R CMD INSTALL --library=output/finetune/rlib src
#
# Run from the repo root:
#
#   Rscript scripts/finetune_simulate.R <n> <seed> <out_dir>

epinowcast_sha <- "e084b080"
grid_name <- "r4"
args <- commandArgs(trailingOnly = TRUE)
n_outbreaks <- as.integer(args[[1]])
seed <- as.integer(args[[2]])
out_dir <- args[[3]]
lib <- Sys.getenv("ENW_LIB", "output/finetune/rlib")
.libPaths(c(lib, .libPaths()))

suppressPackageStartupMessages({
  library(epinowcast)
  library(data.table)
})
options(cmdstanr_warn_inits = FALSE, mc.cores = 1)
enw_set_cache(
  Sys.getenv("ENW_CACHE", "output/finetune/enwcache"), type = "session"
)
model <- suppressMessages(enw_model(threads = FALSE))

# Observation draws per simulation, sharing its costly setup.
draws <- 10L
# Set ENW_DEBUG to see Stan's messages.
debug <- nzchar(Sys.getenv("ENW_DEBUG"))

# The factors of the design and their levels.
factors <- list(
  process = c("growth", "renewal_3", "renewal_7", "renewal_15"),
  gt_shape = c("gamma_0.3", "gamma_0.6", "gamma_1", "lognormal_0.6"),
  trajectory = c(
    "growth", "decline", "flat", "rise_fall_sharp", "rise_fall_gradual",
    "fall_rise", "oscillating", "multi_wave"
  ),
  delay_family = c("lognormal", "gamma", "exponential"),
  delay_form = c("parametric", "non_parametric", "both"),
  delay_change = c("static", "drift", "step", "periodic", "walk"),
  reporting = c("daily", "weekdays"),
  reference_dow = c("no", "yes"),
  observation = c("negbin", "negbin1d", "poisson"),
  strata = c("single", "places", "ages", "places_ages"),
  # How cases spread over strata: "concentrated" puts nearly all in a few,
  # as in BDBV, where most provinces report no case in most weeks.
  spread = c("even", "concentrated")
)

# Simulations per balanced block: a multiple of every factor's levels.
block <- 120L

# The cell of simulation `g`: each factor's level from its shuffled,
# balanced sequence over the block holding `g`, seeded by block and
# factor so it does not touch the simulation's random stream.
cell_for <- function(g) {
  b <- g %/% block
  at <- g %% block + 1L
  if (!exists(".Random.seed", envir = globalenv())) set.seed(NULL)
  saved <- get(".Random.seed", envir = globalenv())
  on.exit(assign(".Random.seed", saved, envir = globalenv()))
  out <- lapply(seq_along(factors), function(k) {
    set.seed(b * 100L + k)
    sample(rep_len(factors[[k]], block))[at]
  })
  names(out) <- names(factors)
  out
}

# A generation time on 1, ..., n days from its shape's family and CV.
gt_pmf <- function(mean, shape) {
  family <- sub("_.*", "", shape)
  cv <- as.numeric(sub(".*_", "", shape))
  cdf <- switch(family,
    gamma = function(d) pgamma(d, 1 / cv^2, 1 / (cv^2 * mean)),
    lognormal = function(d) {
      sdlog <- sqrt(log(1 + cv^2))
      plnorm(d, log(mean) - sdlog^2 / 2, sdlog)
    }
  )
  n <- min(60L, max(5L, ceiling(mean * (1 + 4 * cv))))
  pmf <- diff(cdf(0:n))
  pmf / sum(pmf)
}

# The log Rt that gives daily growth `r` under generation time `gt`
# (the inverse of its discrete Laplace transform).
log_r_from_growth <- function(r, gt) {
  s <- seq_along(gt)
  vapply(r, function(x) -log(sum(gt * exp(-x * s))), numeric(1))
}

# Daily growth r(t) of a trajectory over `n` days, with weekly AR(1)
# noise, within -0.1 to 0.1.
trajectory_path <- function(n, kind, v) {
  t <- seq_len(n) / n
  tilt <- (v$peak - t) / max(v$peak, 1 - v$peak)
  r <- switch(kind,
    growth = rep(v$rate, n),
    decline = rep(-v$rate, n),
    flat = rep(v$flat, n),
    rise_fall_sharp = ifelse(t < v$peak, v$rate, -v$rate),
    rise_fall_gradual = v$rate * tilt,
    fall_rise = -v$rate * tilt,
    oscillating = v$amplitude * sin(2 * pi * seq_len(n) / v$period + v$phase),
    multi_wave = v$amplitude * sin(2 * pi * v$waves * t + v$phase)
  )
  pmin(pmax(r + weekly_noise(n, v$noise_sd), -0.1), 0.1)
}

# Weekly AR(1) noise (autocorrelation 0.8) on `n` days.
weekly_noise <- function(n, sd) {
  weeks <- ceiling(n / 7)
  e <- numeric(weeks)
  for (w in seq_len(weeks)[-1]) e[w] <- 0.8 * e[w - 1] + rnorm(1, 0, sd)
  rep(e, each = 7)[seq_len(n)]
}

# The delay's epinowcast parameters (`refp_mean_int`, `refp_sd_int`), its
# 99th centile and its probabilities on 0, ..., max_delay - 1 days.
# Lognormal is meanlog and sdlog; gamma is shape exp(mean_int) and rate
# sd_int; the exponential rate is exp(-mean_int). With no parametric
# delay ("none"), the hazard's intercept is the constant hazard of a
# geometric delay with the drawn mean.
delay_parameters <- function(family, v, max_delay = NULL) {
  out <- switch(family,
    lognormal = list(
      mean_int = log(v$delay_median), sd_int = v$delay_sdlog,
      q99 = qlnorm(0.99, log(v$delay_median), v$delay_sdlog),
      cdf = function(d) plnorm(d, log(v$delay_median), v$delay_sdlog)
    ),
    gamma = {
      shape <- 1 / v$delay_cv^2
      rate <- shape / v$delay_mean
      list(
        mean_int = log(shape), sd_int = rate,
        q99 = qgamma(0.99, shape, rate),
        cdf = function(d) pgamma(d, shape, rate)
      )
    },
    exponential = list(
      mean_int = log(v$delay_mean),
      q99 = qexp(0.99, 1 / v$delay_mean),
      cdf = function(d) pexp(d, 1 / v$delay_mean)
    ),
    none = {
      hazard <- 1 / (1 + v$delay_mean)
      list(
        hazard = hazard, q99 = qgeom(0.99, hazard),
        cdf = function(d) pgeom(d - 1, hazard)
      )
    }
  )
  if (!is.null(max_delay)) {
    pmf <- diff(out$cdf(0:max_delay))
    out$pmf <- pmf / sum(pmf)
  }
  out
}

# The longest delay in days: the shortest of 3, 4, 6, 8 and 12 weeks at
# or above the delay's 99th centile, at most 12 weeks.
max_delay_for <- function(q99) {
  options <- c(21L, 28L, 42L, 56L, 84L)
  options[min(which(c(options >= q99, TRUE))[1], length(options))]
}

# A simulation's design: its cell and the categorical choices within it.
draw_design <- function(g) {
  cell <- cell_for(g)
  n_places <- switch(cell$strata,
    single = 1L, ages = 1L,
    # Concentrated outbreaks spread over many places, as BDBV's provinces.
    places = if (cell$spread == "concentrated") {
      sample(15:26, 1)
    } else {
      sample(2:26, 1)
    },
    places_ages = sample(6:16, 1)
  )
  n_ages <- switch(cell$strata,
    single = 1L, places = 1L,
    ages = sample(2:8, 1),
    # At most 60 strata; `sample` on a single number n draws from 1:n,
    # so the range is indexed instead.
    places_ages = {
      ages <- 3:max(3L, min(7L, 60L %/% n_places))
      ages[sample.int(length(ages), 1)]
    }
  )
  days <- c("Monday", "Tuesday", "Wednesday", "Thursday", "Friday")
  c(
    list(grid = grid_name, cell_index = g %% block, cell_block = g %/% block),
    cell,
    list(
      n_places = n_places,
      n_ages = n_ages,
      report_days = if (cell$reporting == "weekdays") {
        paste(sort(sample(days, sample(1:3, 1))), collapse = ";")
      } else {
        "all"
      }
    )
  )
}

# The continuous settings of a simulation, drawn again when a draw fails.
draw_values <- function(design) {
  log_uniform <- function(lo, hi) exp(runif(1, log(lo), log(hi)))
  places_ages <- design$strata == "places_ages"
  renewal <- design$process != "growth"
  v <- list(
    weeks = if (places_ages) sample(16:24, 1) else sample(16:32, 1),
    offset = sample(0:6, 1),
    # Places with no cases: concentrated outbreaks leave one to four
    # places active, so most series stay at zero.
    quiet = if (design$spread == "concentrated") {
      1 - sample(1:min(4, design$n_places), 1) / design$n_places
    } else {
      runif(1, 0, 0.9)
    },
    coupling = runif(1, 0.5, 1),
    stratum_sd = if (design$spread == "concentrated") {
      runif(1, 2.5, 4)
    } else {
      runif(1, 0.5, 2)
    },
    gt_mean = if (renewal) {
      as.numeric(sub("renewal_", "", design$process)) * exp(runif(1, -0.2, 0.2))
    } else {
      NA_real_
    },
    # Trajectory: steady rates double or halve in 1 to 4 weeks.
    rate = runif(1, log(2) / 28, log(2) / 7),
    flat = rnorm(1, 0, 0.005),
    peak = runif(1, 0.3, 0.7),
    amplitude = runif(1, 0.02, 0.08),
    period = runif(1, 28, 120),
    phase = runif(1, 0, 2 * pi),
    waves = sample(2:3, 1),
    noise_sd = runif(1, 0.002, 0.01),
    # Even outbreaks grow with their strata; concentrated ones stay small,
    # so active strata also have weeks without cases.
    target_size = if (design$spread == "even") {
      log_uniform(30, 1e5) * max(1, design$n_places * design$n_ages / 26)
    } else {
      log_uniform(30, 3000)
    },
    delay_median = log_uniform(1, 10),
    delay_sdlog = runif(1, 0.5, 2.5),
    delay_mean = log_uniform(2, 21),
    # Gamma delays with shape above 2 fail in the simulate branch.
    delay_cv = runif(1, 0.8, 1.6),
    delay_place_sd = runif(1, 0.1, 0.5),
    delay_age_sd = runif(1, 0.05, 0.25),
    hazard_walk_sd = runif(1, 0.05, 0.3),
    hazard_sd = runif(1, 0, 0.15),
    # Delay over time, as log changes of its location or logit hazard.
    drift = sample(c(-1, 1), 1) * runif(1, 0.1, 0.4),
    step = sample(c(-1, 1), 1) * runif(1, 0.2, 0.5),
    step_at = runif(1, 0.2, 0.8),
    periodic = runif(1, 0.1, 0.4),
    periodic_weeks = runif(1, 4, 12),
    walk_sd = runif(1, 0.03, 0.1),
    reference_dow_sd = runif(1, 0.1, 0.5),
    dow_sd = runif(1, 0.1, 0.6),
    sqrt_phi = log_uniform(0.1, 1)
  )
  family <- if (design$delay_form == "non_parametric") {
    "none"
  } else {
    design$delay_family
  }
  v$max_delay <- max_delay_for(delay_parameters(family, v)$q99)
  v
}

# Expected onsets of each stratum by day, as epinowcast's renewal or
# growth model gives them from the latent seeding and the growth rate.
expected_onsets <- function(size, gt, log_r) {
  n_gt <- length(gt)
  n <- nrow(log_r)
  out <- matrix(0, n_gt + n, length(size))
  out[seq_len(n_gt), ] <- rep(size, each = n_gt)
  rev_gt <- rev(gt)
  for (t in seq_len(n)) {
    past <- out[t:(t + n_gt - 1), , drop = FALSE]
    out[n_gt + t, ] <- exp(log_r[t, ]) * colSums(past * rev_gt)
  }
  out
}

# A formula from its terms, `~ 0` if there are none.
as_formula <- function(intercept, terms) {
  terms <- c(if (intercept) "1" else "0", terms)
  as.formula(paste("~", paste(terms, collapse = " + ")))
}

# Standard deviations of a module's random effects, named by effect.
effect_sds <- function(rdesign, sds) {
  effects <- colnames(rdesign)[-1]
  array(as.numeric(unlist(sds[effects])))
}

simulate_outbreak <- function(design, v) {
  # Four weeks plus the longest delay past the last cut-off, so a block's
  # horizon (four weeks) is complete.
  max_delay <- v$max_delay
  n_time <- 7L * (v$weeks + max_delay %/% 7L + 4L)
  place_ids <- sprintf("P%02d", seq_len(design$n_places))
  age_ids <- sprintf("A%d", seq_len(design$n_ages))
  dates <- as.IDate("2026-01-03") + v$offset + seq_len(n_time) - 1L
  grid <- CJ(
    place = place_ids, age = age_ids, reference_date = dates,
    delay = 0:(max_delay - 1)
  )
  grid[, report_date := reference_date + delay][, delay := NULL]
  grid <- grid[report_date <= max(dates)][, confirm := 0L]
  pobs <- suppressMessages(suppressWarnings(enw_preprocess_data(
    grid, by = c("place", "age"), max_delay = max_delay
  )))
  strata <- unique(pobs$metareference[[1]][, .(.group, place, age)])
  setorder(strata, .group)
  multi_place <- design$n_places > 1
  multi_age <- design$n_ages > 1

  # Expectation: growth r(t) directly, or the Rt(t) giving it.
  gt <- if (design$process == "growth") 1 else gt_pmf(v$gt_mean, design$gt_shape)
  reference_dow <- design$reference_dow == "yes"
  expectation <- enw_expectation(
    r = ~1, generation_time = gt,
    observation = if (reference_dow) ~ (1 | day_of_week) else ~1,
    data = pobs
  )
  n_r <- expectation$data$expr_t
  # Strata share the trajectory and differ by their own weekly noise.
  shared <- trajectory_path(n_r, design$trajectory, v)
  growth <- sapply(seq_len(nrow(strata)), function(i) {
    pmin(pmax(shared + (1 - v$coupling) * weekly_noise(n_r, 0.01), -0.1), 0.1)
  })
  growth <- matrix(growth, nrow = n_r)
  log_r <- if (design$process == "growth") {
    growth
  } else {
    matrix(log_r_from_growth(as.numeric(growth), gt), nrow = n_r)
  }
  place_weight <- exp(rnorm(design$n_places, 0, v$stratum_sd))
  place_weight[runif(design$n_places) < v$quiet] <- 0
  if (all(place_weight == 0)) place_weight[1] <- 1
  age_weight <- exp(rnorm(design$n_ages, 0, v$stratum_sd / 2))
  weight <- place_weight[match(strata$place, place_ids)] *
    age_weight[match(strata$age, age_ids)]
  # The expected outbreak is linear in the seed, so the seed is scaled to
  # give the target size.
  share <- weight / sum(weight)
  v$seed_size <- v$target_size / sum(expected_onsets(share, gt, log_r))
  size <- pmax(v$seed_size * share, exp(-8))
  lelatent <- matrix(
    log(size), nrow = expectation$data$expr_gt_n, ncol = nrow(strata),
    byrow = TRUE
  )

  # Reference date: the delay's form, and its change over time as
  # covariates of the parametric part, or of the hazard if there is none.
  meta <- pobs$metareference[[1]]
  n_weeks <- max(meta$week)
  set(meta, j = "trend", value = meta$week / n_weeks - 0.5)
  set(meta, j = "step", value = as.numeric(meta$week >= v$step_at * n_weeks))
  set(meta, j = "periodic",
      value = sin(2 * pi * meta$week / v$periodic_weeks))
  parametric <- design$delay_form != "non_parametric"
  # epinowcast allows one walk per formula, so a hazard with no parametric
  # part, which already walks over delays, changes by week as iid effects.
  change_terms <- switch(design$delay_change,
    static = NULL, drift = "trend", step = "step", periodic = "periodic",
    walk = if (parametric) "rw(week)" else "(1 | week)"
  )
  change_beta <- switch(design$delay_change,
    drift = v$drift, step = v$step, periodic = v$periodic, 0
  )
  stratum_terms <- c(
    if (multi_place) "(1 | place)", if (multi_age) "(1 | age)"
  )
  hazard <- design$delay_form != "parametric"
  family <- if (parametric) design$delay_family else "none"
  hazard_terms <- c("rw(delay)", "(1 | delay)")
  reference <- enw_reference(
    parametric = if (parametric) {
      as_formula(TRUE, c(change_terms, stratum_terms))
    } else {
      ~0
    },
    distribution = family,
    non_parametric = if (!hazard) {
      ~0
    } else if (parametric) {
      as_formula(FALSE, hazard_terms)
    } else {
      as_formula(TRUE, c(hazard_terms, change_terms, stratum_terms))
    },
    data = pobs
  )
  rd <- reference$data
  delay <- delay_parameters(family, v, max_delay)
  stratum_sd <- list(place = v$delay_place_sd, age = v$delay_age_sd)
  parameters <- list()
  set_change <- function(prefix, cols) {
    beta <- rnorm(length(cols))
    beta[cols %in% c("trend", "step", "periodic")] <- change_beta
    parameters[[paste0(prefix, "_beta")]] <<- array(beta)
  }
  if (parametric) {
    fcols <- colnames(rd$refp_fdesign)
    parameters$refp_mean_int <- delay$mean_int
    set_change("refp_mean", fcols)
    sds <- effect_sds(rd$refp_rdesign, stratum_sd)
    if (length(sds)) {
      parameters$refp_mean_beta_sd <- sds
      parameters$refp_sd_beta_sd <- array(sds / 2)
    }
    if (family != "exponential") {
      parameters$refp_sd_int <- delay$sd_int
      parameters$refp_sd_beta <- array(numeric(length(fcols)))
    }
    if (design$delay_change == "walk") {
      parameters$refp_arima_z <- matrix(
        rnorm(rd$refp_arima_T * rd$refp_arima_G),
        rd$refp_arima_T, rd$refp_arima_G
      )
      parameters$refp_arima_sigma <- array(v$walk_sd)
      # The spread walks with the location's shocks; a zero scale sits on
      # its bound, where Stan cannot start.
      if (family != "exponential") {
        parameters$refp_arima_sd_sigma <- array(v$walk_sd / 2)
      }
    }
  }
  if (hazard) {
    set_change("refnp", colnames(rd$refnp_fdesign))
    parameters$refnp_beta_sd <- effect_sds(
      rd$refnp_rdesign,
      c(list(delay = v$hazard_sd, week = v$walk_sd), stratum_sd)
    )
    if (!parametric) {
      parameters$refnp_int <- qlogis(delay$hazard)
    }
    # The smooth hazard: a walk over delays.
    parameters$refnp_arima_z <- matrix(
      rnorm(rd$refnp_arima_T * rd$refnp_arima_G),
      rd$refnp_arima_T, rd$refnp_arima_G
    )
    parameters$refnp_arima_sigma <- array(v$hazard_walk_sd)
  }
  if (reference_dow) {
    parameters$expl_beta <- array(rnorm(expectation$data$expl_fncol))
    parameters$expl_beta_sd <- array(v$reference_dow_sd)
  }

  # Report date: day-of-week effects, or reports on given weekdays.
  structural <- NULL
  report_terms <- NULL
  if (design$reporting == "daily") {
    report_terms <- "(1 | day_of_week)"
  } else {
    on <- strsplit(design$report_days, ";")[[1]]
    structural <- enw_structural_reporting_metadata(pobs)
    structural[, report := as.integer(weekdays(report_date) %in% on)]
    structural <- structural[, .(.group, date, report_date, report)]
  }
  report <- enw_report(
    non_parametric = as_formula(FALSE, report_terms),
    structural = structural, data = pobs
  )
  if (length(report_terms)) {
    parameters$rep_beta <- array(rnorm(report$data$rep_fncol))
    parameters$rep_beta_sd <- array(v$dow_sd)
  }
  parameters$expr_lelatent_int <- lelatent

  # Observations: the NB1 dispersion on the NB2 scale (epinowcast #937).
  onsets <- expected_onsets(size, gt, log_r)
  v$lambda_hat <- max(1, median(onsets[onsets >= 1]), na.rm = TRUE)
  v$sum_p2 <- sum(delay$pmf^2)
  v$sqrt_phi_obs <- switch(design$observation,
    negbin = v$sqrt_phi,
    negbin1d = v$sqrt_phi * sqrt(v$lambda_hat * v$sum_p2),
    poisson = NA_real_
  )
  if (design$observation != "poisson") {
    parameters$sqrt_phi <- array(v$sqrt_phi_obs)
  }
  sims <- suppressMessages(suppressWarnings(enw_simulate(
    pobs, growth_rate = as.numeric(log_r), parameters = parameters,
    reference = reference, report = report, expectation = expectation,
    obs = enw_obs(family = design$observation, data = pobs), model = model,
    seed = sample.int(.Machine$integer.max, 1), draws = draws,
    show_messages = debug, show_exceptions = debug, refresh = 0
  )))
  # The draws in the order `enw_pp_summary` binds them to the triangle,
  # read directly as summarising them is slow.
  cells <- data.table::copy(sims$new_confirm[[1]])
  data.table::setorderv(cells, c(".group", "reference_date"))
  obs <- sims$fit[[1]]$draws("pp_obs", format = "matrix")
  pp <- rbindlist(lapply(seq_len(nrow(obs)), function(k) {
    cells[, .(.group, place, age, reference_date, report_date,
      draw = k, mean = as.numeric(obs[k, ])
    )]
  }))
  pp[, location := if (multi_place && multi_age) {
    paste(place, age, sep = "-")
  } else if (multi_age) {
    age
  } else {
    place
  }]
  list(
    values = v,
    counts = pp[mean > 0, .(
      outbreak = NA_character_, draw, timestep = "day", location,
      reference_date, report_date, count = as.integer(mean)
    )]
  )
}

# Simulations outside `min_size` to `max_size` cases (the draws' mean) are
# drawn again; many strata allow larger outbreaks.
min_size <- 10
max_size <- function(design) {
  300000 * max(1, design$n_places * design$n_ages / 26)
}

# The software a batch ran on, recorded with each simulation: the same
# seed gives the same counts only on the same R, BLAS and LAPACK.
build_stack <- function(lib) {
  source("scripts/enw_fingerprint.R", local = TRUE)
  info <- sessionInfo()
  list(
    r_version = R.version.string,
    blas = info$BLAS,
    lapack = info$LAPACK,
    enw_fingerprint = enw_fingerprint(lib),
    host = Sys.info()[["nodename"]]
  )
}

if (sys.nframe() == 0L) {
  stack <- build_stack(lib)
  set.seed(seed)
  dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)
  t0 <- Sys.time()
  settings <- list()
  counts <- list()
  failures <- list()
  for (i in seq_len(n_outbreaks)) {
    id <- sprintf("%d-%04d", seed, i)
    design <- draw_design(seed * n_outbreaks + i - 1L)
    t1 <- Sys.time()
    out <- NULL
    for (try in 1:5) {
      v <- draw_values(design)
      res <- tryCatch(simulate_outbreak(design, v), error = function(e) e)
      reason <- if (inherits(res, "error")) {
        conditionMessage(res)
      } else {
        size <- sum(res$counts$count) / draws
        if (size < min_size) {
          "too small"
        } else if (size > max_size(design)) {
          "too large"
        } else {
          NA_character_
        }
      }
      if (is.na(reason)) {
        out <- res
        break
      }
      failures[[length(failures) + 1L]] <- c(
        list(outbreak = id, try = try, reason = reason), design, v
      )
    }
    if (is.null(out)) next
    # Each draw is an outbreak `<seed>-<index>-<draw>`.
    out$counts[, outbreak := paste(id, draw, sep = "-")]
    out$counts[, draw := NULL]
    settings[[length(settings) + 1L]] <- c(
      list(outbreak = id), design, out$values,
      list(
        draws = draws, tries = try, cases = sum(out$counts$count) / draws,
        seconds = as.numeric(Sys.time() - t1, units = "secs")
      ),
      stack
    )
    counts[[length(counts) + 1L]] <- out$counts
  }
  message(sprintf(
    "%d of %d simulations in %.1f s (epinowcast %s, grid %s)",
    length(settings), n_outbreaks,
    as.numeric(Sys.time() - t0, units = "secs"), epinowcast_sha, grid_name
  ))
  # Written to temporary files and renamed into place, counts before
  # settings, so a reader that sees the settings sees the whole batch.
  final <- function(name) file.path(out_dir, sprintf("%s-%d.%s", name, seed,
    c(counts = "csv.gz", settings = "csv", failures = "csv")[[name]]))
  write_atomic <- function(rows, name) {
    # The extension stays last, so fwrite still compresses counts.
    tmp <- file.path(out_dir, paste0(".tmp-", basename(final(name))))
    fwrite(rows, tmp)
    file.rename(tmp, final(name))
  }
  if (length(failures)) {
    write_atomic(rbindlist(failures, fill = TRUE), "failures")
  }
  write_atomic(rbindlist(counts), "counts")
  write_atomic(rbindlist(settings), "settings")
}
