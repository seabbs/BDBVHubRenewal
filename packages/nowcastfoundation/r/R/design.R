#' The most strata of places by ages in a run of `weeks`
#'
#' `max_strata` up to 28 weeks, then fewer, so the reporting triangle
#' (strata by days by delays) and the batch's memory stay about as at 28
#' weeks.
#' @param grid A grid (`read_grid`).
#' @param weeks Weeks in the series.
#' @export
max_strata <- function(grid, weeks) {
  min(grid$max_strata, as.integer(grid$max_strata * 28 / weeks))
}

#' Draw a system's places and ages from the grid's counts
#'
#' @param grid A grid (`read_grid`).
#' @param cell The simulation's categorical settings (`draw_cell`).
#' @return Places and ages, two integers.
#' @export
draw_strata <- function(grid, cell) {
  places <- switch(cell$strata,
    single = 1L, ages = 1L,
    places = draw_count(
      grid, if (cell$spread == "concentrated") "places_concentrated" else "places_even"
    ),
    places_ages = draw_count(grid, "places_by_ages")
  )
  ages <- switch(cell$strata,
    single = 1L, places = 1L,
    ages = draw_count(grid, "ages"),
    places_ages = draw_count(grid, "ages_by_places")
  )
  c(places, ages)
}

#' A system's locations in a whole-system training block
#'
#' The country, then for places by ages each place's and each age's total,
#' for flat places up to `regions` region totals, and every leaf.
#'
#' @param places,ages The system's places and ages.
#' @param regions Most region totals the trainer adds over four or more flat
#'   places.
#' @return The number of locations.
#' @export
system_locations <- function(places, ages, regions = 6) {
  leaves <- places * ages
  if (leaves == 1) return(1)
  totals <- if (places > 1 && ages > 1) {
    places + ages
  } else if (places > 1) {
    regions
  } else {
    0
  }
  1 + totals + leaves
}

#' A simulation's design: its categorical settings and the structure they
#' imply
#'
#' @param grid A grid (`read_grid`).
#' @param weeks Fix the series length (pilots); drawn if `NULL`.
#' @return A named list.
#' @export
draw_design <- function(grid, weeks = NULL) {
  cell <- draw_cell(grid)
  if (is.null(weeks)) {
    weeks <- max(
      as.integer(grid$floors$weeks),
      as.integer(round(draw_setting(grid, "weeks")))
    )
  }
  # The strata, drawn again until the whole system fits the simulator's
  # memory and the trainer's block.
  for (try in 1:1000) {
    n <- draw_strata(grid, cell)
    fits <- n[1] * n[2] <= max_strata(grid, weeks) &&
      system_locations(n[1], n[2], grid$regions) <= grid$max_locations
    if (fits) break
  }
  if (!fits) stop("no strata draw fits the budget")
  n_places <- n[1]
  n_ages <- n[2]
  if (cell$strata_link == "independent") {
    # Each independent stratum is a simulation of its own, so at most
    # `max_independent` strata; concentrated outbreaks keep their many
    # places, of which only the active ones are simulated.
    m <- grid$max_independent
    if (cell$strata == "places" && cell$spread != "concentrated") {
      n_places <- min(n_places, m)
    }
    if (cell$strata == "ages") n_ages <- min(n_ages, m)
    if (cell$strata == "places_ages") {
      n_places <- 2L
      n_ages <- min(n_ages, m %/% 2L)
    }
  }
  week <- c("Monday", "Tuesday", "Wednesday", "Thursday", "Friday",
    "Saturday", "Sunday")
  c(
    cell,
    list(
      weeks = weeks,
      n_places = n_places,
      n_ages = n_ages,
      report_weekdays = switch(cell$report_days,
        daily = "all",
        weekdays = paste(week[1:5], collapse = ";"),
        fixed_days = paste(
          week[sort(sample.int(5L, sample(2:3, 1L)))], collapse = ";"
        ),
        weekly = week[sample.int(7L, 1L)]
      )
    )
  )
}

# The rising shape an early outbreak not brought under control follows:
# its drawn trajectory if it rises, else that trajectory's rising
# counterpart.
rising_path <- c(
  growth = "growth", decline = "growth", flat = "rise_fall_gradual",
  rise_fall_sharp = "rise_fall_sharp",
  rise_fall_gradual = "rise_fall_gradual", fall_rise = "rise_fall_sharp",
  oscillating = "multi_wave", multi_wave = "multi_wave"
)

#' The continuous settings of a simulation, by layer
#'
#' Every continuous setting from its sampling distribution in the grid,
#' counts at least 1 (or a base) as the base plus a Poisson, and choices
#' as Bernoulli draws with the grid's shares.
#'
#' @param grid A grid (`read_grid`).
#' @param design A design (`draw_design`).
#' @return A named list of values.
#' @export
draw_values <- function(grid, design) {
  ds <- function(name, n = 1L) draw_setting(grid, name, n)
  n_p <- design$n_places
  n_s <- n_p * design$n_ages
  early <- design$window == "early"
  endemic <- design$window == "endemic"
  weeks <- design$weeks
  v <- list(
    weeks = weeks,
    # The calendar weekday the series starts on; strata of one outbreak
    # share it (`design$offset`).
    offset = if (is.null(design$offset)) sample(0:6, 1) else design$offset
  )

  # Layer 1: infections.
  active <- min(n_p, draw_count(grid, "active_places"))
  v$quiet <- if (design$spread == "concentrated") 1 - active / n_p else ds("quiet")
  v$coupling <- ds("coupling")
  v$place_rho <- ds("place_rho")
  v$place_sd <- ds("place_sd")
  v$place_lags <- c(0L, draw_count(grid, "place_lag_days", n_p - 1L))
  v$late_at <- ds("late_at")
  v$late_days <- draw_count(grid, "late_days")
  v$mixing_eps <- ds("mixing_eps")
  v$mixing_order <- sample.int(n_p)
  v$mixing_x <- ds("mixing_xy", n_p)
  v$mixing_y <- ds("mixing_xy", n_p)
  v$mixing_range <- ds("mixing_range")
  v$mixing_switch <- if (draw_share(grid, "mixing_switch_on")) "on" else "off"
  v$mixing_switch_at <- ds("mixing_switch_at")
  v$mixing_switch_days <- draw_count(grid, "mixing_switch_days")
  v$import_rate <- ds("import_rate")
  v$import_at <- sort(ds("import_at", draw_count(grid, "import_pulses")))
  v$import_size <- ds("import_size")
  v$stratum_sd <- ds(
    if (design$spread == "concentrated") "stratum_sd_concentrated" else "stratum_sd_even"
  )
  v$path <- if (early) {
    if (draw_share(grid, "controlled")) "controlled" else rising_path[[design$trajectory]]
  } else if (endemic) {
    "endemic"
  } else {
    design$trajectory
  }
  v$gt_mean <- ds("gt_mean")
  v$gt_cv <- ds("gt_cv")
  v$amplitude <- ds("amplitude")
  v$flat <- ds("flat")
  v$peak <- ds("peak")
  v$period <- max(grid$floors$period_days, ds("period"))
  v$phase <- ds("phase")
  v$waves <- 2L + rpois(1, grid$rates$extra_waves)
  v$wave_scale <- ds("wave_scale", v$waves + 1L)
  v$flip_days <- ds("flip_days")
  # Weeks of the series, the longest delay (12 weeks) and the horizon.
  v$walk_r <- log_r_walk(grid, weeks + 18L)
  v$control_days <- ds("control_days")
  v$r_drift <- if (!endemic && draw_share(grid, "drift")) ds("r_drift") else 1
  v$population <- ds(if (endemic) "endemic_population" else "population", n_p)
  v$seed <- max(grid$floors$seed, ds("seed"))
  v$level <- ds(if (endemic) "endemic_level" else "level")
  v$trickle <- ds("trickle")
  n_shifts <- rpois(1, grid$rates$shifts)
  v$shift_at <- ds("shift_at", n_shifts)
  v$shift_size <- ds("shift_size", n_shifts)
  v$shift_days <- ds("shift_days", n_shifts)
  v$shift_reactive <- draw_share(grid, "reactive", n_shifts)
  v$shift_up <- draw_share(grid, "shift_up", n_shifts)
  n_variants <- if (early) 0L else rpois(1, grid$rates$variants_per_year * weeks / 52)
  v$variant_at <- ds("variant_at", n_variants)
  v$variant_escape <- ds("escape", n_variants)
  v$variant_r <- ds("variant_r", n_variants)
  v$variant_days <- ds("shift_days", n_variants)
  # Places with no cases; the first has cases if every place would not.
  quiet <- which(rbinom(n_p, 1, v$quiet) == 1)
  if (length(quiet) == n_p) quiet <- setdiff(quiet, 1L)
  v$quiet_places <- quiet
  # Places seeded mid-outbreak: one to three places with cases after the
  # first, which starts at the outset.
  cases <- setdiff(seq_len(n_p), quiet)
  later <- cases[-1]
  v$late_places <- if (length(later)) {
    later[sample.int(length(later), min(length(later), draw_count(grid, "late_places")))]
  } else {
    integer(0)
  }

  # Layer 2: observation.
  v$ascertainment <- stratum_values(
    grid, design$observation_strata, n_s, "ascertainment", probability = TRUE
  )
  v$incubation_mean <- stratum_values(
    grid, design$observation_strata, n_s, "incubation_mean"
  )
  v$sqrt_phi <- ds("sqrt_phi")
  v$obs_phi_sd <- ds("obs_phi_sd")
  v$obs_sqrt_phi <- if (design$obs_strata == "shared") {
    rep(v$sqrt_phi, n_s)
  } else {
    v$sqrt_phi * exp(v$obs_phi_sd * ds("pool_z", n_s))
  }
  v$obs_family <- rep(design$observation, n_s)

  # Layer 3: reporting delays.
  v$delay_median <- ds("delay_median")
  v$delay_sdlog <- ds("delay_sdlog")
  v$delay_mean <- ds("delay_mean")
  v$delay_cv <- 1 / sqrt(ds("delay_shape"))
  v$delay_place_sd <- ds("delay_place_sd")
  v$delay_age_sd <- ds("delay_age_sd")
  v$hazard_walk_sd <- ds("hazard_walk_sd")
  v$hazard_sd <- ds("hazard_sd")
  v$drift <- (2 * rbinom(1, 1, 0.5) - 1) * ds("drift")
  v$step <- (2 * rbinom(1, 1, 0.5) - 1) * ds("step")
  v$step_at <- ds("step_at")
  v$periodic <- ds("periodic")
  v$periodic_weeks <- ds("periodic_weeks")
  v$walk_sd <- ds("walk_sd")
  v$delay_walk <- v$walk_sd * cumsum(c(0, ds("walk_step", weeks + 17L)))
  v$smooth_amp <- ds("smooth_amp")
  v$smooth_period <- max(grid$floors$period_days, ds("smooth_period"))
  v$smooth_phase <- ds("phase")
  v$slow_at <- ds("slow_at")
  v$slow_weeks <- draw_count(grid, "slow_weeks")
  v$slow_mult <- ds("slow_mult")
  family <- if (design$delay_form == "non_parametric") "none" else design$delay_family
  v$delay_dow_sd <- ds("delay_dow_sd")
  v$max_delay <- max_delay_for(delay_parameters(family, v)$q99)

  # Layer 4: the reporting system.
  v$reference_dow_sd <- ds("reference_dow_sd")
  v$dow_sd <- ds("dow_sd")
  places <- if (design$shocks == "none") {
    integer(0)
  } else {
    cases[sample.int(length(cases), min(length(cases), draw_count(grid, "shock_places")))]
  }
  v$shock_places <- rep(places, draw_count(grid, "shocks_per_place", length(places)))
  n_shocks <- length(v$shock_places)
  v$shock_at <- ds("shock_at", n_shocks)
  v$shock_weeks <- draw_count(grid, "shock_weeks", n_shocks)
  v$shock_mult <- ds("shock_mult", n_shocks)
  v$revise_over <- stratum_values(
    grid, design$reporting_strata, n_s, "revise_over", probability = TRUE
  )
  v$revise_move <- stratum_values(
    grid, design$reporting_strata, n_s, "revise_move", probability = TRUE
  )
  v$revise_keep <- stratum_values(grid, design$reporting_strata, n_s, "revise_keep")
  v
}
