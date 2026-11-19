simulate_outbreak <- function(grid, design, v, ctx) {
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
  pobs <- suppressMessages(suppressWarnings(epinowcast::enw_preprocess_data(
    grid, by = c("place", "age"), max_delay = max_delay
  )))
  rm(grid)
  strata <- unique(pobs$metareference[[1]][, .(.group, place, age)])
  setorder(strata, .group)
  multi_place <- design$n_places > 1
  multi_age <- design$n_ages > 1

  # Latent infections: growth r(t) directly, or the Rt(t) giving it, with
  # mixing between places and imports, run here and injected into
  # epinowcast (`latent_expectation`).
  gt <- gt_pmf(v$gt_mean, design$gt_family, v$gt_cv)
  reference_dow <- design$reference_dow == "yes"
  n_r <- length(unique(pobs$metareference[[1]]$date)) - length(gt)
  latent_run <- latent_onsets(grid, n_r, gt, strata, design, v)
  onsets <- latent_run$onsets
  v <- latent_run$values
  latent <- latent_run$latent[, `:=`(date = dates, location = "all")]
  # Days before the outbreak first expects one onset a day.
  total <- rowSums(onsets)
  v$low_start_days <- if (any(total >= 1)) {
    which(total >= 1)[1] - 1L
  } else {
    length(total)
  }
  latent <- latent_expectation(
    pobs, onsets,
    observation = if (reference_dow) ~ (1 | day_of_week) else ~1
  )
  expectation <- latent$expectation

  # Reference date: the delay's form, and its change over time as
  # covariates of the parametric part, or of the hazard if there is none.
  meta <- pobs$metareference[[1]]
  n_weeks <- max(meta$week)
  set(meta, j = "trend", value = meta$week / n_weeks - 0.5)
  set(meta, j = "step", value = as.numeric(meta$week >= v$step_at * n_weeks))
  set(meta, j = "periodic",
      value = sin(2 * pi * meta$week / v$periodic_weeks))
  {
    # The delay walk, smooth between weeks, and its smooth cycle.
    day <- as.integer(meta$date - min(meta$date))
    set(meta, j = "walk_daily", value = daily_walk(day, v$delay_walk))
    set(meta, j = "smooth", value = sin(
      2 * pi * day / v$smooth_period + v$smooth_phase
    ))
  }
  # A backlog's covariate is the log of its multiplier, so shocks of
  # different sizes share one effect (1, or -1 on the logit hazard).
  ref_dates <- data.table(reference_date = meta$date)
  windows <- lapply(seq_along(v$shock_places), function(k) {
    shock_window(ref_dates, v, k)
  })
  backlog <- numeric(nrow(meta))
  for (k in seq_along(windows)) {
    w <- windows[[k]]
    on <- meta$place == sprintf("P%02d", v$shock_places[k]) &
      meta$date >= w[1] & meta$date <= w[2]
    backlog[on] <- backlog[on] + log(v$shock_mult[k])
  }
  set(meta, j = "backlog", value = backlog)
  slow <- slowdown_window(data.table(reference_date = meta$date), v)
  set(meta, j = "slowdown", value = as.numeric(
    meta$date >= slow[1] & meta$date <= slow[2]
  ))
  parametric <- design$delay_form != "non_parametric"
  # epinowcast allows one walk per formula, so a hazard with no parametric
  # part, which already walks over delays, changes by week as iid effects.
  change_terms <- switch(design$delay_change,
    static = NULL, drift = "trend", step = "step", periodic = "periodic",
    walk = "walk_daily",
    smooth = "smooth",
    slowdown = "slowdown"
  )
  change_beta <- switch(design$delay_change,
    drift = v$drift, step = v$step, periodic = v$periodic, 0
  )
  if (design$shocks == "backlog") change_terms <- c(change_terms, "backlog")
  place_delays <- multi_place && design$delay_places != "shared"
  stratum_terms <- c(
    if (place_delays) "(1 | place)", if (multi_age) "(1 | age)"
  )
  hazard <- design$delay_form != "parametric"
  family <- if (parametric) design$delay_family else "none"
  hazard_terms <- c("rw(delay)", "(1 | delay)")
  # Garbage is collected before each of epinowcast's large steps: after a
  # large step R's next collection is gigabytes away.
  gc()
  reference <- epinowcast::enw_reference(
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
  reference <- drop_builder_frame(reference)
  rd <- reference$data
  delay <- delay_parameters(family, v, max_delay)
  stratum_sd <- list(place = v$delay_place_sd, age = v$delay_age_sd)
  parameters <- list()
  set_change <- function(prefix, cols) {
    beta <- rnorm(length(cols))
    beta[cols %in% c("trend", "step", "periodic")] <- change_beta
    # A backlog multiplies the mean delay: log of it on the location, or
    # its negative on the logit hazard.
    sign <- if (prefix == "refp_mean") 1 else -1
    beta[cols == "backlog"] <- sign
    beta[cols == "slowdown"] <- log(v$slow_mult) * sign
    beta[cols == "walk_daily"] <- sign
    beta[cols == "smooth"] <- v$smooth_amp * sign
    parameters[[paste0(prefix, "_beta")]] <<- array(beta)
  }
  if (parametric) {
    fcols <- colnames(rd$refp_fdesign)
    parameters$refp_mean_int <- delay$mean_int
    set_change("refp_mean", fcols)
    sds <- effect_sds(rd$refp_rdesign, stratum_sd)
    if (length(sds)) {
      parameters$refp_mean_beta_sd <- sds
      # A shifted place moves its mean, its spread half as much; an
      # independent place moves both alike.
      spread <- if (design$delay_places == "independent") 1 else 0.5
      parameters$refp_sd_beta_sd <- effect_sds(rd$refp_rdesign, list(
        place = spread * v$delay_place_sd, age = v$delay_age_sd / 2
      ))
    }
    if (family != "exponential") {
      parameters$refp_sd_int <- delay$sd_int
      parameters$refp_sd_beta <- array(numeric(length(fcols)))
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
  gc()
  structural <- NULL
  report_terms <- NULL
  if (design$reporting == "daily") {
    report_terms <- "(1 | day_of_week)"
  } else {
    on <- strsplit(design$report_days, ";")[[1]]
    structural <- epinowcast::enw_structural_reporting_metadata(pobs)
    structural[, report := as.integer(weekdays(report_date) %in% on)]
    structural <- structural[, .(.group, date, report_date, report)]
  }
  if (design$shocks == "outage") {
    if (is.null(structural)) {
      structural <- epinowcast::enw_structural_reporting_metadata(pobs)
      structural[, report := 1L]
    }
    for (k in seq_along(windows)) {
      w <- windows[[k]]
      groups <- strata[place == sprintf("P%02d", v$shock_places[k]), .group]
      structural[
        .group %in% groups & report_date >= w[1] & report_date <= w[2],
        report := 0L
      ]
    }
  }
  if (design$bunching == "yes") {
    if (is.null(structural)) {
      structural <- epinowcast::enw_structural_reporting_metadata(pobs)
      structural[, report := 1L]
    }
    held <- bunch_dates(structural, v)
    structural[report_date %in% held, report := 0L]
    v$bunch_dates <- as.character(held)
  }
  report <- epinowcast::enw_report(
    non_parametric = as_formula(FALSE, report_terms),
    structural = structural, data = pobs
  )
  report <- drop_builder_frame(report)
  rm(structural)
  if (length(report_terms)) {
    parameters$rep_beta <- array(rnorm(report$data$rep_fncol))
    parameters$rep_beta_sd <- array(v$dow_sd)
  }
  parameters$expr_lelatent_int <- latent$lelatent + log(ctx$expected_scale)

  # Observations: the NB1 dispersion on the NB2 scale (epinowcast #937),
  # per stratum.
  v$lambda_hat <- max(1, median(onsets[onsets >= 1]), na.rm = TRUE)
  v$sum_p2 <- sum(delay$pmf^2)
  v$sqrt_phi_obs <- ifelse(v$obs_family == "negbin1d",
    v$obs_sqrt_phi * sqrt(v$lambda_hat * v$sum_p2), v$obs_sqrt_phi
  )
  # `enw_simulate` in two steps, so the design matrices and the rest of
  # the model's inputs are freed once written, before Stan runs.
  gc()
  stan <- suppressMessages(suppressWarnings(enw_simulate_data(
    pobs, growth_rate = latent$growth_rate, parameters = parameters,
    reference = reference, report = report, expectation = expectation,
    obs = epinowcast::enw_obs(family = "poisson", data = pobs)
  )))
  # The draws in the order `enw_pp_summary` binds them to the triangle,
  # read directly as summarising them is slow.
  cells <- data.table::copy(stan$cells)
  data.table::setorderv(cells, c(".group", "reference_date"))
  stan$cells <- NULL
  rm(pobs, meta, reference, report, expectation, latent, onsets)
  stan$file <- write_stan_data(
    stan$data, tempfile("standata-", fileext = ".json"),
    ctx$model$variables()$data
  )
  stan$data <- NULL
  gc()
  mu <- suppressMessages(suppressWarnings(enw_simulate_draw(
    stan, ctx$model, seed = sample.int(.Machine$integer.max, 1),
    show_messages = ctx$debug, show_exceptions = ctx$debug, refresh = 0
  ))) / ctx$expected_scale
  rm(stan)
  pp <- rbindlist(lapply(seq_len(ctx$draws), function(k) {
    cells[, .(.group, place, age, reference_date, report_date,
      draw = k,
      mean = observe(mu, .group, v$obs_family, v$sqrt_phi_obs)
    )]
  }))
  pp[, location := if (multi_place && multi_age) {
    paste(place, age, sep = "-")
  } else if (multi_age) {
    age
  } else {
    place
  }]
  if (design$revisions == "yes") {
    pp <- revise_counts(pp, v)
  }
  list(
    values = v,
    latent = latent,
    counts = pp[mean != 0, .(
      outbreak = NA_character_, draw, timestep = "day", location,
      reference_date, report_date, count = as.integer(mean)
    )]
  )
}

# Simulate one design: up to five draws of its values, each kept if it
# runs and reports at least `min_cases` (`redraw_reason`). Returns the
# outbreak (or NULL) and a failure row per failed try.
simulate_cell <- function(grid, design, ctx, id = NA_character_) {
  if (design$strata_link == "independent" &&
      design$n_places * design$n_ages > 1) {
    return(simulate_independent(grid, design, ctx, id))
  }
  failures <- list()
  for (try in 1:5) {
    v <- draw_values(grid, design)
    res <- tryCatch(simulate_outbreak(grid, design, v, ctx), error = function(e) e)
    reason <- if (inherits(res, "error")) {
      conditionMessage(res)
    } else {
      redraw_reason(sum(res$counts$count) / ctx$draws, grid)
    }
    if (is.na(reason)) {
      return(list(out = res, failures = failures))
    }
    failures[[length(failures) + 1L]] <- failure_row(
      id, try, reason, design, v
    )
  }
  list(out = NULL, failures = failures)
}

# Independent strata: each stratum is a single-stratum simulation with a
# design of its own (drawn from the grid), run through
# `simulate_cell`, and the strata are combined into one outbreak. A
# stratum that gives no outbreak in five tries is drawn again from a new
# cell, up to five cells; then the outbreak fails. The outbreak's values
# are the first stratum's, with each stratum's cell index, trajectory,
# delay family, observation family and size as ";" text.
simulate_independent <- function(grid, design, ctx, id = NA_character_) {
  place_ids <- sprintf("P%02d", seq_len(design$n_places))
  age_ids <- sprintf("A%d", seq_len(design$n_ages))
  labels <- if (design$n_places > 1 && design$n_ages > 1) {
    as.vector(outer(place_ids, age_ids, paste, sep = "-"))
  } else if (design$n_ages > 1) {
    age_ids
  } else {
    place_ids
  }
  # Quiet places (a concentrated outbreak's many) have no cases, so only
  # the strata of places with cases are simulated.
  quiet <- draw_values(grid, design)$quiet_places
  place_of <- match(sub("-.*", "", labels), place_ids)
  active <- which(!(place_of %in% quiet))
  failures <- list()
  parts <- list()
  latents <- list()
  record <- list()
  strata_values <- list()
  for (s in active) {
    run <- NULL
    for (attempt in 1:5) {
      one <- draw_design(grid)
      one$strata <- "single"
      one$n_places <- 1L
      one$n_ages <- 1L
      one$spread <- "even"
      one$seeding <- "start"
      one$mixing <- "none"
      one$strata_link <- "linked"
      # One family per outbreak.
      one$observation <- design$observation
      res <- simulate_cell(grid, one, ctx, paste(id, labels[s], sep = "/"))
      failures <- c(failures, res$failures)
      if (!is.null(res$out)) {
        run <- res$out
        break
      }
    }
    if (is.null(run)) return(list(out = NULL, failures = failures))
    run$counts[, location := labels[s]]
    run$latent[, location := labels[s]]
    latents[[length(latents) + 1L]] <- run$latent
    parts[[length(parts) + 1L]] <- run$counts
    record[[s]] <- list(
      trajectory = one$trajectory,
      delay_family = one$delay_family,
      observation = run$values$obs_family[1],
      size = sum(run$counts$count) / ctx$draws, max_delay = run$values$max_delay
    )
    strata_values[[length(strata_values) + 1L]] <- run$values
    if (s == active[1]) values <- run$values
  }
  record <- Filter(Negate(is.null), record)
  join <- function(k) paste(vapply(record, function(r) {
    as.character(r[[k]])
  }, ""), collapse = ";")
  values$stratum_trajectory <- join("trajectory")
  values$stratum_delay_family <- join("delay_family")
  values$stratum_observation <- join("observation")
  values$stratum_size <- join("size")
  values$max_delay <- max(vapply(record, function(r) r$max_delay, 1))
  values$quiet_places <- quiet
  list(
    out = list(
      values = values, counts = rbindlist(parts, use.names = TRUE),
      latent = rbindlist(latents, use.names = TRUE)
    ),
    failures = failures,
    strata_values = strata_values
  )
}

#' Simulate a batch of outbreaks from a grid
#'
#' Each simulation draws its categorical settings (`draw_design`), its
#' values by layer (`draw_values`), its latent path by renewal
#' (`latent_onsets`), and its release triangle by epinowcast, observed
#' with the grid's noise, then revised. A draw that errors or gives no case
#' is drawn again, up to five times.
#'
#' @param grid A grid (`read_grid`).
#' @param seed The batch's seed.
#' @param n Simulations.
#' @param model An epinowcast model (`epinowcast::enw_model()`).
#' @param draws Observation draws per simulation.
#' @param weeks Fix every series' length (pilots), or `NULL`.
#' @param debug Show Stan's messages.
#' @return A list: `counts` (outbreak, timestep, location, reference_date,
#'   report_date, count: each outbreak's daily release triangle as
#'   increments), `settings` (one row per simulation), `failures` (one row
#'   per failed try) and `latent` (outbreak, location, date, infections,
#'   onsets, r, reff).
#' @export
simulate_batch <- function(grid, seed, n, model, draws = 1L, weeks = NULL,
                           debug = FALSE) {
  ctx <- list(
    model = model, draws = draws, debug = debug,
    expected_scale = grid$constants$expected_scale
  )
  set.seed(seed)
  settings <- list()
  counts <- list()
  latent <- list()
  failures <- list()
  for (i in seq_len(n)) {
    id <- sprintf("%d-%04d", seed, i)
    design <- draw_design(grid, weeks = weeks)
    t1 <- Sys.time()
    run <- simulate_cell(grid, design, ctx, id)
    failures <- c(failures, run$failures)
    out <- run$out
    if (is.null(out)) next
    # Each draw is an outbreak `<seed>-<index>-<draw>`.
    out$counts[, outbreak := paste(id, draw, sep = "-")]
    out$counts[, draw := NULL]
    counts[[length(counts) + 1L]] <- out$counts
    latent[[length(latent) + 1L]] <- out$latent[, outbreak := id]
    settings[[length(settings) + 1L]] <- flat_values(c(
      list(outbreak = id), design, out$values,
      list(
        draws = draws, tries = length(run$failures) + 1L,
        cases = sum(out$counts$count) / draws,
        seconds = as.numeric(Sys.time() - t1, units = "secs")
      )
    ))
  }
  list(
    counts = data.table::rbindlist(counts),
    settings = data.table::rbindlist(settings, fill = TRUE),
    failures = data.table::rbindlist(failures, fill = TRUE),
    latent = data.table::rbindlist(latent, use.names = TRUE)
  )
}

#' Why a simulation is drawn again, if it is
#'
#' A simulation reporting fewer than `grid$floors$min_cases` cases (its
#' draws' mean) is drawn again; so are draws that error (`simulate_cell`).
#'
#' @param size Reported cases, the draws' mean.
#' @param grid A grid (`read_grid`).
#' @return The reason, or `NA` if the simulation is kept.
#' @export
redraw_reason <- function(size, grid) {
  if (size < grid$floors$min_cases) {
    sprintf("fewer than %s cases", grid$floors$min_cases)
  } else {
    NA_character_
  }
}
