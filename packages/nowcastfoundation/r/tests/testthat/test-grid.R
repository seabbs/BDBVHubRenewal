test_that("categorical settings are drawn at the grid's shares", {
  grid <- test_grid()
  set.seed(1)
  cells <- lapply(1:20000, function(i) draw_cell(grid))
  share <- function(k, level) mean(vapply(cells, `[[`, "", k) == level)
  expect_equal(share("window", "early"), 0.25, tolerance = 0.03)
  expect_equal(share("window", "endemic"), 0.35, tolerance = 0.03)
  expect_equal(share("delay_change", "static"), 0.4, tolerance = 0.03)
  expect_equal(share("report_days", "daily"), 0.5, tolerance = 0.03)
  expect_equal(share("report_days", "weekdays"), 0.25, tolerance = 0.03)
  expect_equal(share("report_days", "fixed_days"), 0.15, tolerance = 0.03)
  expect_equal(share("report_days", "weekly"), 0.1, tolerance = 0.03)
  expect_equal(share("revisions", "yes"), 0.3, tolerance = 0.03)
  expect_equal(share("observation", "poisson"), 0.2, tolerance = 0.03)
  expect_false(any(c("process", "gt_shape") %in% names(grid$factors)))
  # One report-days factor: no separate reporting or bunching.
  expect_false(any(c("reporting", "bunching") %in% names(grid$factors)))
  expect_false(any(grepl("^bunch", c(names(grid$distributions), names(grid$counts)))))
})

test_that("every distribution is named, has a layer and a reason", {
  grid <- test_grid()
  known <- c("lognormal", "normal", "halfnormal", "gamma", "beta", "uniform")
  d <- grid$distributions
  expect_true(all(vapply(d, function(x) x$dist %in% known, TRUE)))
  layers <- c("strata", "infections", "observation", "delays", "reporting")
  expect_true(all(vapply(d, function(x) x$layer %in% layers, TRUE)))
  expect_true(all(vapply(d, function(x) nzchar(x$why), TRUE)))
  # Only the cyclic phase is uniform.
  uniform <- names(d)[vapply(d, function(x) x$dist == "uniform", TRUE)]
  expect_equal(uniform, "phase")
})

test_that("draws are not truncated", {
  grid <- test_grid()
  set.seed(1)
  x <- draw_setting(grid, "delay_mean", 1e5)
  expect_true(min(x) < 1 && max(x) > 21)
  expect_error(draw_setting(grid, "nope"), "nope")
})

test_that("series run a median 30 weeks and at least 4", {
  grid <- test_grid()
  set.seed(2)
  w <- vapply(1:3000, function(i) draw_design(grid)$weeks, 1)
  expect_gte(min(w), 4)
  expect_true(median(w) >= 28 && median(w) <= 32)
  expect_true(any(w > 52))
})

test_that("values keep the gamma shape below 2 and periods a month", {
  grid <- test_grid()
  set.seed(3)
  for (g in 0:400) {
    d <- draw_design(grid)
    v <- draw_values(grid, d)
    expect_true(1 / v$delay_cv^2 < 2)
    expect_gte(v$period, 30)
    expect_gte(v$smooth_period, 30)
    expect_gte(v$waves, 2)
    expect_gte(length(v$wave_scale), v$waves + 1L)
    expect_gte(v$seed, 1)
    expect_length(v$ascertainment, d$n_places * d$n_ages)
  }
})

test_that("late places are among the places with cases", {
  grid <- test_grid()
  set.seed(4)
  for (g in 0:300) {
    d <- draw_design(grid)
    v <- draw_values(grid, d)
    expect_false(any(v$late_places %in% v$quiet_places))
    expect_false(1L %in% v$late_places)
  }
})

test_that("a draw is redrawn if it reports fewer than the grid's minimum", {
  grid <- test_grid()
  expect_equal(grid$floors$min_cases, 10)
  expect_equal(redraw_reason(9.5, grid), "fewer than 10 cases")
  expect_equal(redraw_reason(0, grid), "fewer than 10 cases")
  expect_true(is.na(redraw_reason(10, grid)))
})

test_that("every simulated system fits the trainer's block budget whole", {
  grid <- test_grid()
  set.seed(11)
  locs <- leaves <- numeric(0)
  for (i in 1:5000) {
    d <- draw_design(grid)
    locs <- c(locs, system_locations(d$n_places, d$n_ages))
    leaves <- c(leaves, d$n_places * d$n_ages)
    expect_lte(d$n_places * d$n_ages, max_strata(grid, d$weeks))
  }
  expect_lte(max(locs), grid$max_locations)
  # Sizes still vary: single series, small systems and large ones.
  expect_true(any(leaves == 1) && any(leaves >= 20))
})

test_that("a system's locations are the country, group totals and leaves", {
  expect_equal(system_locations(1, 1), 1)
  expect_equal(system_locations(5, 1), 1 + 5 + 6)
  expect_equal(system_locations(1, 4), 1 + 4)
  expect_equal(system_locations(3, 4), 1 + 3 + 4 + 12)
})

test_that("every factor has a layer and aggregates are drawn from the grid", {
  grid <- test_grid()
  expect_setequal(names(grid$factor_layers), names(grid$factors))
  layers <- c("strata", "infections", "observation", "delays", "reporting")
  expect_true(all(unlist(grid$factor_layers) %in% layers))
  a <- grid$aggregation
  expect_setequal(names(a$parents), c("exact", "unallocated", "late_parent"))
  expect_equal(grid$distributions[[a$unallocated_share]]$dist, "beta")
  expect_equal(grid$distributions[[a$parent_lag_days]]$dist, "lognormal")
})

test_that("a draw is redrawn only when it fails or reports too few cases", {
  grid <- test_grid()
  # No sim QC thresholds: size, weekly rises and mean
  # delays are left as drawn.
  expect_null(grid$qc)
  expect_false(exists("qc_reason", envir = asNamespace("nowcastfoundation")))
  expect_true(is.na(redraw_reason(grid$floors$min_cases, grid)))
  expect_match(redraw_reason(grid$floors$min_cases - 1, grid), "fewer than")
})

test_that("strata of one outbreak can share its length and calendar offset", {
  grid <- test_grid()
  set.seed(12)
  d <- draw_design(grid, weeks = 11L)
  expect_equal(d$weeks, 11L)
  d$offset <- 4L
  expect_equal(draw_values(grid, d)$offset, 4L)
})

test_that("the reporting delay may vary by reference weekday", {
  grid <- test_grid()
  expect_setequal(names(grid$factors$delay_dow), c("no", "yes"))
  expect_equal(grid$factor_layers$delay_dow, "delays")
  d <- grid$distributions$delay_dow_sd
  expect_equal(d$layer, "delays")
  expect_true(nzchar(d$why))
  set.seed(3)
  v <- draw_values(grid, draw_design(grid))
  expect_gt(v$delay_dow_sd, 0)
})

test_that("weekday effects are rare and mostly small", {
  grid <- test_grid()
  share <- function(f) {
    w <- unlist(grid$factors[[f]])
    unname(w["yes"] / sum(w))
  }
  expect_equal(share("report_dow"), 1 / 2)
  expect_equal(share("reference_dow"), 1 / 3)
  expect_equal(share("delay_dow"), 1 / 3)
  expect_equal(grid$factor_layers$report_dow, "reporting")
  for (k in c("dow_sd", "reference_dow_sd")) {
    d <- grid$distributions[[k]]
    expect_equal(c(d$dist, d$median, d$sdlog), c("lognormal", 0.12, 0.5))
  }
  d <- grid$distributions$delay_dow_sd
  expect_equal(c(d$dist, d$median, d$sdlog), c("lognormal", 0.08, 0.5))
})

test_that("each report-days level reports on its weekdays", {
  grid <- test_grid()
  week <- c("Monday", "Tuesday", "Wednesday", "Thursday", "Friday",
    "Saturday", "Sunday")
  set.seed(8)
  seen <- list()
  for (i in 1:400) {
    d <- draw_design(grid)
    on <- strsplit(d$report_weekdays, ";")[[1]]
    seen[[d$report_days]] <- c(seen[[d$report_days]], length(on))
    if (d$report_days == "daily") expect_equal(on, "all")
    if (d$report_days == "weekdays") expect_equal(on, week[1:5])
    if (d$report_days == "fixed_days") expect_true(all(on %in% week[1:5]))
    if (d$report_days != "daily") expect_true(all(on %in% week))
  }
  expect_setequal(unique(seen$fixed_days), 2:3)
  expect_equal(unique(seen$weekly), 1L)
})
