test_that("categorical settings are drawn at the grid's shares", {
  grid <- r7()
  set.seed(1)
  cells <- lapply(1:20000, function(i) draw_cell(grid))
  share <- function(k, level) mean(vapply(cells, `[[`, "", k) == level)
  expect_equal(share("window", "early"), 0.25, tolerance = 0.03)
  expect_equal(share("window", "endemic"), 0.35, tolerance = 0.03)
  expect_equal(share("delay_change", "static"), 0.4, tolerance = 0.03)
  expect_equal(share("bunching", "yes"), 0.15, tolerance = 0.05)
  expect_equal(share("revisions", "yes"), 0.3, tolerance = 0.03)
  expect_equal(share("observation", "poisson"), 0.2, tolerance = 0.03)
  expect_false(any(c("process", "gt_shape") %in% names(grid$factors)))
})

test_that("every distribution is named, has a layer and a reason", {
  grid <- r7()
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
  grid <- r7()
  set.seed(1)
  x <- draw_setting(grid, "delay_mean", 1e5)
  expect_true(min(x) < 1 && max(x) > 21)
  expect_error(draw_setting(grid, "nope"), "nope")
})

test_that("series run a median 30 weeks and at least 4", {
  grid <- r7()
  set.seed(2)
  w <- vapply(1:3000, function(i) draw_design(grid)$weeks, 1)
  expect_gte(min(w), 4)
  expect_true(median(w) >= 28 && median(w) <= 32)
  expect_true(any(w > 52))
})

test_that("values keep the gamma shape below 2 and periods a month", {
  grid <- r7()
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
  grid <- r7()
  set.seed(4)
  for (g in 0:300) {
    d <- draw_design(grid)
    v <- draw_values(grid, d)
    expect_false(any(v$late_places %in% v$quiet_places))
    expect_false(1L %in% v$late_places)
  }
})

test_that("a draw is redrawn if it reports fewer than the grid's minimum", {
  grid <- r7()
  expect_equal(grid$floors$min_cases, 10)
  expect_equal(redraw_reason(9.5, grid), "fewer than 10 cases")
  expect_equal(redraw_reason(0, grid), "fewer than 10 cases")
  expect_true(is.na(redraw_reason(10, grid)))
})

test_that("every simulated system fits the trainer's block budget whole", {
  grid <- r7()
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
  grid <- r7()
  expect_setequal(names(grid$factor_layers), names(grid$factors))
  layers <- c("strata", "infections", "observation", "delays", "reporting")
  expect_true(all(unlist(grid$factor_layers) %in% layers))
  a <- grid$aggregation
  expect_setequal(names(a$parents), c("exact", "unallocated", "late_parent"))
  expect_equal(grid$distributions[[a$unallocated_share]]$dist, "beta")
  expect_equal(grid$distributions[[a$parent_lag_days]]$dist, "lognormal")
})

test_that("a draw failing the sim QC is redrawn, with the grid's thresholds", {
  grid <- r7()
  expect_named(grid$qc, c("max_cases_per_stratum", "max_weekly_jump", "max_mean_delay_days"))
  day <- data.table::as.IDate("2026-01-05")
  counts <- function(ref, delay, n, loc = "P01") {
    data.table::data.table(location = loc, reference_date = day + ref,
      report_date = day + ref + delay, count = n)
  }
  ok <- counts(0:27, 2L, 20L)
  expect_true(is.na(qc_reason(ok, grid)))
  # A thousand-fold rise in one week.
  jump <- rbind(counts(0:6, 1L, 1L), counts(7:13, 1L, 5000L))
  expect_match(qc_reason(jump, grid), "rise")
  # A mean delay past the limit.
  slow <- counts(0:27, 60L, 20L)
  expect_match(qc_reason(slow, grid), "delay")
  # Too many cases per stratum with cases.
  big <- counts(0:27, 2L, 30000L)
  expect_match(qc_reason(big, grid), "per stratum")
})
