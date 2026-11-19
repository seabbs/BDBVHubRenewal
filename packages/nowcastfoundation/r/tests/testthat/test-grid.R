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
  layers <- c("infections", "observation", "delays", "reporting")
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
