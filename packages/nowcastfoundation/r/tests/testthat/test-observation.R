test_that("onsets follow infections by the incubation delay", {
  pmf <- incubation_pmf(5)
  expect_true(abs(sum((seq_along(pmf) - 1) * pmf) - 4.6) < 0.5)
  x <- matrix(0, 40, 2)
  x[10, ] <- c(100, 50)
  y <- convolve_rows(x, list(pmf, pmf))
  expect_equal(colSums(y), c(100, 50), tolerance = 1e-6)
  expect_true(all(y[1:9, ] == 0))
})

test_that("onsets are ascertained infections after the incubation delay", {
  grid <- test_grid()
  set.seed(6)
  d <- draw_design(grid)
  d$strata <- "places"
  d$n_places <- 2L
  d$n_ages <- 1L
  d$seeding <- "start"
  d$mixing <- "none"
  d$window <- "established"
  v <- draw_values(grid, d)
  v$quiet_places <- integer(0)
  gt <- gt_pmf(v$gt_mean, d$gt_family, v$gt_cv)
  run <- latent_onsets(grid, 100L, gt, strata_of(2), d, v)
  a <- run$values$ascertainment
  pmfs <- lapply(run$values$incubation_mean, incubation_pmf)
  # An established window's history before the span is its level.
  before <- run$infections[1, ]
  expect_equal(
    run$onsets, sweep(convolve_rows(run$infections, pmfs, before), 2, a, `*`)
  )
})

test_that("a constant history before the span convolves to the same level", {
  pmf <- incubation_pmf(8)
  x <- matrix(c(100, 40), 30, 2, byrow = TRUE)
  y <- convolve_rows(x, list(pmf, pmf), before = c(100, 40))
  expect_equal(y, x, tolerance = 1e-9)
  # Without the history the first rows ramp up from near zero.
  expect_lt(convolve_rows(x, list(pmf, pmf))[1, 1], 5)
})

test_that("observation is NB1 four times as often as Poisson", {
  grid <- test_grid()
  expect_equal(unlist(grid$factors$observation), c(negbin1d = 4, poisson = 1))
  expect_equal(grid$distributions$sqrt_phi$dist, "halfnormal")
  expect_equal(grid$distributions$sqrt_phi$scale, 0.25)
})
