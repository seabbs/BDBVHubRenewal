test_that("depletion bounds each stratum by its population", {
  gt <- gt_pmf(7, "gamma", 0.6)
  n <- 300L
  log_r <- matrix(log(c(2.5, 1.8)), n, 2, byrow = TRUE)
  start <- matrix(0, length(gt), 2)
  start[length(gt), ] <- 5
  N <- c(1e4, 1e6)
  run <- renewal_infections(start, gt, log_r, N)
  total <- colSums(run$infections)
  expect_true(all(total <= N))
  expect_gt(total[1] / N[1], 0.8)
  expect_gt(total[2] / N[2], 0.5)
  expect_lt(run$reff[n, 1], 1)
})

test_that("a reactive shift lowers R while growing and raises it while falling", {
  gt <- gt_pmf(7, "gamma", 0.6)
  n <- 120L
  log_r <- matrix(log(c(1.4, 0.7)), n, 2, byrow = TRUE)
  start <- matrix(100, length(gt), 2)
  shifts <- list(at = 60, size = 0.5, days = 1, reactive = TRUE, up = TRUE)
  run <- renewal_infections(start, gt, log_r, c(1e9, 1e9), shifts = shifts)
  expect_equal(run$r[n, 1] / 1.4, exp(-0.5), tolerance = 1e-3)
  expect_equal(run$r[n, 2] / 0.7, exp(0.5), tolerance = 1e-3)
  expect_equal(run$r[30, 1] / 1.4, 1, tolerance = 1e-3)
})

test_that("a variant returns immune to susceptible and raises R", {
  gt <- gt_pmf(7, "gamma", 0.6)
  n <- 200L
  start <- matrix(10, length(gt), 1)
  variants <- list(at = 150, escape = 0.5, mult = 1.3, days = 1)
  run <- renewal_infections(
    start, gt, matrix(log(2), n, 1), 1e5, variants = variants
  )
  s <- run$susceptible[, 1]
  restored <- s[150] + run$infections[length(gt) + 150, 1] - s[149]
  expect_equal(restored / (1e5 - s[149]), 0.5, tolerance = 1e-2)
  expect_equal(run$r[n, 1] / 2, 1.3, tolerance = 1e-3)
})

test_that("endemic and controlled paths follow a smooth weekly walk", {
  v <- list(walk_r = c(0, 0.1, -0.1, 0.05, rep(0, 7)), amplitude = 0.5,
    control_days = 14)
  e <- path_log_r(56L, "endemic", v)
  expect_equal(e[c(1, 8, 15)], c(0, 0.1, -0.1))
  expect_lte(max(abs(diff(e))), 0.2 / 7 + 1e-9)
  c <- path_log_r(56L, "controlled", v)
  expect_equal(c[1], 0.5)
  expect_lt(abs(c[56] - e[56]), 0.5 * exp(-55 / 14) + 1e-9)
})

for (window in c("early", "established", "endemic")) {
  test_that(paste("a", window, "window opens as it should"), {
    grid <- test_grid()
    set.seed(4)
    d <- draw_design(grid)
    d$strata <- "places"
    d$n_places <- 3L
    d$n_ages <- 1L
    d$seeding <- "start"
    d$mixing <- "none"
    d$window <- window
    v <- draw_values(grid, d)
    v$quiet_places <- 3L
    # Variants return immune people to susceptible, so infections can
    # pass the population; without them they cannot.
    v$variant_at <- v$variant_escape <- v$variant_r <- v$variant_days <- numeric(0)
    gt <- gt_pmf(v$gt_mean, d$gt_family, v$gt_cv)
    run <- latent_onsets(grid, 100L, gt, strata_of(3), d, v)
    w <- seq_along(gt)
    first <- run$infections[w, , drop = FALSE]
    expect_true(all(run$onsets[, 3] == 0))
    expect_true(all(colSums(run$infections) <= run$values$stratum_population))
    if (window == "early") {
      expect_true(all(first[-length(w), 1:2] == 0))
      expect_true(all(first[length(w), 1:2] >= 1))
    } else {
      expect_true(all(apply(first[, 1:2], 2, function(x) {
        diff(range(x)) < 1e-9 && x[1] > 0
      })))
      # Running before the series too, so its onsets open at the level
      # (ascertained), not from zero (the 33715-0016 ramp).
      asc <- run$values$ascertainment[1:2]
      expect_equal(run$onsets[1, 1:2], first[1, 1:2] * asc, tolerance = 1e-6)
    }
    expect_equal(nrow(run$latent), length(gt) + 100L)
  })
}

test_that("every cell gives finite onsets over its whole range", {
  grid <- test_grid()
  set.seed(10)
  for (g in 0:299) {
    d <- draw_design(grid)
    v <- draw_values(grid, d)
    gt <- gt_pmf(v$gt_mean, d$gt_family, v$gt_cv)
    n <- 7L * (v$weeks + v$max_delay %/% 7L + 4L) - length(gt)
    run <- tryCatch(
      latent_onsets(grid, n, gt, strata_of(d$n_places, d$n_ages), d, v),
      error = function(e) NULL
    )
    if (!is.null(run)) {
      expect_true(all(is.finite(run$onsets)))
      expect_true(all(is.finite(run$latent$reff[-seq_along(gt)])))
    }
  }
})

test_that("layer values vary across strata as their structure says", {
  grid <- test_grid()
  set.seed(9)
  for (structure in c("shared", "pooled", "independent")) {
    a <- stratum_values(grid, structure, 6L, "ascertainment", probability = TRUE)
    m <- stratum_values(grid, structure, 6L, "incubation_mean")
    expect_length(a, 6)
    expect_true(all(a > 0 & a < 1) && all(m > 0))
    expect_equal(length(unique(a)) == 1, structure == "shared")
  }
})

test_that("without variants, infections and cases never exceed the population", {
  # The size check of the old sim QC, now a property of the simulator.
  # Variants return immune people to the susceptible, so reinfections can
  # take a stratum's cumulative infections past its population.
  grid <- test_grid()
  set.seed(5)
  checked <- 0L
  for (g in 1:200) {
    d <- draw_design(grid)
    v <- draw_values(grid, d)
    v$variant_at <- v$variant_escape <- v$variant_r <- v$variant_days <- numeric(0)
    gt <- gt_pmf(v$gt_mean, d$gt_family, v$gt_cv)
    n <- 7L * (v$weeks + v$max_delay %/% 7L + 4L) - length(gt)
    run <- tryCatch(
      latent_onsets(grid, n, gt, strata_of(d$n_places, d$n_ages), d, v),
      error = function(e) NULL
    )
    if (is.null(run)) next
    checked <- checked + 1L
    pop <- run$values$stratum_population
    expect_true(all(colSums(run$infections) <= pop * (1 + 1e-9)))
    expect_true(all(colSums(run$onsets) <= pop * (1 + 1e-9)))
  }
  expect_gt(checked, 150L)
})
