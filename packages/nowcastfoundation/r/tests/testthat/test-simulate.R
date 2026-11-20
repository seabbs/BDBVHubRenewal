# The whole engine through epinowcast: needs the pinned epinowcast (and
# cmdstan), so it runs where they are installed (ENW_LIB) and is skipped
# elsewhere.
test_that("a batch simulates outbreaks through epinowcast", {
  lib <- Sys.getenv("ENW_LIB")
  skip_if_not(nzchar(lib) && dir.exists(file.path(lib, "epinowcast")),
    "needs the pinned epinowcast in ENW_LIB")
  withr::local_libpaths(lib, action = "prefix")
  grid <- r7()
  model <- suppressMessages(epinowcast::enw_model(threads = FALSE))
  out <- simulate_batch(grid, 99902L, 2L, model = model, weeks = 8L)
  expect_equal(nrow(out$settings), 2L,
    info = paste(unique(out$failures$reason), collapse = " | "))
  expect_gt(nrow(out$counts), 0)
  expect_setequal(unique(out$latent$outbreak), out$settings$outbreak)
  expect_true(all(c("infections", "onsets", "r", "reff", "date") %in% names(out$latent)))
  expect_true(all(out$counts$reference_date <= out$counts$report_date))
  # Each stratum's population is recorded with its location, for the QC.
  for (k in seq_len(nrow(out$settings))) {
    where <- strsplit(out$settings$stratum_location[k], ";")[[1]]
    size <- strsplit(as.character(out$settings$stratum_population[k]), ";")[[1]]
    expect_equal(length(where), length(size))
    seen <- unique(out$counts[outbreak == out$settings$outbreak[k] |
      startsWith(outbreak, paste0(out$settings$outbreak[k], "-")), location])
    expect_true(all(seen %in% where))
  }
})

test_that("independent strata share the outbreak's dates", {
  lib <- Sys.getenv("ENW_LIB")
  skip_if_not(nzchar(lib) && dir.exists(file.path(lib, "epinowcast")),
    "needs the pinned epinowcast in ENW_LIB")
  withr::local_libpaths(lib, action = "prefix")
  grid <- r7()
  set.seed(13)
  d <- draw_design(grid, weeks = 8L)
  d$strata <- "places"; d$n_places <- 3L; d$n_ages <- 1L
  d$spread <- "even"; d$strata_link <- "independent"
  ctx <- list(model = suppressMessages(epinowcast::enw_model(threads = FALSE)),
    draws = 1L, debug = FALSE, expected_scale = grid$constants$expected_scale)
  run <- simulate_cell(grid, d, ctx, "t")
  skip_if(is.null(run$out), "no outbreak in this draw")
  first <- run$out$latent[, .(start = min(date), end = max(date)), by = location]
  expect_equal(data.table::uniqueN(first$start), 1L)
  expect_equal(run$out$values$stratum_location, sprintf("P%02d", 1:3))
  expect_length(run$out$values$stratum_population, 3L)
})

test_that("a draw reporting fewer than min_cases is drawn again", {
  lib <- Sys.getenv("ENW_LIB")
  skip_if_not(nzchar(lib) && dir.exists(file.path(lib, "epinowcast")),
    "needs the pinned epinowcast in ENW_LIB")
  withr::local_libpaths(lib, action = "prefix")
  grid <- r7()
  grid$floors$min_cases <- 1e12
  grid$tries <- 2L
  grid$designs <- 1L
  model <- suppressMessages(epinowcast::enw_model(threads = FALSE))
  expect_error(
    simulate_batch(grid, 99903L, 1L, model = model, weeks = 8L),
    "no design in 1"
  )
})

test_that("the reporting delay varies by reference weekday when drawn", {
  lib <- Sys.getenv("ENW_LIB")
  skip_if_not(nzchar(lib) && dir.exists(file.path(lib, "epinowcast")),
    "needs the pinned epinowcast in ENW_LIB")
  withr::local_libpaths(lib, action = "prefix")
  grid <- r7()
  ctx <- list(model = suppressMessages(epinowcast::enw_model(threads = FALSE)),
    draws = 1L, debug = FALSE, expected_scale = grid$constants$expected_scale)
  spread <- function(dow) {
    set.seed(21)
    d <- draw_design(grid, weeks = 12L)
    d$strata <- "single"; d$n_places <- 1L; d$n_ages <- 1L
    d$window <- "established"; d$trajectory <- "flat"
    d$delay_form <- "parametric"; d$delay_family <- "lognormal"
    d$delay_change <- "static"; d$reporting <- "daily"; d$shocks <- "none"
    d$bunching <- "no"; d$revisions <- "no"; d$reference_dow <- "no"
    d$imports <- "none"; d$delay_dow <- dow; d$report_dow <- "no"
    v <- draw_values(grid, d)
    v$level <- 2000; v$ascertainment <- 0.5; v$dow_sd <- 1e-6
    v$delay_dow_sd <- 1
    v$variant_at <- v$variant_escape <- v$variant_r <- v$variant_days <- numeric(0)
    v$shift_at <- v$shift_size <- v$shift_days <- numeric(0)
    v$shift_reactive <- v$shift_up <- logical(0)
    out <- simulate_outbreak(grid, d, v, ctx)$counts
    out[, delay := as.integer(report_date - reference_date)]
    by_day <- out[, .(mean = sum(delay * count) / sum(count)),
      by = .(dow = weekdays(reference_date))]
    max(by_day$mean) / min(by_day$mean)
  }
  expect_gt(spread("yes"), 1.2)
  expect_lt(spread("no"), 1.1)
})
