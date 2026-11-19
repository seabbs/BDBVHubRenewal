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
