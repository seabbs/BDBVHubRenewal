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
