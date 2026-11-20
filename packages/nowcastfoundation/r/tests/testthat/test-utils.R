test_that("JSON arrays have jsonlite's text at every slab size", {
  skip_if_not_installed("jsonlite")
  set.seed(2)
  cases <- list(
    c(-0, runif(20), 0, 1, 1e15, -2.5e-7),
    matrix(sample(c(0, 1, 0.25, -0, 1 / 3), 35, TRUE), 7, 5),
    array(sample(0:9, 24, TRUE), dim = c(2, 3, 4)),
    array(sample(0:9, 120, TRUE), dim = c(3, 2, 4, 5)),
    array(runif(12), dim = c(2, 1, 3, 2)),
    matrix(runif(6), 6, 1),
    matrix(runif(6), 1, 6)
  )
  for (x in cases) {
    for (slab in c(1, 3, 7, 1e6)) {
      f <- withr::local_tempfile()
      file.create(f)
      write_json_array(x, f, slab)
      ours <- gsub("[[:space:]]", "", paste(readLines(f, warn = FALSE), collapse = ""))
      theirs <- as.character(jsonlite::toJSON(x, always_decimal = TRUE, digits = NA))
      expect_identical(ours, theirs)
    }
  }
})

test_that("the epinowcast internals the engine calls exist in the pin", {
  lib <- Sys.getenv("ENW_LIB")
  skip_if_not(nzchar(lib) && dir.exists(file.path(lib, "epinowcast")),
    "needs the pinned epinowcast in ENW_LIB")
  withr::local_libpaths(lib, action = "prefix")
  for (f in c(".resolve_growth_rate_length", ".map_centred_intercepts")) {
    expect_true(
      exists(f, envir = asNamespace("epinowcast"), inherits = FALSE),
      info = paste("epinowcast has no", f)
    )
  }
})
