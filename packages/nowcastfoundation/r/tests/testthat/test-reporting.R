test_that("revisions change releases but not the final counts", {
  set.seed(7)
  days <- data.table::as.IDate("2026-01-01") + 0:59
  pp <- data.table::CJ(draw = 1L, location = c("P01", "P02"),
    reference_date = days, delay = 0:9)
  pp[, report_date := reference_date + delay][, delay := NULL]
  pp <- pp[report_date <= max(days)]
  pp[, mean := rpois(.N, 20)]
  v <- list(revise_over = 0.05, revise_move = 0.05, revise_keep = 14)
  got <- revise_counts(pp, v)
  final <- function(x) x[, .(n = sum(mean)), keyby = .(location, reference_date)]
  expect_equal(final(got), final(pp))
  data.table::setorder(got, location, reference_date, report_date)
  got[, cum := cumsum(mean), by = .(location, reference_date)]
  expect_true(any(got$mean < 0))
  expect_true(all(got$cum >= 0))
  expect_true(all(got$reference_date <= got$report_date))
})

test_that("the delay walk moves smoothly between weeks", {
  walk <- c(0, 0.3, 0.1, 0.4)
  x <- daily_walk(0:21, walk)
  expect_equal(x[c(1, 8, 15, 22)], walk)
  expect_equal(max(abs(diff(x))), 0.3 / 7)
})
