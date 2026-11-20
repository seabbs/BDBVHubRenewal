# The grid config, from the repository root.
grid_file <- function() {
  testthat::test_path("..", "..", "..", "..", "..", "config", "finetune", "grid.yaml")
}
test_grid <- function() read_grid(grid_file())

# A strata table of `places` places by `ages` ages, as epinowcast orders
# its groups.
strata_of <- function(places, ages = 1L) {
  s <- data.table::CJ(
    place = sprintf("P%02d", seq_len(places)), age = sprintf("A%d", seq_len(ages))
  )
  s[, .group := .I][]
}
