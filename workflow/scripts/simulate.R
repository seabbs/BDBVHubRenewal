# The simulate rule's wrapper: read the grid config, load the engine
# (packages/nowcastfoundation/r) with the pinned epinowcast, simulate one
# batch and write its counts, settings, failures and latent paths, each
# to a temporary file renamed into place (counts before settings, so a
# reader that sees the settings sees the whole batch).
#
#   Rscript workflow/scripts/simulate.R <grid.yaml> <n> <seed> <out_dir>
#
# ENW_LIB (the pinned epinowcast's library), ENW_DRAWS, ENW_WEEKS (a
# pilot's fixed length), ENW_CACHE and ENW_DEBUG as the rule sets them.
args <- commandArgs(trailingOnly = TRUE)
lib <- Sys.getenv("ENW_LIB", "output/finetune/rlib")
.libPaths(c(lib, .libPaths()))
suppressPackageStartupMessages({
  library(epinowcast, lib.loc = lib)
  library(data.table)
})
pkgload::load_all("packages/nowcastfoundation/r", quiet = TRUE)
options(cmdstanr_warn_inits = FALSE, mc.cores = 1)
enw_set_cache(
  Sys.getenv("ENW_CACHE", "output/finetune/enwcache"), type = "session"
)
grid <- read_grid(args[[1]])
n <- as.integer(args[[2]])
seed <- as.integer(args[[3]])
out_dir <- args[[4]]
weeks <- Sys.getenv("ENW_WEEKS")
source("scripts/enw_fingerprint.R", local = TRUE)
info <- sessionInfo()
stack <- list(
  r_version = R.version.string, blas = info$BLAS, lapack = info$LAPACK,
  enw_fingerprint = enw_fingerprint(lib), host = Sys.info()[["nodename"]]
)
t0 <- Sys.time()
out <- simulate_batch(
  grid, seed, n,
  model = suppressMessages(enw_model(threads = FALSE)),
  draws = as.integer(Sys.getenv("ENW_DRAWS", "1")),
  weeks = if (nzchar(weeks)) as.integer(weeks),
  debug = nzchar(Sys.getenv("ENW_DEBUG"))
)
if (!nrow(out$settings)) {
  stop("no simulation of ", n, " succeeded; first failure: ",
    out$failures$reason[1])
}
out$settings[, names(stack) := stack]
message(sprintf(
  "%d of %d simulations in %.1f s (epinowcast %s, grid %s)",
  nrow(out$settings), n, as.numeric(Sys.time() - t0, units = "secs"),
  grid$epinowcast_sha, grid$name
))
dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)
write_atomic <- function(rows, name, ext) {
  final <- file.path(out_dir, sprintf("%s-%d.%s", name, seed, ext))
  tmp <- file.path(out_dir, paste0(".tmp-", basename(final)))
  fwrite(rows, tmp)
  file.rename(tmp, final)
}
if (nrow(out$failures)) write_atomic(out$failures, "failures", "csv")
write_atomic(out$latent, "latent", "csv.gz")
write_atomic(out$counts, "counts", "csv.gz")
write_atomic(out$settings, "settings", "csv")
