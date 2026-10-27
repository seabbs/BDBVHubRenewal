# Gate: validate a submission and its model metadata with hubValidations.
#
# Usage: Rscript scripts/validate-hub.R <hub_path> <submission.csv> <metadata.yml>
#
# Runs scripts/validate_with_hubvalidations.R, which copies the hub's
# `hub-config/` into a temporary hub, places each file there and runs
# `validate_submission` and `validate_model_metadata`. The hub checkout is
# never modified. The submission must sit in a directory named after its
# model id, as it will in the hub (`model-output/<model_id>/`); the
# metadata file's directory is searched for `<model_id>.yml`.
#
# The submission window check is skipped unless CHECK_WINDOW=true, so
# files can be checked outside the window. Set VALIDATION_JSON to write
# per-check results as JSON. Exits 1 if any check fails.
args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 3) {
  stop("Usage: validate-hub.R <hub_path> <submission.csv> <metadata.yml>")
}
script_dir <- dirname(normalizePath(sub(
  "^--file=", "", grep("^--file=", commandArgs(FALSE), value = TRUE)[1]
)))
validator <- file.path(script_dir, "validate_with_hubvalidations.R")
metadata <- normalizePath(args[[3]])
opts <- c(
  validator,
  "--hub", normalizePath(args[[1]]),
  "--metadata-dir", dirname(metadata)
)
if (!identical(Sys.getenv("CHECK_WINDOW"), "true")) {
  opts <- c(opts, "--skip-window")
}
json <- Sys.getenv("VALIDATION_JSON")
if (nzchar(json)) {
  opts <- c(opts, "--json", json)
}
cat("hubValidations", format(packageVersion("hubValidations")), "\n")
status <- system2(
  file.path(R.home("bin"), "Rscript"),
  c(shQuote(opts), shQuote(normalizePath(args[[2]])), shQuote(metadata))
)
quit(status = status)
