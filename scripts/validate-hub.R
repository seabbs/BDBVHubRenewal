# Validate a submission and its model metadata with hubValidations.
#
# Usage: Rscript scripts/validate-hub.R <hub_path> <submission.csv> <metadata.yml>
#
# Copies both files into a checkout of the hub and runs the checks the
# hub runs on pull requests. The submission window check is skipped
# unless CHECK_WINDOW=true, so files can be checked outside the window.
args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 3) {
  stop("Usage: validate-hub.R <hub_path> <submission.csv> <metadata.yml>")
}
hub_path <- normalizePath(args[[1]])
submission <- args[[2]]
metadata <- args[[3]]
model_id <- sub("\\.ya?ml$", "", basename(metadata))
out_dir <- file.path(hub_path, "model-output", model_id)
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)
file.copy(submission, out_dir, overwrite = TRUE)
file.copy(metadata, file.path(hub_path, "model-metadata"), overwrite = TRUE)

meta <- hubValidations::validate_model_metadata(
  hub_path = hub_path,
  file_path = basename(metadata)
)
print(meta)
check_window <- identical(Sys.getenv("CHECK_WINDOW"), "true")
v <- hubValidations::validate_submission(
  hub_path = hub_path,
  file_path = file.path(model_id, basename(submission)),
  skip_submit_window_check = !check_window
)
print(v)
hubValidations::check_for_errors(meta)
hubValidations::check_for_errors(v, verbose = TRUE)
