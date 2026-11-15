# Copyright (c) 2026 Sam Abbott. MIT licence, see LICENSE.
# Print a fingerprint of the epinowcast installed in a library: the MD5 of
# its functions' code and its Stan files. It is the same across R
# versions, so two machines running the simulator can be compared.
#
#   Rscript scripts/enw_fingerprint.R output/finetune/rlib
lib <- commandArgs(trailingOnly = TRUE)[[1]]
ns <- loadNamespace("epinowcast", lib.loc = lib)
code <- unlist(lapply(sort(ls(ns, all.names = TRUE)), function(f) {
  c(f, deparse(get(f, ns), control = NULL))
}))
stan <- sort(list.files(
  file.path(lib, "epinowcast", "stan"),
  recursive = TRUE, full.names = TRUE
))
path <- tempfile()
writeLines(c(code, unlist(lapply(stan, readLines))), path)
cat(unname(tools::md5sum(path)), "\n")
