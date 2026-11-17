
ex <- parse("scripts/finetune_simulate.R")
for (e in ex) {
  txt <- paste(deparse(e), collapse = "")
  if (grepl("commandArgs|args\\[\\[|^if \\(sys.nframe", txt)) next
  if (grepl("^lib <-", txt)) { lib <- "output/finetune/rlib"; next }
  eval(e, globalenv())
}
draws <- 1L

set.seed(11)
g <- 0L
repeat { cell <- cell_for(g); if (cell$strata == "places_ages" && cell$strata_link == "linked" && cell$delay_form == "parametric" && cell$delay_family == "lognormal") break; g <- g + 1L }
d <- draw_design(g)
d$weeks <- 52L; d$n_places <- 8L; d$n_ages <- 4L; d$delay_change <- "walk"; d$reporting <- "daily"
v <- draw_values(d)
v$delay_median <- 12; v$delay_sdlog <- 1.2; v$max_delay <- 84L
t0 <- Sys.time()
out <- simulate_outbreak(d, v)
cat("worst", nrow(out$counts), sum(out$counts$count), as.numeric(Sys.time() - t0, units = "secs"), "\n")
