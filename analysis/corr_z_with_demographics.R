library(tidyverse)
library(corrplot)
NZ <- 256

fit_covariates_tests = readRDS("~/01_repos/CardiacMotionRL/mlruns/2/6a4d73fb59f24d97b37764afdedd4185/artifacts/output/latent_vector_cov_adj_summaries.rds" %>% glue::glue())
pvals <- sapply(fit_covariates_tests, function(x) x[,'Pr(>|t|)'])
pvals <- pvals[2:nrow(pvals),]
pvals[pvals > 0.01] = 1
pvals <- -log10(pvals)
pvals[pvals > 15] <- 15
# pvals <- pvals[, names(sort(-apply(pvals, 2, sum)))]
pvals <- t(pvals)[1:24,]

FILENAME <- "~/01_repos/CardiacMotionRL/mlruns/2/6a4d73fb59f24d97b37764afdedd4185/artifacts/output/figures/corr_demographics_pvals.png" %>% glue::glue()
png(FILENAME, width = 1000, height = 2000)

corrplot(
  pvals,
  method="color",
  is.corr = FALSE, 
  tl.cex = 2, cl.cex = 2, tl.col = "black", tl.srt = 75,
  
  #col = c(rev(COL1("Reds", 100)), COL1("Blues", 100)),
  col = COL1("Blue", 200),
  addgrid.col = 'black',
  # cl.pos = 'b'
)

title(
  main="-log10(p)" %>% glue::glue(), 
  cex.main = 3,   font.main= 4
)

dev.off()