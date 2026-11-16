suppressPackageStartupMessages({
  library(PLIER)
  library(jsonlite)
})
args <- commandArgs(trailingOnly = TRUE)
stopifnot(length(args) == 1L)
setwd(normalizePath(args[1]))
out <- "result"
if (!dir.create(out, showWarnings = FALSE)) stop("Use a new fit directory")
input <- "input/differential.csv"
gmt <- "input/ReactomePathways.gmt"
sha <- function(path) strsplit(system2("sha256sum", path, stdout = TRUE), " ")[[1]][1]
manifest <- fromJSON("input/manifest.json")
stopifnot(sha(input) == manifest$hashes[[basename(input)]])
data <- read.csv(input, check.names = FALSE, stringsAsFactors = FALSE, colClasses = c(ID = "character"))
ids <- as.character(data$ID)
stopifnot(length(ids) > 0L, !anyNA(ids), all(nzchar(trimws(ids))), !anyDuplicated(ids))
X <- t(as.matrix(data[, -1]))
storage.mode(X) <- "double"
colnames(X) <- ids
stopifnot(nrow(X) == manifest$differential_features, all(is.finite(X)))
if (min(dim(X)) < 20L) stop("The 20-factor fit requires at least 20 samples and 20 selected proteins")
center <- rowMeans(X)
sdev <- apply(X, 1, sd)
stopifnot(all(sdev >= 1e-8))
Y <- sweep(sweep(X, 1, center, "-"), 1, sdev, "/")
write.csv(data.frame(protein = rownames(X), center = center, scale = sdev),
          file.path(out, "normalization.csv"), row.names = FALSE)

gene <- toupper(trimws(sub("^[^_]*_", "", rownames(X))))
indices <- split(seq_along(gene), gene)
paths <- list()
for (line in readLines(gmt, warn = FALSE)) {
  fields <- strsplit(line, "\t")[[1]]
  if (length(fields) < 3) next
  members <- unique(toupper(trimws(fields[-c(1, 2)])))
  mapped <- intersect(members, names(indices))
  if (length(mapped) >= 5) paths[[fields[1]]] <- mapped
}
prior <- matrix(0, nrow = nrow(X), ncol = length(paths),
                dimnames = list(rownames(X), names(paths)))
for (j in seq_along(paths)) {
  for (g in paths[[j]]) prior[indices[[g]], j] <- 1
}
saveRDS(prior, file.path(out, "prior_matrix.rds"))

source <- deparse(PLIER)
anchor <- "if (BdiffCount > 5)"
stopifnot(sum(grepl(anchor, source, fixed = TRUE)) == 1L)
source <- sub(anchor, "if (FALSE && BdiffCount > 5)", source, fixed = TRUE)
fit <- eval(parse(text = source), envir = environment(PLIER))
set.seed(42)
sv <- svd(Y)
set.seed(42)
result <- fit(Y, prior, svdres = sv, k = 20, frac = 0.7, max.iter = 2000,
              trace = TRUE, scale = FALSE, maxPath = 10, doCrossval = TRUE,
              minGenes = 5, tol = 1e-6, seed = 42, allGenes = TRUE,
              pathwaySelection = "complete")
stopifnot(identical(dim(result$B), c(20L, length(ids))), all(is.finite(result$B)))
factor_ids <- sprintf("V5_PLIER_LV%03d", seq_len(nrow(result$B)))
dictionary <- data.frame(factor = factor_ids, index = seq_along(factor_ids),
                         PLIER_name = rownames(result$B),
                         withPrior = seq_along(factor_ids) %in% result$withPrior)
B <- t(result$B)
colnames(B) <- factor_ids
write.csv(dictionary, file.path(out, "factor_dictionary.csv"), row.names = FALSE)
write.csv(data.frame(ID = ids, B, check.names = FALSE),
          file.path(out, "B_all.csv"), row.names = FALSE)
metadata <- list(patients = ncol(Y), protein_features = nrow(Y), pathways = ncol(prior),
                 K = 20, seed = 42, plier_version = as.character(packageVersion("PLIER")),
                 input_sha256 = sha(input), gmt_sha256 = sha(gmt))
write_json(metadata, file.path(out, "fit.json"), pretty = TRUE, auto_unbox = TRUE)
writeLines(capture.output(sessionInfo()), file.path(out, "sessionInfo.txt"))
