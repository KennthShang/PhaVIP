library(ggplot2)
library(reshape2)




args <- commandArgs(trailingOnly = TRUE)

pth <- args[1]
infile <- paste(pth,".csv", sep="")
outfile <- paste(pth,".txt", sep="")

mode <- args[2]
res  <- args[3]

wolf = read.csv(infile)
cat("figure,label,accession\n", file = outfile)
for (i in c(1:nrow(wolf))){
  seq <- as.list(strsplit(wolf$sequence[i], ""))
  flag = try(seq.cgr <- cgr(seq[[1]], seq.base =mode, res = 64), silent=TRUE)
  if ('try-error' %in% class(flag)){
    next
  }
  seq_vec <- vectorize(seq.cgr)
  cat(seq_vec,       file = outfile, append=TRUE)
  cat(",",           file = outfile, append=TRUE)
  cat(wolf$label[i], file = outfile, append=TRUE)
  cat(",",           file = outfile, append=TRUE)
  cat(wolf$accession[i], file = outfile, append=TRUE)
  cat("\n",          file = outfile, append=TRUE)
}



