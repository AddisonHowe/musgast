library(MouseGastrulationData)
library(scuttle)
library(zellkonverter)

DEFAULT_OUT_PATH = "./data/gast_data_full.h5ad"

args = commandArgs(trailingOnly=TRUE)

# Argument parsing
if (length(args)==1) {
    out_fpath = args[1]
} else{
  # default output file
  out_fpath = DEFAULT_OUT_PATH
}

sce <- EmbryoAtlasData(get.spliced=TRUE)

# Keep only called cells
singlets <- which(!(colData(sce)$doublet | colData(sce)$stripped))
sce <- sce[,singlets]

# Write sce object to h5ad file
writeH5AD(sce, file=out_fpath)
