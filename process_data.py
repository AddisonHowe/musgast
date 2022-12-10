"""
process_data.py

Process downloaded data. 

Command Line Arguments:
	input_data: path to input data file.
	output_data: path to write output data.
	gene_list: list of genes to be kept regardless of cleaning.

Results:
	Trained model parameters saved in directory <model_outdir>. May overwrite.
	Latent data saved to file <output_data> if h5ad format.
"""

import argparse 
import latentvelo as ltv
import numpy as np
import scvelo as scv
import gc

GENE_LIST_DEFAULT = [
    'T',  # Bra
    'Cdx2',
    'Sox1',
    'Sox2',
    'Tbx6',
    'Otx2'
]

# Parse arguments
parser = argparse.ArgumentParser()

parser.add_argument('-i', '--input_data', required=True)
parser.add_argument('-o', '--output_data', required=True)
parser.add_argument('--gene_list', type=str, nargs='+', default=GENE_LIST_DEFAULT)

args = parser.parse_args()

# Housekeeping
input_data = args.input_data
output_data = args.output_data
gene_list = args.gene_list

# Load data
adata = scv.read(input_data)

# Subsample
np.random.seed(1)
adata = adata[np.random.choice(adata.shape[0], size=30000, replace=False)]

# Change index to SYMBOL instead of ENSEMBL for splicing purposes
adata.var.index = adata.var['SYMBOL']

# Remove cells with stage "mixed_gastrulation"
adata = adata[adata.obs['stage'] != 'mixed_gastrulation']

# Remove Extraembryonic cells
adata = adata[~np.isin(adata.obs['celltype'], ['ExE ectoderm','ExE endoderm','ExE mesoderm'])]

# Rename "un/spliced_counts" to "un/spliced"
adata.layers['spliced'] = adata.layers['spliced_counts']
del adata.layers['spliced_counts']
adata.layers['unspliced'] = adata.layers['unspliced_counts']
del adata.layers['unspliced_counts']

# Copy cell types to field "celltype_names"
adata.obs['celltype_names'] = adata.obs['celltype'].copy().values

# set up experimental time
adata.obs['exp_time'] = np.array([float(t[1:]) for t in adata.obs['stage']])
adata.obs['exp_time'] = adata.obs['exp_time']/adata.obs['exp_time'].max()

# Filter low count genes
scv.pp.filter_genes(adata, min_shared_counts=10, retain_genes=gene_list)
gc.collect()

# Apply LatentVelo cleaning
ltv.utils.anvi_clean_recipe(adata, batch_key='sequencing.batch', celltype_key='celltype', n_top_genes=2000)
gc.collect()

# Convert umap data from dataframe to numpy array and rename
adata.obsm['X_umap'] = adata.obsm['umap'].to_numpy()

# Write output data
adata.write_h5ad(output_data)
