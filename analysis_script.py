"""
analysis_script.py

Analyze processed data.

Command Line Arguments:
	input_data: path to input data file containing processed gene counts.
	input_latent_data: path to input data file containing latent space info.
	output_data: path to write output data.
	model_outdir: path to directory in which to save model parameters.

Results:
	Trained model parameters saved in directory <model_outdir>. May overwrite.
	Latent data saved to file <output_data> if h5ad format.
"""

import latentvelo as ltv
import numpy as np
import matplotlib.pyplot as plt
import scvelo as scv
import scanpy
import gc

# Parse arguments
parser = argparse.ArgumentParser()

parser.add_argument('-i', '--input_data', required=True)
parser.add_argument('-l', '--input_latent_data', required=True)
parser.add_argument('-o', '--output_data', required=True)
parser.add_argument('-m', '--model_outdir', required=True)

args = parser.parse_args()

# Housekeeping
latent_adata = args.input_data

COLOR_MAP_NEW = {
	"Allantois" : "#532C8A",
	"Anterior Primitive Streak" : "#c19f70",
	"Blood progenitors 1" : "#f9decf",
	"Blood progenitors 2" : "#c9a997",
	"Cardiomyocytes" : "#B51D8D",
	"Caudal epiblast" : "#9e6762",
	"Caudal Mesoderm" : "#3F84AA",
	"Def. endoderm" : "#F397C0",
	"Nascent mesoderm" : "#C594BF",
	"Mixed mesoderm" : "#DFCDE4",#
	"Endothelium" : "#eda450",
	"Epiblast" : "#635547",
	"Erythroid1" : "#C72228",
	"Erythroid2" : "#EF4E22",
	"Erythroid3" : "#f77b59",
	"ExE ectoderm" : "#989898",
	"ExE endoderm" : "#7F6874",
	"ExE mesoderm" : "#8870ad",
	"Rostral neurectoderm" : "#65A83E",
	"Forebrain/Midbrain/Hindbrain" : "#647a4f",
	"Gut" : "#EF5A9D",
	"Haematoendothelial progenitors" : "#FBBE92",
	"Caudal neurectoderm": "#354E23",
	"Intermediate mesoderm" : "#139992",
	"Neural crest": "#C3C388",
	"NMP" : "#8EC792",
	"Notochord" : "#0F4A9C",
	"Paraxial mesoderm" : "#8DB5CE",
	"Parietal endoderm" : "#1A1A1A",
	"PGC" : "#FACB12",
	"Pharyngeal mesoderm" : "#C9EBFB",
	"Primitive Streak" : "#DABE99",
	"Mesenchyme" : "#ed8f84",
	"Somitic mesoderm" : "#005579",
	"Spinal cord" : "#CDE088",
	"Surface ectoderm" : "#BBDCA8",
	"Visceral endoderm" : "#F6BFCB",
	"Mes1": "#c4a6b2",
	"Mes2":"#ca728c",
	"Cardiomyocytes" : "#B51D8D",
}


# Compute velocity graph
scv.tl.velocity_graph(latent_adata, vkey='spliced_velocity')

# Add new colors to dataframes
latent_adata.uns['celltype_names_colors'] = COLOR_MAP_NEW
adata.uns['celltype_names_colors'] = COLOR_MAP_NEW

# Compute velocity embedding
scv.pl.velocity_embedding_stream(
	latent_adata, 
	basis='umap',vkey='spliced_velocity',
	color=['latent_time', 'stage', 'celltype_names'], color_map='coolwarm',
	title=['LatentVelo Latent time', 'stage'], 
	cutoff_perc=0, alpha=1, legend_fontsize=18, legend_fontoutline=3, 
	fontsize=18, figsize=(6,5), size=75, legend_loc='right')

# Add latent values to dataframe
for i in range(latent_adata.obsm['zr'].shape[1]):
    latent_adata.obs['zr'+str(i+1)] = latent_adata.obsm['zr'][:,i].copy()
    
scv.pl.umap(latent_adata, color=['zr1','zr2','zr3','zr4', 'zr5', 'zr6'], size=50,
           save='{0}/gastrulation_zr.png'.format(datdir), 
            legend_fontsize=18, legend_fontoutline=3, fontsize=18)





