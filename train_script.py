"""
train_script.py

Train a VAE model on processed data. 

Command Line Arguments:
	input_data: path to input data file.
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

# Parse arguments
parser = argparse.ArgumentParser()

parser.add_argument('-i', '--input_data', default="./data/data_processed.h5ad")
parser.add_argument('-o', '--output_data', default="./data/latent_data.h5ad")
parser.add_argument('-m', '--model_outdir', default="./data/model")

args = parser.parse_args()

# Housekeeping
data_in = args.input_data  # "gast_data_full_processed.h5ad"
data_out = args.output_data
model_outdir = args.model_outdir  #"gast_full_model"

# Load processed data
adata = scv.read(data_in)

# Construct the model
model = ltv.models.AnnotVAE(observed=2000, latent_dim=70, zr_dim=6, h_dim=7,
                            encoder_hidden=80,
                            celltypes=34, exp_time=True, time_reg=True,
                            time_reg_weight=0.2, time_reg_decay = 25, 
                            batch_correction=True, 
                            batches=len(adata.obs['sequencing.batch'].unique()))

# Train the model
epochs, val_ae, val_traj = ltv.train_anvi(
	model, adata, 
	epochs=50, 
	batch_size=1000, 
    name=model_outdir, 
    grad_clip=10, 
    learning_rate=0.7e-2)

# Generate and write latent data object
latent_adata = ltv.output_results(model, adata)
latent_adata.write(data_out)
