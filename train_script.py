"""
train_script.py

Train a VAE model on processed data. 

Command Line Arguments:
	input_data: path to input data file.
	output_data: path to write output data.
	model_outdir: path to directory in which to save model parameters.
    (optional) epochs: number of epochs to train.

Results:
	Trained model parameters saved in directory <model_outdir>. May overwrite.
	Latent data saved to file <output_data> in h5ad format.
"""

import os
import argparse
import scvelo as scv
import latentvelo as ltv

# Parse arguments
parser = argparse.ArgumentParser()

parser.add_argument('-i', '--input_data', required=True)
parser.add_argument('-o', '--output_data', required=True)
parser.add_argument('-m', '--model_outdir', required=True)
parser.add_argument('-e', '--epochs', type=int, default=50)

args = parser.parse_args()

# Housekeeping
data_in = args.input_data  # "gast_data_full_processed.h5ad"
data_out = args.output_data
model_outdir = args.model_outdir  #"gast_full_model"
epochs = args.epochs

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
	epochs=epochs, 
	batch_size=1000, 
    name=model_outdir, 
    grad_clip=10, 
    learning_rate=0.7e-2)

# Generate and write latent data object
latent_adata = ltv.output_results(model, adata)
latent_adata.write(data_out)
