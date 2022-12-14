
import argparse
import torch
import scvelo as scv
import latentvelo as ltv

# Parse arguments
parser = argparse.ArgumentParser()

parser.add_argument('-i', '--input_data', required=True)
parser.add_argument('-o', '--output_data', required=True)
parser.add_argument('-p', '--params', required=True)

args = parser.parse_args()

# Housekeeping
data_in = args.input_data
data_out = args.output_data
model_params_fpath = args.params

# Load processed data
adata = scv.read(data_in)

# Construct the model
model = ltv.models.AnnotVAE(
	observed=2000, latent_dim=70, zr_dim=6, h_dim=7,
    encoder_hidden=80,
    celltypes=34, exp_time=True, time_reg=True,
    time_reg_weight=0.2, time_reg_decay = 25, 
    batch_correction=True, 
	batches=len(adata.obs['sequencing.batch'].unique()),
	device='cuda'
)
model.load_state_dict(torch.load(model_params_fpath, 
								 map_location=torch.device('cuda')))

# Generate and write latent data object
latent_adata = ltv.output_results(model, adata)
latent_adata.write(data_out)
