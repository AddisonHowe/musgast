"""
analysis_script.py

Analyze processed data.

Command Line Arguments:
	input_data: path to input data file containing processed gene counts.
	input_latent_data: path to input data file containing latent space info.
	outdir: path to output directory.
	params: path to directory in which model parameters are saved.

Results:
	Trained model parameters saved in directory <model_outdir>. May overwrite.
	Latent data saved to file <output_data> if h5ad format.
"""

import argparse
import anndata
import latentvelo as ltv
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors
import scvelo as scv
import os
from helpers import COLOR_MAP, transition_scores
import gc
import torch
from umap import UMAP
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE, Isomap
from sklearn.inspection import DecisionBoundaryDisplay


# Parse arguments
parser = argparse.ArgumentParser()

parser.add_argument('-i', '--input_data', required=True)
parser.add_argument('-l', '--input_latent_data', required=True)
parser.add_argument('-o', '--outdir', required=True)
parser.add_argument('-p', '--params', required=True)

args = parser.parse_args()

# Housekeeping
adata = anndata.read_h5ad(args.input_data)
latent_adata = anndata.read_h5ad(args.input_latent_data)
outdir = args.outdir
model_params_fpath = args.params
os.makedirs(outdir, exist_ok=True)

scv.settings.autosave = False
scv.settings.autoshow = False
scv.settings.figdir = outdir
scv.settings.file_format_figs = "png"

# Compute velocity graph
scv.tl.velocity_graph(latent_adata, vkey='spliced_velocity')

# Add new colors to dataframes
latent_adata.uns['celltype_names_colors'] = COLOR_MAP
adata.uns['celltype_names_colors'] = COLOR_MAP

print(adata[(adata.obs['celltype_names'] == 'Endothelium')].obs.index)
print(adata[(adata.obs['celltype_names'] == 'Erythroid1')].obs.index)
print(adata[(adata.obs['celltype_names'] == 'Gut')].obs.index)
print(adata[(adata.obs['celltype_names'] == 'Allantois')].obs.index)
print(adata[(adata.obs['celltype_names'] == 'Mesenchyme')].obs.index)
print(adata[(adata.obs['celltype_names'] == 'Forebrain/Midbrain/Hindbrain')].obs.index)

# print(adata[['cell_70494', 'cell_130339', 'cell_19330', 'cell_93806', 'cell_116037',
#        'cell_96434']])

# Compute velocity embedding
scv.pl.velocity_embedding_stream(
	latent_adata, 
	basis='umap',vkey='spliced_velocity',
	color=['latent_time', 'stage', 'celltype_names'], color_map='coolwarm',
	title=['LatentVelo Latent time', 'stage'], 
	cutoff_perc=0, alpha=1, legend_fontsize=18, legend_fontoutline=3, 
	fontsize=18, figsize=(6,5), size=75, legend_loc='right', 
	save='velocity_embedding_stream.png')

scv.pl.velocity_embedding_stream(
	latent_adata, 
	basis='umap',vkey='spliced_velocity',
    color=['celltype_names'], color_map='coolwarm',
    title=['celltypes'], 
	cutoff_perc=0, alpha=1, 
	legend_fontsize=18, legend_fontoutline=3, fontsize=18,
    figsize=(6,5), size=75, legend_loc='right', 
	save='Mouse_gastrulation_velocity.png')

scv.pl.velocity_embedding_stream(
	latent_adata, 
	vkey='spliced_velocity', 
    color=['latent_time','stage'], color_map='coolwarm', 
	title=['Latent time', 'Time point'],
	size=50, legend_loc='right',
    legend_fontsize=18, legend_fontoutline=3, fontsize=18,
    save='gastrulation_velocity_times.png')

# Add latent values to dataframe
for i in range(latent_adata.obsm['zr'].shape[1]):
    latent_adata.obs['zr'+str(i+1)] = latent_adata.obsm['zr'][:,i].copy()
    
scv.pl.umap(
	latent_adata, 
	color=['zr1','zr2','zr3','zr4', 'zr5', 'zr6'], 
	size=50, legend_fontsize=18, legend_fontoutline=3, fontsize=18,
	save='gastrulation_zr.png')

remap_cat_dict = {
	'Primitive Streak'				 	: 'Prim. Streak',
    'Anterior Primitive Streak' 	 	: 'Ant. Prim. Streak',
    'Nascent mesoderm'				 	: 'Nasc. mesoderm', 
    'Haematoendothelial progenitors' 	: 'Haem. endothel. prog.', 
    'Blood progenitors 1' 			 	: 'Blood prog. 1', 
    'Blood progenitors 2' 			 	: 'Blood prog. 2'
}

latent_adata.obs['new_celltypes'] = latent_adata.obs['celltype_names'].replace(
	remap_cat_dict).astype('category')


# Plot Transition Scores
mesoderm_edges = [
	('Epiblast', 'Prim. Streak'),
    ('Prim. Streak', 'Nasc. mesoderm')]

latent_adata.obsm['X_latent'] = latent_adata.X.copy()
latent_adata.obsm['velocity_latent'] = latent_adata.layers['spliced_velocity'].copy()

mesoderm_scores = ltv.ev.cross_boundary_correctness(
	latent_adata, 'new_celltypes', 'velocity_latent', mesoderm_edges,
    x_emb='X_latent', majority_vote=True, return_raw=True)

blood_edges = [
	('Haem. endothel. prog.', 'Endothelium'),
	('Haem. endothel. prog.', 'Blood prog. 1'),
	('Blood prog. 1', 'Blood prog. 2'),
	('Blood prog. 2', 'Erythroid1'),
	('Erythroid1', 'Erythroid2'),
	('Erythroid2', 'Erythroid3')]

latent_adata.obsm['X_latent'] = latent_adata.X.copy()
latent_adata.obsm['velocity_latent'] = latent_adata.layers['spliced_velocity'].copy()

blood_scores = ltv.ev.cross_boundary_correctness(
	latent_adata, 'new_celltypes', 'velocity_latent', blood_edges,
    x_emb='X_latent', majority_vote=True, return_raw=True)

gut_edges = [
	('Epiblast', 'Prim. Streak'), 
    ('Prim. Streak', 'Ant. Prim. Streak'),
    ('Ant. Prim. Streak', 'Def. endoderm'),
    ('Def. endoderm', 'Gut'),
	('Visceral endoderm', 'Gut')]

latent_adata.obsm['X_latent'] = latent_adata.X.copy()
latent_adata.obsm['velocity_latent'] = latent_adata.layers['spliced_velocity'].copy()

gut_scores = ltv.ev.cross_boundary_correctness(
	latent_adata, 'new_celltypes', 'velocity_latent', gut_edges,
    x_emb='X_latent', majority_vote=True, return_raw=True)

fig, ax=plt.subplots(1, 3, figsize=(12, 2))
transition_scores(mesoderm_scores, raw=True, ax=ax[0])
ax[0].set(xlim=(0.,1))
ax[0].tick_params(axis='y')

transition_scores(blood_scores, raw=True, ax=ax[1])
ax[1].set(xlim=(0.,1))
ax[1].tick_params(axis='y')

transition_scores(gut_scores, raw=True, ax=ax[2])
ax[2].set(xlim=(0.,1))
ax[2].tick_params(axis='y')

fig.tight_layout()
fig.subplots_adjust(wspace=2.7)
fig.savefig(f'{outdir}/gastrulation_transition_scores.png')


# free up some space
del adata.layers['Mu']
del adata.layers['Ms']
del adata.layers['spliced_counts']
del adata.layers['unspliced_counts']
del adata.layers['mask_spliced']
del adata.layers['mask_unspliced']
gc.collect()

# Load model
model = ltv.models.AnnotVAE(
	observed=2000, latent_dim=70, zr_dim=6, h_dim=7,
    encoder_hidden=80,
    celltypes=34, exp_time=True, time_reg=True,
    time_reg_weight=0.2, time_reg_decay = 25, 
    batch_correction=True, 
	batches=len(adata.obs['sequencing.batch'].unique()),
	device='cpu'
)
model.load_state_dict(torch.load(model_params_fpath, 
								 map_location=torch.device('cpu')))

# Compute cell trajectories
z_traj, times = ltv.cell_trajectories(model, adata)

# Plot
pca = PCA(n_components=6).fit(latent_adata.X)
umap = UMAP(n_components=2, min_dist=0.5, n_neighbors=100).fit(
	pca.transform(latent_adata.X))

transformed = umap.transform(pca.transform(latent_adata.X))
latent_adata.obsm['X_umap_latent'] = transformed
scv.pp.neighbors(latent_adata, use_rep='X', n_neighbors=30)
scv.tl.velocity_graph(latent_adata, vkey='spliced_velocity')

scv.pl.velocity_embedding_stream(
	latent_adata, 
	basis='umap_latent', vkey='spliced_velocity', 
    color=['celltype_names'], 
	alpha=1, size=50, frameon='artist', legend_loc='right', cutoff_perc=0,
    xlabel = 'Latent UMAP1', ylabel='Latent UMAP2', 
	title='', 
	legend_fontsize=16, fontsize=15.5,
    save='mouse_gastrulation_latent_umap.png')


# Trajectories
cells = [
	'cell_71068', #'cell_65328',
	'cell_123777', #'cell_33062',
	'cell_19760', #'cell_23583',
	'cell_95759', #'cell_117901',
	'cell_118985', #'cell_138449',
	'cell_97853', #'cell_70231'
]
cells = [adata.obs.index.get_loc(c) for c in cells]

color = '#00008B'
cmap = matplotlib.colors.ListedColormap([
	'#1f78b4', '#b2df8a', '#6a3d9a', '#cab2d6'])


fig,ax=plt.subplots(1, 6, figsize=(30,4))
ax=ax.flatten()

latent_adata.obsm['X_umap'] = latent_adata.obsm['X_umap_latent']

z_traj_pca = umap.transform(pca.transform(
	z_traj[cells[0],times[cells[0],:,0] <= \
		latent_adata.obs['latent_time'][cells[0]],:70].detach().cpu().numpy()))

ax[0].plot(z_traj_pca[:,0], z_traj_pca[:,1], 
	    	color=color, linewidth=2)
ax[0].plot([z_traj_pca[-1,0]], [z_traj_pca[-1,1]], 
			color=color, linewidth=2, marker='o', markersize=9)
ax[0].plot([z_traj_pca[0,0]], [z_traj_pca[0,1]], 
			color='k', linewidth=2, marker='s', markersize=6)

scv.pl.umap(
	latent_adata, ax=ax[0], 
	color='celltype_names', frameon='artist',
    xlabel='Latent UMAP1', ylabel='Latent UMAP2', show=False, 
	fontsize=11, title='Endothelial', legend_loc='none', size=30)


z_traj_pca = umap.transform(pca.transform(
	z_traj[cells[1],times[cells[1],:,0] <= \
		latent_adata.obs['latent_time'][cells[1]],:70].detach().cpu().numpy()))

ax[1].plot(z_traj_pca[:,0], z_traj_pca[:,1], 
			color=color, linewidth=2)
ax[1].plot([z_traj_pca[-1,0]], [z_traj_pca[-1,1]], 
			color=color, linewidth=2, marker='o', markersize=9)
ax[1].plot([z_traj_pca[0,0]], [z_traj_pca[0,1]], 
			color='k', linewidth=2, marker='s', markersize=6)

scv.pl.umap(
	latent_adata, ax=ax[1], 
	color='celltype_names',
    xlabel='Latent UMAP1', ylabel='Latent UMAP2', show=False, 
	fontsize=11, title='Erythroid', legend_loc='none', size=30)

z_traj_pca = umap.transform(pca.transform(
	z_traj[cells[2],times[cells[2],:,0] <= \
		latent_adata.obs['latent_time'][cells[2]],:70].detach().cpu().numpy()))

ax[2].plot(z_traj_pca[:,0], z_traj_pca[:,1], 
			color=color, linewidth=2)
ax[2].plot([z_traj_pca[-1,0]], [z_traj_pca[-1,1]], 
			color=color, linewidth=2, marker='o', markersize=9)
ax[2].plot([z_traj_pca[0,0]], [z_traj_pca[0,1]], 
			color='k', linewidth=2, marker='s', markersize=6)

scv.pl.umap(
	latent_adata, ax=ax[2], 
	color='celltype_names',
    xlabel='Latent UMAP1', ylabel='Latent UMAP2', 
	fontsize=11, show=False, title='Gut', legend_loc='none', size=30)

z_traj_pca = umap.transform(pca.transform(
	z_traj[cells[3],times[cells[3],:,0] <= \
		latent_adata.obs['latent_time'][cells[3]],:70].detach().cpu().numpy()))

ax[3].plot(z_traj_pca[:,0], z_traj_pca[:,1], 
			color=color, linewidth=2)
ax[3].plot([z_traj_pca[-1,0]], [z_traj_pca[-1,1]], 
			color=color, linewidth=2, marker='o', markersize=9)
ax[3].plot([z_traj_pca[0,0]], [z_traj_pca[0,1]], 
			color='k', linewidth=2, marker='s', markersize=6)

scv.pl.umap(
	latent_adata, ax=ax[3], 
	color='celltype_names',
    xlabel='Latent UMAP1', ylabel='Latent UMAP2', show=False, 
	fontsize=11, title='Allantois', legend_loc='none', size=30)

z_traj_pca = umap.transform(pca.transform(
	z_traj[cells[4],times[cells[4],:,0] <= \
		latent_adata.obs['latent_time'][cells[4]],:70].detach().cpu().numpy()))

ax[4].plot(z_traj_pca[:,0], z_traj_pca[:,1], 
			color=color, linewidth=2)
ax[4].plot([z_traj_pca[-1,0]], [z_traj_pca[-1,1]], 
			color=color, linewidth=2, marker='o', markersize=9)
ax[4].plot([z_traj_pca[0,0]], [z_traj_pca[0,1]], 
			color='k', linewidth=2, marker='s', markersize=6)

scv.pl.umap(
	latent_adata, ax=ax[4], 
	color='celltype_names',
    xlabel='Latent UMAP1', ylabel='Latent UMAP2', show=False, 
	fontsize=11, title='Mesenchyme', legend_loc='none', size=30)


z_traj_pca = umap.transform(pca.transform(
	z_traj[cells[5],times[cells[5],:,0] <= \
		latent_adata.obs['latent_time'][cells[5]],:70].detach().cpu().numpy()))
ax[5].plot(z_traj_pca[:,0], z_traj_pca[:,1],
			color=color, linewidth=2)
ax[5].plot([z_traj_pca[-1,0]], [z_traj_pca[-1,1]],
			color=color, linewidth=2, marker='o', markersize=9)
ax[5].plot([z_traj_pca[0,0]], [z_traj_pca[0,1]],
			color='k', linewidth=2, marker='s', markersize=6)

scv.pl.umap(
	latent_adata, ax=ax[5], 
	color='celltype_names',
    xlabel='Latent UMAP1', ylabel='Latent UMAP2', show=False, 
	fontsize=11, title='Forebrain/Midbrain/Hindbrain', 
    legend_loc='right', legend_fontsize=11, size=30)

plt.tight_layout()
plt.savefig(f'{outdir}/Mouse_gastrulation_trajectories.png', dpi=500)

print("FINISHED")
