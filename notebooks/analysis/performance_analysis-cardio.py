# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.17.3
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %%
import pandas as pd
import mlflow
import ipywidgets as widgets
from ipywidgets import interact
import os, sys
import glob
import torch
from pprint import pprint
from easydict import EasyDict
import pickle as pkl

import numpy as np
import torch
import yaml

os.environ['HOME'] = "/home/user"
os.environ['CARDIAC_MOTION_REPO'] = os.environ["HOME"] + "/01_repos/CardiacMotion"
os.chdir(os.environ['CARDIAC_MOTION_REPO'])

sys.path.append(os.environ['CARDIAC_MOTION_REPO'])

from utils.image_helpers import generate_gif, merge_gifs_horizontally

# %%
from tqdm.notebook import trange, tqdm

# %%
MLFLOW_TRACKING_URI = f"{os.environ['HOME']}/01_repos/CardiacMotion/mlruns/"
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

# %%
MLFLOW_URI = mlflow.tracking.get_tracking_uri()
# EXPERIMENT_ID = mlflow.get_experiment_by_name(EXPERIMENT_NAME).experiment_id

# PREFIX = f"{MLFLOW_URI}/{EXPERIMENT_ID}"

# %%
# meta_yamls = [f"{PREFIX}/{run}/meta.yaml" for run in os.listdir(PREFIX) if os.path.exists(f"{PREFIX}/{run}/meta.yaml")]
# 
# count = 0
# for meta_yaml_path in meta_yamls:
#     meta_yaml = yaml.safe_load(open(meta_yaml_path))    
#     if meta_yaml['experiment_id'] != EXPERIMENT_ID:
#         meta_yaml['experiment_id'] = EXPERIMENT_ID
#         count += 1
#         yaml.dump(meta_yaml, open(meta_yaml_path, "wt"))
#         
# if count != 0:
#     print(f"{count} runs's experiments were fixed to match the experiment of the parent folder")

# %%
# Choose runs with good performance
df = mlflow.search_runs(experiment_ids=[str(i) for i in range(2,9)])
df = df[(df["metrics.val_rec_ratio_to_time_mean"] < 1) & (df["params.dataset_n_timeframes"] == '10')]

# %% [markdown]
# ___

# %%
# columns
normalized_metrics = df.columns[df.columns.str.startswith("metrics.val_") & df.columns.str.contains("ratio")]

dataset_params = df.columns[df.columns.str.startswith("params.dataset_")].to_list()

params = df.columns[df.columns.str.startswith("params.")].to_list()

arch_params = [
    'params.n_channels_enc', 
    'params.n_channels_dec_c', 
    'params.n_channels_dec_s', 
    'params.latent_dim_c', 
    'params.latent_dim_s',
    'params.z_aggr_function',
    'params.reduction_factors',
]

# To not display columns that have the same value for all rows
# https://stackoverflow.com/questions/57365283/how-to-show-columns-that-have-different-values-in-rows
def diff_cols(df):
    
    my_cols = []
    for col in df.columns:
        if df[col].nunique(dropna=False) > 1:
            my_cols.append(col)
    return df[my_cols].copy()


columns = ["experiment_id", "run_id"] + params + normalized_metrics.tolist()

df_reduced = diff_cols(
    df[columns].reset_index(drop=True)
).sort_values("experiment_id")

df_reduced["partition"] = df_reduced.experiment_id.apply(lambda expid: mlflow.get_experiment(expid).name)
df_reduced = df_reduced.set_index("run_id")
df_reduced

# %% [markdown]
# ### Get run IDs based on metrics 

# %%
good_runs = df_reduced.index

kk = { tuple(df_reduced.loc[run, ["partition", "metrics.val_rec_ratio_to_time_mean"]]): run for run in good_runs }
kk = {(k[0], round(k[1], 3)): v for k, v in kk.items()}

run_w = widgets.Select(options=kk)
run_w

# %% [markdown]
# ### Load weights

# %%
runid = run_w.value
expid = df_reduced.loc[run_w.value].experiment_id

ckpt_dir = f"{os.environ['HOME']}/01_repos/CardiacMotion/{expid}/{runid}/checkpoints"
ckpt_path = f"{ckpt_dir}/{os.listdir(ckpt_dir)[0]}"

# %%
# ckpt_path = f"/root/Rodrigo_repos/CardiacMotion/2/{run_w.value}/checkpoints/epoch=334-step=38524.ckpt"

# model_weights = torch.load(ckpt_path, map_location=torch.device('cpu'))["state_dict"]
model_weights = torch.load(ckpt_path)["state_dict"]
print(f"Loaded weights from checkpoint:\n {ckpt_path}")
# model_weights = EasyDict(model_weights)
model_weights = EasyDict({k.replace("model.", ""): v for k, v in model_weights.items()})

# %% [markdown]
# ### Initialize weights

# %%
from main_autoencoder_cardiac import *
from config.load_config import load_yaml_config

from models.Model3D import Encoder3DMesh, Decoder3DMesh
from models.Model4D import DECODER_C_ARGS, DECODER_S_ARGS, ENCODER_ARGS
from models.Model4D import DecoderStyle, DecoderContent, DecoderTemporalSequence 
from models.Model4D import EncoderTemporalSequence, AutoencoderTemporalSequence
from lightning.ComaLightningModule import CoMA_Lightning
from models.lightning.EncoderLightningModule import TemporalEncoderLightning
from models.TemporalAggregators import TemporalAggregator, FCN_Aggregator

POLYNOMIAL_DEGREE = 10
DOWNSAMPLING = 3

config = load_yaml_config("config_folded_c_and_s.yaml")
config.network_architecture.convolution.parameters.polynomial_degree = [POLYNOMIAL_DEGREE] * 4
config.network_architecture.pooling.parameters.downsampling_factors = [3, 3, 2, 2] # * 4
config.network_architecture.latent_dim_c = 8 
config.network_architecture.latent_dim_s = 8

# %%
from fuzzywuzzy import fuzz, process

partition = df_reduced.loc[runid, ["partition"]].item()
PARTITION = process.extractOne(partition, partitions.keys())[0]

# %%
FACES_FILE = "utils/CardioMesh/data/faces_and_downsampling_mtx_frac_0.1_LV.pkl"

MEAN_ACROSS_CYCLE_FILE = f"utils/CardioMesh/data/cached/mean_shape_time_avg__{PARTITION}.npy"
PROCRUSTES_FILE = f"utils/CardioMesh/data/cached/procrustes_transforms_{PARTITION}.pkl"    
SUBSETTING_MATRIX_FILE = f"/home/user/01_repos/CardioMesh/data/cached/subsetting_matrix_{PARTITION}.pkl" 

subsetting_matrix = pkl.load(open(SUBSETTING_MATRIX_FILE, "rb"))

ID = "1000511"
fhm_mesh = Cardiac3DMesh(
   filename=f"/mnt/data/workshop/workshop-user1/datasets/meshes/Results_Yan/{ID}/models/FHM_res_0.1_time001.npy",
   faces_filename="/home/user/01_repos/CardioMesh/data/faces_fhm_10pct_decimation.csv",
   subpart_id_filename="/home/user/01_repos/CardioMesh/data/subpartIDs_FHM_10pct.txt"
)

template = EasyDict({
  "v": np.load(MEAN_ACROSS_CYCLE_FILE),
  "f": fhm_mesh[partitions[PARTITION]].f
})

# %% [markdown]
# #### Cardiac dataset

# %%
NT = 10 # config.dataset.parameters.T
cardiac_dataset = CardiacMeshPopulationDataset(
    root_path="data/cardio/Results", 
    procrustes_transforms=PROCRUSTES_FILE,
    faces=template.f,
    subsetting_matrix=subsetting_matrix,
    template_mesh= template,
    N_subj=1000,
    phases_filter=1+(50/NT)*np.array(range(NT))
)

print(f"Length of dataset: {len(cardiac_dataset)}") 

# %%
# datamodule = get_datamodule(config.dataset, batch_size=config.batch_size)

mesh_dm = CardiacMeshPopulationDM(cardiac_dataset, batch_size=32)
mesh_dm.setup()

x = EasyDict(next(iter(mesh_dm.train_dataloader())))

mesh_template = mesh_dm.dataset.template_mesh
coma_args = get_coma_args(config)
coma_matrices = get_coma_matrices(config, mesh_template, PARTITION)
coma_args.update(coma_matrices)

enc_config = EasyDict({k: v for k, v in coma_args.items() if k in ENCODER_ARGS})
encoder = Encoder3DMesh(**enc_config)

enc_config.latent_dim = config.network_architecture.latent_dim_c + config.network_architecture.latent_dim_s 

h = encoder.forward_conv_stack(x.s_t, preserve_graph_structure=False)

z_aggr = FCN_Aggregator(
    features_in = NT*h.shape[-1],
    features_out= enc_config.latent_dim
)

t_encoder = EncoderTemporalSequence(
    encoder3d = encoder,
    z_aggr_function=z_aggr
)

decoder_config_c = EasyDict({ k:v for k,v in coma_args.items() if k in DECODER_C_ARGS })
decoder_config_s = EasyDict({ k:v for k,v in coma_args.items() if k in DECODER_S_ARGS })    
decoder_content = DecoderContent(decoder_config_c)
decoder_style = DecoderStyle(decoder_config_s, phase_embedding_method="exp_v1")
t_decoder = DecoderTemporalSequence(decoder_content, decoder_style)
    
t_ae = AutoencoderTemporalSequence(
    encoder=t_encoder,
    decoder=t_decoder
)

# %%
t_ae.load_state_dict(model_weights)
t_ae = t_ae.to("cuda:0")

# %%
# model_weights = torch.load("/home/user/01_repos/CardiacMotion/4/8a320f3c1d2b4d799da91be75205ca81/checkpoints/epoch=344-step=252884.ckpt")

# %%
# lit_module = CoMA_Lightning(
#     model=t_ae, 
#     loss_params=config.loss, 
#     optimizer_params=config.optimizer,
#     additional_params=config,
#     mesh_template=mesh_template
# )

# %% [markdown]
# ### Load input meshes

# %%
mesh_dl = torch.utils.data.DataLoader(cardiac_dataset, batch_size=8, num_workers=16)

# %% [markdown]
# ___

# %% [markdown]
# ### Generate animation

# %%
x["s_t"] = x["s_t"].to("cuda:0")

# %%
output = t_ae(x["s_t"])
s_t, s_hat_t = x["s_t"], output[2]

# %%
# import pyvista as pv
#         
# connectivity = np.c_[np.ones(template.f.shape[0]) * 3, template.f].astype(int)
# pv.set_plot_theme("document")
# pv.start_xvfb()
# 
# plotter = pv.Plotter(notebook=False, off_screen=True)
# FILENAME = f"test_{PARTITION}_rec.gif"
# plotter.open_gif(FILENAME)
# 
# mesh4D = t_ae(x["s_t"])[2]
# # mesh4D = x["s_t"]
# mesh4D = mesh4D.detach().cpu().numpy()[0].astype("float32")
# 
# pv_mesh = pv.PolyData(mesh4D[0], connectivity)
# plotter.add_mesh(pv_mesh, show_edges=False, opacity=0.5, color="red") 
# 
# for t, _ in tqdm(enumerate(mesh4D)):
#     # print(t)
#     kk = pv.PolyData(mesh4D[t], connectivity)
#     plotter.camera_position = "xz" # camera_position
#     plotter.update_coordinates(kk.points, render=False)
#     plotter.render()             
#     plotter.write_frame()
# 

# %% [markdown]
# ### Generate gif's

# %%
subj_ids = list(range(64))

# %%
faces = template.f

# %%
f"{os.environ['HOME']}/01_repos/CardiacMotion/mlruns/{expid}/{runid}/artifacts/output/gif"        

# %%
# ODIR = f"{os.environ['HOME']}/01_repos/CardiacMotion/mlruns/{expid}/{runid}/artifacts/output/gif"        
# os.makedirs(ODIR)

for subj_id in subj_ids:

    for camera in ["xz", "xy", "yz"]:    
        for suffix, st in {"original": s_t, "reconstruction": s_hat_t}.items():
            mesh4D = st.detach().cpu().numpy().astype("float32")[subj_id]        
            gifpath = generate_gif(
                mesh4D,
                faces, 
                camera_position=camera,
                filename=f"{ODIR}/id{subj_id}_{suffix}_{camera}.gif", 
            )
        
        merge_gifs_horizontally(
            f"{ODIR}/id{subj_id}_original_{camera}.gif", 
            f"{ODIR}/id{subj_id}_reconstruction_{camera}.gif", 
            f"{ODIR}/id{subj_id}_{camera}.gif"
        )
        
        os.remove(f"{ODIR}/id{subj_id}_original_{camera}.gif")
        os.remove(f"{ODIR}/id{subj_id}_reconstruction_{camera}.gif")

#b64 = base64.b64encode(
#  open(gifpath,'rb').read()
#).decode('ascii')
#b64 = base64.b64encode(
#    open(gifpath,'rb').read()
#).decode('ascii')

# %%
@interact
def show_gif(subj_id=widgets.IntSlider(min=0,max=63)):
    
    # subj_id = cardiac_dataset.ids[subj_id]                
    
    gifpath = generate_gif(
        torch.stack(pp).mean(0), #cardiac_dataset[subj_idx], 
        faces, filename="kk.gif", camera_position="xz"
    )
    
    b64 = base64.b64encode(
        open(gifpath,'rb').read()
    ).decode('ascii')
    
    display(HTML(f'<img src="data:image/gif;base64,{b64}" />'))       


# %% [markdown]
# ___

# %% [markdown]
# ### Save $\textbf{z}$ to file

# %%
torch.cuda.empty_cache()

zs = []

for i, x in tqdm(enumerate(mesh_dl)):
    
    # if (i % 10) == 0:
    # print(i)
        
    if i < (len(zs)-1):
        continue
    
    x['s_t'] = x['s_t'].to("cuda:0")
    z = t_ae.encoder(x['s_t'])
    z = z['mu'].detach().cpu().numpy()
    zs.append(z)
    
    
    # zs.append(z)
    torch.cuda.empty_cache() 

zs_concat = np.concatenate(zs)
z_df = pd.DataFrame(zs_concat, index=cardiac_dataset.ids)
del zs_concat, zs

# colnames before: 0, 1, 2, 3
z_df.columns = [ f"z{str(i).zfill(3)}" for i in range(16) ]
# colnames after: z000, z001, z002, z003

z_df = z_df.reset_index().rename({"index": "ID"}, axis=1)
z_df.head()

MLRUNS_DIR = "/mnt/data/workshop/workshop-user1/output/CardiacMotion/mlruns"
# RUN_ID = "8c1ffa20cacc4b6c88e18159e01867b4"
ZFILE = f"{MLRUNS_DIR}/{expid}/{runid}/artifacts/latent_vector.csv"
z_df.to_csv(ZFILE, index=False)
print(ZFILE)

# %% [markdown]
# $\textbf{z}$ correlation matrix

# %%
import seaborn as sns
from scipy.cluster import hierarchy

z_corr_df = z_df.corr().abs()
dendrogram = hierarchy.linkage(z_corr_df, method='average')
reordered_matrix = z_corr_df.iloc[hierarchy.leaves_list(dendrogram), hierarchy.leaves_list(dendrogram)]

sns.heatmap(
    # z_corr_df, 
    reordered_matrix, 
    cmap='Greys', 
    xticklabels=True, yticklabels=True,    
);
