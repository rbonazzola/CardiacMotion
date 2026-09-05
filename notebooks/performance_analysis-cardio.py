#!/usr/bin/env python
# coding: utf-8

import os
import sys
import glob
import pickle as pkl
from pprint import pprint

import numpy as np
import pandas as pd
import torch
import yaml
import mlflow
import ipywidgets as widgets
from ipywidgets import interact
from easydict import EasyDict
from tqdm.notebook import trange, tqdm

# Set environment variables
os.environ['HOME'] = "/home/user"
os.environ['CARDIAC_MOTION_REPO'] = os.path.join(os.environ["HOME"], "01_repos/CardiacMotion")
os.chdir(os.environ['CARDIAC_MOTION_REPO'])
sys.path.append(os.environ['CARDIAC_MOTION_REPO'])

from utils.image_helpers import generate_gif, merge_gifs_horizontally

# MLflow setup
MLFLOW_TRACKING_URI = os.path.join(os.environ['HOME'], "01_repos/CardiacMotion/mlruns/")
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
MLFLOW_URI = mlflow.tracking.get_tracking_uri()

# Load and filter runs
df = mlflow.search_runs(experiment_ids=[str(i) for i in range(2, 9)])
df = df[(df["metrics.val_rec_ratio_to_time_mean"] < 1) & (df["params.dataset_n_timeframes"] == '10')]

# Define columns of interest
normalized_metrics = df.columns[df.columns.str.startswith("metrics.val_") & df.columns.str.contains("ratio")]
dataset_params = df.columns[df.columns.str.startswith("params.dataset_")].to_list()
params = df.columns[df.columns.str.startswith("params.")].to_list()
arch_params = [
    'params.n_channels_enc', 'params.n_channels_dec_c', 'params.n_channels_dec_s',
    'params.latent_dim_c', 'params.latent_dim_s', 'params.z_aggr_function', 'params.reduction_factors'
]

# Utility function to filter columns with varying values
def diff_cols(df):
    return df[[col for col in df.columns if df[col].nunique(dropna=False) > 1]].copy()

columns = ["experiment_id", "run_id"] + params + normalized_metrics.tolist()
df_reduced = diff_cols(df[columns].reset_index(drop=True)).sort_values("experiment_id")
df_reduced["partition"] = df_reduced.experiment_id.apply(lambda expid: mlflow.get_experiment(expid).name)
df_reduced = df_reduced.set_index("run_id")

# Widget for selecting runs
good_runs = df_reduced.index
run_options = {tuple(df_reduced.loc[run, ["partition", "metrics.val_rec_ratio_to_time_mean"]]): run for run in good_runs}
run_options = {(k[0], round(k[1], 3)): v for k, v in run_options.items()}
run_w = widgets.Select(options=run_options)

# Load model weights
runid = run_w.value
expid = df_reduced.loc[runid].experiment_id
ckpt_dir = os.path.join(os.environ['HOME'], "01_repos/CardiacMotion", str(expid), runid, "checkpoints")
ckpt_path = os.path.join(ckpt_dir, os.listdir(ckpt_dir)[0])
model_weights = torch.load(ckpt_path)["state_dict"]
model_weights = EasyDict({k.replace("model.", ""): v for k, v in model_weights.items()})

# Model initialization
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
config.network_architecture.pooling.parameters.downsampling_factors = [3, 3, 2, 2]
config.network_architecture.latent_dim_c = 8
config.network_architecture.latent_dim_s = 8

# Load dataset and model
from fuzzywuzzy import fuzz, process
partition = df_reduced.loc[runid, ["partition"]].item()
PARTITION = process.extractOne(partition, partitions.keys())[0]

FACES_FILE = "utils/CardioMesh/data/faces_and_downsampling_mtx_frac_0.1_LV.pkl"
MEAN_ACROSS_CYCLE_FILE = f"utils/CardioMesh/data/cached/mean_shape_time_avg__{PARTITION}.npy"
PROCRUSTES_FILE = f"utils/CardioMesh/data/cached/procrustes_transforms_{PARTITION}.pkl"
SUBSETTING_MATRIX_FILE = os.path.join(os.environ['HOME'], "01_repos/CardioMesh/data/cached/subsetting_matrix_{PARTITION}.pkl")

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

NT = 10
cardiac_dataset = CardiacMeshPopulationDataset(
    root_path="data/cardio/Results",
    procrustes_transforms=PROCRUSTES_FILE,
    faces=template.f,
    subsetting_matrix=subsetting_matrix,
    template_mesh=template,
    N_subj=1000,
    phases_filter=1 + (50 / NT) * np.array(range(NT))
)

print(f"Length of dataset: {len(cardiac_dataset)}")

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
z_aggr = FCN_Aggregator(features_in=NT * h.shape[-1], features_out=enc_config.latent_dim)
t_encoder = EncoderTemporalSequence(encoder3d=encoder, z_aggr_function=z_aggr)

decoder_config_c = EasyDict({k: v for k, v in coma_args.items() if k in DECODER_C_ARGS})
decoder_config_s = EasyDict({k: v for k, v in coma_args.items() if k in DECODER_S_ARGS})
decoder_content = DecoderContent(decoder_config_c)
decoder_style = DecoderStyle(decoder_config_s, phase_embedding_method="exp_v1")
t_decoder = DecoderTemporalSequence(decoder_content, decoder_style)
t_ae = AutoencoderTemporalSequence(encoder=t_encoder, decoder=t_decoder)

t_ae.load_state_dict(model_weights)
t_ae = t_ae.to("cuda:0")

# Generate animations and GIFs
x["s_t"] = x["s_t"].to("cuda:0")
output = t_ae(x["s_t"])
s_t, s_hat_t = x["s_t"], output[2]

subj_ids = list(range(64))
faces = template.f

ODIR = os.path.join(os.environ['HOME'], "01_repos/CardiacMotion/mlruns", str(expid), runid, "artifacts/output/gif")
os.makedirs(ODIR, exist_ok=True)

for subj_id in subj_ids:
    for camera in ["xz", "xy", "yz"]:
        for suffix, st in {"original": s_t, "reconstruction": s_hat_t}.items():
            mesh4D = st.detach().cpu().numpy().astype("float32")[subj_id]
            gifpath = generate_gif(
                mesh4D,
                faces,
                camera_position=camera,
                filename=os.path.join(ODIR, f"id{subj_id}_{suffix}_{camera}.gif"),
            )

        merge_gifs_horizontally(
            os.path.join(ODIR, f"id{subj_id}_original_{camera}.gif"),
            os.path.join(ODIR, f"id{subj_id}_reconstruction_{camera}.gif"),
            os.path.join(ODIR, f"id{subj_id}_{camera}.gif")
        )

        os.remove(os.path.join(ODIR, f"id{subj_id}_original_{camera}.gif"))
        os.remove(os.path.join(ODIR, f"id{subj_id}_reconstruction_{camera}.gif"))

# Save latent vectors
torch.cuda.empty_cache()
zs = []

for i, x in tqdm(enumerate(mesh_dl)):
    x['s_t'] = x['s_t'].to("cuda:0")
    z = t_ae.encoder(x['s_t'])
    z = z['mu'].detach().cpu().numpy()
    zs.append(z)
    torch.cuda.empty_cache()

zs_concat = np.concatenate(zs)
z_df = pd.DataFrame(zs_concat, index=cardiac_dataset.ids)
z_df.columns = [f"z{str(i).zfill(3)}" for i in range(16)]
z_df = z_df.reset_index().rename({"index": "ID"}, axis=1)

MLRUNS_DIR = "/mnt/data/workshop/workshop-user1/output/CardiacMotion/mlruns"
ZFILE = os.path.join(MLRUNS_DIR, str(expid), runid, "artifacts/latent_vector.csv")
z_df.to_csv(ZFILE, index=False)
print(ZFILE)

# Correlation matrix of latent vectors
import seaborn as sns
from scipy.cluster import hierarchy

z_corr_df = z_df.corr().abs()
dendrogram = hierarchy.linkage(z_corr_df, method='average')
reordered_matrix = z_corr_df.iloc[hierarchy.leaves_list(dendrogram), hierarchy.leaves_list(dendrogram)]
sns.heatmap(reordered_matrix, cmap='Greys', xticklabels=True, yticklabels=True)
