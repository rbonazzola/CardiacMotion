# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.17.3
#   kernelspec:
#     display_name: cardiac_motion
#     language: python
#     name: python3
# ---

# %%
import os, sys
HOME = os.environ["HOME"]
CARDIAC_GWAS_REPO = f"{HOME}/01_repos/CardiacGWAS"
CARDIAC_COMA_REPO = f"{HOME}/01_repos/CardiacCOMA"
CARDIAC_MOTION_REPO = f"{HOME}/01_repos/CardiacMotion"
MLRUNS_DIR = f"{CARDIAC_MOTION_REPO}/mlruns"
os.chdir(CARDIAC_MOTION_REPO)

from easydict import EasyDict

import re
import glob

import mlflow
from mlflow.tracking import MlflowClient

import torch
import torch.nn.functional as F

import ipywidgets as widgets
from ipywidgets import interact, interactive, fixed, interact_manual
from IPython.display import Image
from IPython import embed

import numpy as np
import pandas as pd
import shlex
from subprocess import check_output

import pickle as pkl
import pytorch_lightning as ptl

from argparse import Namespace
import matplotlib.pyplot as plt

from copy import copy, deepcopy
from pprint import pprint
from tqdm import tqdm

sys.path.insert(0, '..')

from config.cli_args import overwrite_config_items
import pyvista as pv
# from utils.mlflow_helpers import get_model_pretrained_weights


# %%
from utils.CardioMesh.CardiacMesh import Cardiac3DMesh, transform_mesh

from models.Model3D import Autoencoder3DMesh
from lightning.ComaLightningModule import CoMA_Lightning
from config.load_config import load_yaml_config, to_dict

from scipy.linalg import orthogonal_procrustes
from typing import Dict, List
from IPython import embed
import logging

from trimesh import Trimesh

# %%
from tqdm.notebook import trange, tqdm

# %%
MLFLOW_TRACKING_URI = f"{os.environ['HOME']}/01_repos/CardiacMotion/mlruns/"
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

# %%
# Choose runs with good performance
df = mlflow.search_runs(experiment_ids=[str(i) for i in range(3, 9)])
# df = df[(df["metrics.val_rec_ratio_to_time_mean"] < 1) & (df["params.dataset_n_timeframes"] == '25')]
# df = df[(df["params.dataset_n_timeframes"] == '50')]

# %%
lv_df = df.query("experiment_id == '4'").sort_values("metrics.val_recon_loss_s").head(20)
lv_df['artifact_uri'] = lv_df.artifact_uri.apply(lambda x: x.replace("user", "rodrigo"))

# %%
from pathlib import Path


# %%

# %%
@interact
def show_shape(i=widgets.IntSlider(min=0, max=20)):

    ckpts = list(Path(lv_df.artifact_uri.iloc[i]).rglob(f"*.ckpt"))
    ckpt = str(ckpts[0])
    print(ckpt)
    model_weights = torch.load(ckpt, map_location='cpu')['state_dict']
    return pd.DataFrame([ (name, matrix.shape) for name, matrix in model_weights.items() if "fcn" in name or "dec_lin" in name ])# .iloc[30:]


# %%

# %%
lv_df

# %%
torch.load(lv_df.artifact_uri.iloc[0])

# %%
lv_df
