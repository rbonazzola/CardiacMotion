# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.17.3
#   kernelspec:
#     display_name: cardio
#     language: python
#     name: cardio
# ---

# %% [markdown]
# # Development

# %%
import sys, os
os.chdir("..")

import utils.VTKHelpers
sys.path.append("utils/VTKHelpers/")

from config.load_config import load_yaml_config
from CardiacMesh import Cardiac3DMesh, Cardiac4DMesh, CardiacMeshPopulation
from models import layers

import pickle as pkl
import yaml
from pprint import pprint
from argparse import Namespace
import logging

import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader, random_split
import pytorch_lightning as pl

import ipywidgets as widgets
from IPython.display import display, HTML

import os
import pickle as pkl
from utils import mesh_operations
from utils.helpers import *

import mlflow.pytorch
from mlflow.tracking import MlflowClient

# %%
# # %%javascript
# $('<div id="toc"></div>').css({position: 'fixed', top: '120px', left: 0}).appendTo(document.body);
# $.getScript('https://kmahelona.github.io/ipython_notebook_goodies/ipython_notebook_toc.js');

# %% [markdown]
# **Select configuration file**

# %%
config_files_w = widgets.Dropdown(
    options=[x for x in os.listdir("config_files") if x.endswith("yaml")],
    value="config_folded_c_and_s.yaml"
)
display(config_files_w)

# %%
config = load_yaml_config(os.path.join("config_files", config_files_w.value))

# %% [markdown]
# ## PyTorch Lightning DataModule

# %% [markdown]
# ### Synthetic Meshes

# %%
from data.SyntheticDataModules import SyntheticMeshesDM
from data.DataModules import CardiacMeshPopulationDataset, CardiacMeshPopulationDM
import vedo
from utils import mesh_operations

sphere = vedo.Sphere()
template_mesh = "data/vedo_sphere_template.vtk"
sphere.write(template_mesh, binary=False)

M, A, D, U = mesh_operations.generate_transform_matrices(
    Cardiac3DMesh(template_mesh), 
    config.network_architecture.pooling.parameters.downsampling_factors
)

A_t, D_t, U_t = ([scipy_to_torch_sparse(x) for x in X] for X in (A, D, U))
n_nodes = [len(M[i].v) for i in range(len(M))]

dm = SyntheticMeshesDM(
  batch_size=config.batch_size, 
  data_params=config.dataset.parameters.__dict__, 
  preprocessing_params=config.dataset.preprocessing
)
dm.setup()

# %% [markdown]
# ### Cardiac meshes

# %%
from main import get_coma_matrices, get_coma_args, get_dm_model_trainer

# %%
coma_args = get_coma_args(config, dm)

# %%
from models.Mo

# %%
from models.model import Coma4D
from models.model_c_and_s import Coma4D_C_and_S
from models.ComaLightningModule import CoMA
coma4D = Coma4D_C_and_S(**coma_args)
model = CoMA(coma4D, config)

# %%
from models.model_c_and_s import Mean_Aggregator, DFT_Aggregator, FCN_Aggregator

mean_agg = Mean_Aggregator()
dft_agg = DFT_Aggregator()
fcn_agg = FCN_Aggregator(features_in=20*32, features_out=10)

print(mean_agg(xx).shape)
print(dft_agg(xx).shape)
print(fcn_agg(xx).shape)

# %% [markdown]
# ## PyTorch Lightning Module

# %% [markdown]
# ### Testing PL module

# %%
# train
trainer = pl.Trainer()
trainer.fit(model, datamodule=dm)

# %% [markdown]
# ## Load data

# %%
# import objgraph
# objgraph.show_refs(config, max_depth=2, )

# %%
popu = CardiacMeshPopulation(config.root_folder)
print(popu.as_numpy_array().shape)

# %%
