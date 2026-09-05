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

# %%
import mlflow
import os, sys

import torch
import torch.nn.functional as F

import ipywidgets as widgets
from ipywidgets import interact
from IPython.display import Image
from mlflow.tracking import MlflowClient

import pickle as pkl
import pytorch_lightning as pl

from argparse import Namespace
import matplotlib.pyplot as plt

import surgeon_pytorch
from surgeon_pytorch import Inspect, get_layers

import numpy as np
from IPython import embed
sys.path.insert(0, '..')

# %% [markdown]
# # MLflow

# %%
TRACKING_URI = "file:///home/rodrigo/CISTIB/repos/CardiacMotion/mlruns"
mlflow.set_tracking_uri(TRACKING_URI)

# %%
# default_experiment = "Synthetic data"
default_experiment = "Synthetic data 2"

experiment_w = widgets.Select(
    options=[exp.name for exp in mlflow.list_experiments()],
    value=default_experiment
)
display(experiment_w)

# %%
exp_id = mlflow.get_experiment_by_name(experiment_w.value).experiment_id

# %% [markdown]
# ### Retrieving runs

# %%
# runs_list = mlflow.search_runs(experiment_ids=[exp_id], output_format="list")
runs_df = mlflow.search_runs(experiment_ids=[exp_id],)
runs_df = runs_df[runs_df.status == "FINISHED"].reset_index(drop=True)

# %%
test_ratio_cols = runs_df.columns[runs_df.columns.str.contains("test.*ratio")]
runs_df[test_ratio_cols]

# %% [markdown]
# ### Getting artifacts

# %%
client = MlflowClient()
local_dir = "/tmp/artifact_downloads"

if not os.path.exists(local_dir):
    os.mkdir(local_dir)

# %%
client._tracking_client.list_artifacts(
    runs_df.run_id[12]
)


# %%

# %%
def display_gif(i):
    kk = client.download_artifacts(
      runs_df.run_id[i], 
      "animations", 
      local_dir
    )
    gif_file = os.path.join(kk, os.listdir(kk)[0])
    gif = Image(data=open(gif_file,'rb').read(), format='png')
    print(runs_df[sorted(runs_df.columns[runs_df.columns.str.startswith("params")])].iloc[i])
    display(gif)    
    
run_id_w = widgets.IntSlider(min=0, max=12)
interact(display_gif, i=run_id_w);

# %%
print(run_id_w.value)
tmp_model_path = client.download_artifacts(
    runs_df.run_id[run_id_w.value], 
    "model", 
    local_dir
)

# %%
model_path = os.path.join(tmp_model_path, "data/model.pth")

# %%
# torch.load(os.path.basename(model))
model = torch.load(model_path, map_location=torch.device('cpu'))

# %%
runs_df[runs_df.columns[runs_df.columns.str.startswith("params.dataset")]]

# %% [markdown]
# ### Create (or retrieve) dataset

# %%
from data.synthetic.SyntheticMeshPopulation import SyntheticMeshPopulation
from data.SyntheticDataModules import SyntheticMeshesDataset
from torch.utils.data import DataLoader
from main import get_datamodule

# %% [markdown]
# Re-build configuration

# %%
runinfo = dict(runs_df.iloc[run_id_w.value])

dataset_params = {
    "N": 10,
    "amplitude_static_max": float(runinfo["params.dataset_max_static_amplitude"]),
    "amplitude_dynamic_max": float(runinfo["params.dataset_max_dynamic_amplitude"]),
    "T": int(runinfo["params.dataset_n_timeframes"]),
    "freq_max": int(runinfo["params.dataset_freq_max"]),
    "l_max": int(runinfo["params.dataset_l_max"]),
    "mesh_resolution": int(runinfo["params.dataset_resolution"]),
    "random_seed": runinfo["params.dataset_random_seed"]
}    

config = {
    "batch_size": 32,
    "parameters": dataset_params,
    "preprocessing": Namespace(**{ "center_around_mean": False })
}

config = Namespace(**config)

# %%
mesh_popu = SyntheticMeshesDataset(config.parameters, config.preprocessing)
meshes_loader = DataLoader(mesh_popu)


# %% [markdown]
# Adapt model from PyTorch Geometric 2.0.3 to 2.0.4 (attribute `__explain__` changes to `_explain` producing an `AttributeError`)

# %%
def upgrade_pyg_model_203_to_204(model):
    
    for i, _ in enumerate(model.model.cheb_enc):
        model.model.cheb_enc[i]._explain = model.model.cheb_enc[i].__explain__
        
    for i, _ in enumerate(model.model.cheb_dec_c):
        model.model.cheb_dec_c[i]._explain = model.model.cheb_dec_c[i].__explain__
    
    for i, _ in enumerate(model.model.cheb_dec_s):
        model.model.cheb_dec_s[i]._explain = model.model.cheb_dec_s[i].__explain__
    
    model.model.pool._explain = model.model.pool.__explain__
    
    return model


# %%
model = upgrade_pyg_model_203_to_204(model)


# %%
def get_temporal_z(model, x):
    
    '''
    model: PyTorch Lightning Module.
    x: temporal sequence of meshes (point clouds)
    '''
    
    x = subject[0]

    self = model.model
    
    if self.phase_input:
        x = self.phase_tensor(x)
    
    n_timeframes = config.parameters["T"]
    x = x.reshape(1, n_timeframes, -1, 2*self.filters_enc[0])
        
    for i in range(self.n_layers):  
        x = self.cheb_enc[i](x, self.A_edge_index[i], self.A_norm[i])
        x = F.relu(x)
        x = self.pool(x, self.downsample_matrices[i])
        
    
    x = x.reshape(x.shape[0], self.n_timeframes, self._n_features_before_z)    
    mu_c, mu_s = [], []
    
    # Iterate through time points
    for i in range(n_timeframes):
        mu = self.enc_lin_mu(x[:,i,:])
        mu_c.append(mu[:,:self.z_c])
        mu_s.append(mu[:,self.z_c:])
        
    # convert list of 1D-tensors to 2D numpy.array
    z_c = np.array([kk.detach().numpy() for kk in mu_c])
    z_s = np.array([kk.detach().numpy() for kk in mu_s])
    
    return {"z_c": z_c, "z_s": z_s}

# %%
for subject in meshes_loader:
    break

# %%
x = subject[0]
z = get_temporal_z(model, x)

# %% [markdown]
# ### Inspect intermediate layers with `surgeon_pytorch`

# %%
model_wrapped = Inspect(model, layer={'layer1': 'x1', 'layer2': 'x2'})

# %%
get_layers(model)

# %% [markdown]
# #### 

# %%
surgeon_pytorch.inspect.get_module(model, "model.cheb_enc")

# %%
surgeon_pytorch.inspect.get_module(model, "model.cheb_enc")


# %%
def plot_z_vs_t(i):
    z_s = z["z_s"]
    plt.plot(z_s[:,0,i-1])
    plt.ylabel(f'z_{i}')
    plt.title("z(t) before temporal aggregation")
    plt.show()

interact(plot_z_vs_t, i=widgets.IntSlider(min=1, max=z["z_s"].shape[-1]));
