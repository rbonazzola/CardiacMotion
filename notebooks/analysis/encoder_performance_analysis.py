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
import os, sys
os.chdir("..")
sys.path.append(os.getcwd())

# %%
import mlflow
import ipywidgets as widgets
from ipywidgets import interact
import os
import glob
import torch
from pprint import pprint
from easydict import EasyDict
from tqdm import tqdm

import numpy as np
import torch

from torch.utils.data import DataLoader
from data.SyntheticDataModules import SyntheticMeshesDataset, SyntheticMeshesDM
from utils.helpers import get_coma_args

# %%
from config.load_config import load_yaml_config
config = load_yaml_config("config_files/config_folded_c_and_s.yaml")
config.dataset.random_seed = 135

mesh_ds = SyntheticMeshesDataset(config.dataset.parameters, config.dataset.preprocessing)
mesh_dl = DataLoader(mesh_ds)
# mesh_dm = SyntheticMeshesDM(mesh_ds, )
# mesh_dm.setup()

# %%
from torch_models import Encoder3DMesh, EncoderTemporalSequence, FCN_Aggregator

# %%
ckpt_path = "/app/Rodrigo_repos/CardiacMotion/model.pt"
model_weights = torch.load(ckpt_path)["model_state_dict"]
model_weights = EasyDict(model_weights)

# %%
coma_args = get_coma_args(config, mesh_dl.dataset)
x = EasyDict(next(iter(mesh_dl)))

# %%
enc_params = {
    "phase_input" : False, 
    "num_conv_filters_enc" : [8, 8, 8, 8], 
    "num_features" : 3,
    "cheb_polynomial_order" : [6, 6, 6, 6],
    "n_layers" : 4,
    "n_nodes" : coma_args.n_nodes,
    "is_variational" : True,
    "latent_dim" : len(x.z_c) + len(x.z_s),
    "template": coma_args.template,
    "adjacency_matrices": coma_args.adjacency_matrices,
    "downsample_matrices": coma_args.downsample_matrices,
    "activation_layers": ["Sigmoid", "Sigmoid", "Sigmoid", None]
}

enc_params = EasyDict(enc_params)

encoder = Encoder3DMesh(**enc_params)

h = encoder.forward_conv_stack(x.s_t, preserve_graph_structure=False)

z_taggr = FCN_Aggregator(
    features_in=20*h.shape[-1],
    features_out=enc_params.latent_dim
)

t_encoder = EncoderTemporalSequence(
    encoder3d=encoder,
    z_aggr_function=z_taggr
)

#t_encoder_ptl = TemporalEncoderLightning(
#  model=t_encoder,
#  params=config
#)

# trainer = pl.Trainer(gpus=1)

# z = encoder.forward_conv_stack(x.s_t, preserve_graph_structure=False)
# z = z_taggr(z)
# z.shape

# %%
# {k.replace("model.", ""): v.shape for k, v in model_weights.items()}

# %%
mse_zc = []
mse_zs = []

for batch in tqdm(mesh_dl):
    
  z_c = torch.Tensor(batch["z_c"])  
  z_s = torch.Tensor(batch["z_s"])  
    
  z_c_hat = t_encoder(batch["s_t"])['mu'][0,:9]
  z_s_hat = t_encoder(batch["s_t"])['mu'][0,9:]  
  
  mse_zc.append(((z_c - z_c_hat)**2).detach().numpy())
  mse_zs.append(((z_s - z_s_hat)**2).detach().numpy())
    

# %%
np.array(mse_zc).mean(0)

# %%
np.array(mse_zs).mean(0)


# %%
@interact
def get_ckptpath(run=run_w):
    
    # chkpt_dir = f"{MLFLOW_TRACKING_URI}/{EXPERIMENT_ID}/{RUNID}/"
    
    REPO_DIR = "/app/Rodrigo_repos/CardiacMotion"
    global ckpt_path, model_weights    
    ckpt_path = glob.glob(f"{REPO_DIR}/1/{run}/checkpoints/*ckpt")    
    if len(ckpt_path) == 1:
      ckpt_path = ckpt_path[0]
    elif len(ckpt_path) == 0:
      ckpt_path = None
    
    model_weights = torch.load(ckpt_path)["state_dict"]
    model_weights = EasyDict(model_weights)
    
    print(mlflow.get_run(run).data.metrics["test_rec_ratio_to_pop_mean"])
    return ckpt_path


# %%
n_params = 0

for module_name, weights in model_weights.items():
    module_name = module_name.replace("model.", "")
    n_params += np.prod(weights.shape)
    # print(f'{module_name}: {weights.shape}')
    
n_params

# %%
