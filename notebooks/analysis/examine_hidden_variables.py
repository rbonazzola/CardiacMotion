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
import sys, os
os.chdir("..")

import utils.CardioMesh
sys.path.append("utils/CardioMesh/")

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

from utils.run_helpers import Run

# %%
MESHES_PATH = "/mnt/data/workshop/workshop-user1/datasets/meshes/Results_Yan/"

fhm_mesh = Cardiac3DMesh(
  filename=f"{MESHES_PATH}/1000511/models/FHM_res_0.1_time001.npy",
  faces_filename=f"{os.environ['HOME']}/01_repos/CardioMesh/data/faces_fhm_10pct_decimation.csv",
  subpart_id_filename=f"{os.environ['HOME']}/01_repos/CardioMesh/data/subpartIDs_FHM_10pct.txt"
)


def get_model(partition="left_ventricle", polynomial_degree=10, n_channels=[16, 16, 32, 32], nt_encoder=10, nt_decoder=50):

    from main_autoencoder_cardiac import get_coma_args
    
    partitions = {
      "left_atrium" : ("LA", "MVP", "PV1", "PV2", "PV3", "PV4", "PV5"),
      "right_atrium" : ("RA", "TVP", "PV6", "PV7"),
      "left_ventricle" : ("LV", "AVP", "MVP"),
      "right_ventricle" : ("RV", "PVP", "TVP"),
      "biventricle" : ("LV", "AVP", "MVP", "RV", "PVP", "TVP"),
      "aorta" : ("aorta",)
    }
    
    # FACES_FILE = "utils/CardioMesh/data/faces_and_downsampling_mtx_frac_0.1_LV.pkl"
    MEAN_ACROSS_CYCLE_FILE = f"utils/CardioMesh/data/cached/mean_shape_time_avg__{partition}.npy"
    PROCRUSTES_FILE = f"utils/CardioMesh/data/cached/procrustes_transforms_{partition}.pkl"    
    SUBSETTING_MATRIX_FILE = f"utils/CardioMesh/data/cached/subsetting_matrix_{partition}.pkl" 
    
    subsetting_matrix = pkl.load(open(SUBSETTING_MATRIX_FILE, "rb"))
    
    mean_shape = np.load(MEAN_ACROSS_CYCLE_FILE)
    faces = fhm_mesh[partitions[partition]].f
    template = EasyDict({ "v": mean_shape, "f": faces })
    
    N_subj = 10
    NT = nt_encoder
    PHASES = 1 + (50/NT) * np.array(range(NT)) # 1, 6, 11, 16, 21...
    
    cardiac_dataset = CardiacMeshPopulationDataset(
        root_path=MESHES_PATH, 
        procrustes_transforms=PROCRUSTES_FILE,
        faces=faces,
        subsetting_matrix=subsetting_matrix,
        template_mesh= EasyDict({ "v": mean_shape, "f": faces }),
        N_subj=N_subj,
        phases_filter=PHASES
    )
    
    mesh_dm = CardiacMeshPopulationDM(cardiac_dataset, batch_size=8)        
       
    mesh_dm.setup()
    x = EasyDict(next(iter(mesh_dm.train_dataloader())))
       
    config = load_yaml_config("config_folded_c_and_s.yaml")
    
    POLYNOMIAL_DEGREE = polynomial_degree
    DOWNSAMPLING_FACTORS = [3, 3, 2, 2]
    config.network_architecture.convolution.channels_enc = n_channels
    config.network_architecture.convolution.channels_dec_c = n_channels
    config.network_architecture.convolution.channels_dec_s = n_channels
    config.network_architecture.convolution.parameters.polynomial_degree = [POLYNOMIAL_DEGREE] * 4
    config.network_architecture.pooling.parameters.downsampling_factors = DOWNSAMPLING_FACTORS
    
    config.network_architecture.latent_dim_c = 8 
    config.network_architecture.latent_dim_s = 8
    config.loss.regularization.weight = 0
    
    coma_args = get_coma_args(config)
    coma_matrices = get_coma_matrices(config, template, partition)
    coma_args.update(coma_matrices)
    
    enc_config = EasyDict({k: v for k, v in coma_args.items() if k in ENCODER_ARGS})
    encoder = Encoder3DMesh(**enc_config)
    
    enc_config.latent_dim = config.network_architecture.latent_dim_c + config.network_architecture.latent_dim_s 
    
    h = encoder.forward_conv_stack(x.s_t, preserve_graph_structure=False)
    
    z_aggr = FCN_Aggregator(features_in = NT*h.shape[-1], features_out= enc_config.latent_dim)
    t_encoder = EncoderTemporalSequence(encoder3d = encoder, z_aggr_function=z_aggr, is_variational=coma_args.is_variational)   
    
    decoder_config_c = EasyDict({ k:v for k,v in coma_args.items() if k in DECODER_C_ARGS })
    decoder_config_s = EasyDict({ k:v for k,v in coma_args.items() if k in DECODER_S_ARGS })
    decoder_content = DecoderContent(decoder_config_c)
    decoder_style = DecoderStyle(decoder_config_s, phase_embedding_method="exp_v1", n_timeframes=nt_decoder)
    t_decoder = DecoderTemporalSequence(decoder_content, decoder_style, is_variational=coma_args.is_variational)
        
    t_ae = AutoencoderTemporalSequence(encoder=t_encoder, decoder=t_decoder, is_variational=coma_args.is_variational)
    t_ae.decoder._mode = "inference"
    
    if torch.cuda.is_available():
        t_ae = t_ae.to("cuda:0")
    
    return t_ae


# %%
runinfo = Run.list_runs(experiment_id='4').sort_values('metrics.test_recon_loss_s').iloc[0]
run = Run(runinfo, load_model=True, load_dataloader=False)


# %%
@interact
def count_non_zero(i=widgets.IntSlider(min=0, max=9), k=widgets.IntSlider(min=0, max=100)):
    
    model = run.model.encoder.encoder_3d_mesh
    
    x = cardiac_dataset[i].s_t.to("cuda:0")
    
    y = []
    y.append(deepcopy(x))
    
    for i, layer in enumerate(model.layers): 
           
       print(model.layers[layer]["graph_conv"])
       
       t = model.layers[layer]["graph_conv"](y[-1], model.matrices["A_edge_index"][i], model.matrices["A_norm"][i])
       t = model.layers[layer]["pool"](t, model.matrices["downsample"][i])
       t = model.layers[layer]["activation_function"](t)
       y.append(t)
    
    # kk0 = run.model.encoder.encoder_3d_mesh(x.unsqueeze(0))['mu'][0][0].flatten() 
    
    global kk

    kk = y[3]
    
    # kk = [ 
    #        run.model.encoder.encoder_3d_mesh(x.unsqueeze(0))['mu'][0][t].flatten()[kk0!=0].cpu().detach().numpy() 
    #    for t in range(0, 50, 5) 
    # ]
    
    # pp = np.array(kk) / np.array(kk).max(0).reshape(1,-1)
    
    phases = list(range(0, 50, 2))
    
    # print(kkt[kk0 != 0])
    plt.plot(kk[phases,k].cpu().detach().numpy())
