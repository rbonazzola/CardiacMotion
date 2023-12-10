import os, sys
CARDIAC_MOTION = f"{os.environ['HOME']}/01_repos/CardiacMotionRL"
sys.path.append(CARDIAC_MOTION)
sys.path.append(f"{CARDIAC_MOTION}/utils")

import mlflow
from mlflow.tracking import MlflowClient

from tqdm import tqdm
from IPython import embed
import argparse

import glob
import re
import pickle as pkl
from easydict import EasyDict
import pprint

import random
import numpy as np
import pandas as pd 

import torch
from torch import Tensor
from torch.utils.data import TensorDataset, DataLoader, random_split, SubsetRandomSampler
from typing import Union, List, Optional

import ipywidgets as widgets
from ipywidgets import interact

from data.DataModules import CardiacMeshPopulationDM, CardiacMeshPopulationDataset
from utils.CardioMesh.CardiacMesh import Cardiac3DMesh, transform_mesh
from utils.image_helpers import generate_gif, merge_gifs_horizontally

from main_autoencoder_cardiac import *
from config.load_config import load_yaml_config

from models.Model3D import Encoder3DMesh, Decoder3DMesh
from models.Model4D import DECODER_C_ARGS, DECODER_S_ARGS, ENCODER_ARGS
from models.Model4D import DecoderStyle, DecoderContent, DecoderTemporalSequence 
from models.Model4D import EncoderTemporalSequence, AutoencoderTemporalSequence
from lightning.ComaLightningModule import CoMA_Lightning
from models.lightning.EncoderLightningModule import TemporalEncoderLightning
from models.TemporalAggregators import TemporalAggregator, FCN_Aggregator

################################################################################

mlflow_uri = "/home/rodrigo/01_repos/CardiacMotion/mlruns/"
mlflow.set_tracking_uri(mlflow_uri)

runs_df = mlflow.search_runs(experiment_ids=['4'])
    
if len(runs_df) == 0:
    raise ValueError(f"No runs found under URI {mlflow_uri} and experiment {experiment_ids}.")

runs_df = runs_df[runs_df["metrics.test_recon_loss"] < 3]
runs_df = runs_df.set_index(["experiment_id", "run_id"], drop=False)
# print(runs_df)

runs_df.artifact_uri = runs_df.artifact_uri.apply(lambda x: x.replace("/home/rodrigo/CISTIB/repos/", "/mnt/data/workshop/workshop-user1/output/"))
runs_df.artifact_uri = runs_df.artifact_uri.apply(lambda x: x.replace("/home/home01/scrb/01_repos/", "/mnt/data/workshop/workshop-user1/output/"))    
runs_df.artifact_uri = runs_df.artifact_uri.apply(lambda x: x.replace("/1/", "/3/"))
runs_df.artifact_uri = runs_df.artifact_uri.apply(lambda x: x.replace("/user/", "/rodrigo/"))

################################################################################

class Run():
    
    def __init__(self, run_id, exp_id):
        
        self.exp_id = exp_id
        self.run_id = run_id

        self._PARTITION = "left_ventricle"
        self._FACES_FILE = "utils/CardioMesh/data/faces_and_downsampling_mtx_frac_0.1_LV.pkl"
        self._MEAN_ACROSS_CYCLE_FILE = f"utils/CardioMesh/data/cached/mean_shape_time_avg__{self._PARTITION}.npy"
        self._PROCRUSTES_FILE = f"utils/CardioMesh/data/cached/procrustes_transforms_{self._PARTITION}.pkl"
        self._SUBSETTING_MATRIX_FILE = f"/home/user/01_repos/CardioMesh/data/cached/subsetting_matrix_{self._PARTITION}.pkl"
    
    def get_ckpt_path(self):
        
        checkpoint_locations = {}
    
        for i, row in runs_df.iterrows():
            
            artifact_uri = row.artifact_uri
            artifact_uri = artifact_uri.replace("file://", "")
            checkpoints = []
            try:
                basepath = os.path.join(artifact_uri, "restored_model_checkpoint")        
                checkpoints += [ os.path.join(basepath, x) for x in os.listdir(basepath)]
            except FileNotFoundError:
                pass
            
            try:
                basepath = os.path.join(os.path.dirname(artifact_uri), "checkpoints")        
                checkpoints += [ os.path.join(basepath, x) for x in os.listdir(basepath)]
            except FileNotFoundError:
                pass
            
            if len(checkpoints) > 1:
                # finetuned_runs_2/1/3b09d025cc1446f3a0c27f9b27b69340/checkpoints/epoch=129-val_recon_loss=0.4883_val_kld_loss=94.8229.ckpt.ckpt
                regex = re.compile(".*epoch=(.*)-.*.ckpt")
                epochs = []
                for chkpt_file in checkpoints:
                    epoch = int(regex.match(chkpt_file).group(1))
                    epochs.append(epoch)
                argmax = epochs.index(max(epochs))
                chkpt_file = checkpoints[argmax]
            elif len(checkpoints) == 1:
                chkpt_file = checkpoints[0]
            elif len(checkpoints) == 0:
                chkpt_file = None
                
            checkpoint_locations[row.run_id] = chkpt_file
          
        checkpoint_locations = { 
            k:v for k,v in checkpoint_locations.items() if v is not None
        }
        
        return checkpoint_locations[self.run_id]
    # runs = checkpoint_locations.keys()
        
    def load_weights(self):
        
        ckpt_path = self.get_ckpt_path()
        model_weights = torch.load(ckpt_path, map_location=torch.device('cpu'))["state_dict"]
        print(f"Loaded weights from checkpoint:\n {ckpt_path}")
        model_weights = EasyDict({k.replace("model.", ""): v for k, v in model_weights.items()})        
        model_weights = EasyDict({k.replace("z_aggr_function", "z_aggr_function_mu"): v for k, v in model_weights.items()})
        
        return model_weights
    
    
    def build_model_from_ckpt(self):
        
        try:
            model_weights["encoder.encoder_3d_mesh.layers.layer_2.graph_conv.lins.11.weight"]
            POLYNOMIAL_DEGREE = 12
        except:
            POLYNOMIAL_DEGREE = 10
    
        config.network_architecture.convolution.parameters.polynomial_degree = [POLYNOMIAL_DEGREE] * 4
        ################################################
      
        subsetting_matrix = pkl.load(open(SUBSETTING_MATRIX_FILE, "rb"))
    
        template = EasyDict({
          "v": np.load(MEAN_ACROSS_CYCLE_FILE),
          "f": fhm_mesh[partitions[PARTITION]].f
        })


    def _get_z_path(self):
        
        z_file = f"{mlflow_uri}/4/{self.run_id}/artifacts/latent_vector.csv"
        if not os.path.exists(z_file):
            raise FileNotFoundError(f"{z_file} does not exist. Skipping...")
            
        return z_file

    def get_z_df(self):
        
        z_file = self._get_z_path()
        z_df = pd.read_csv(z_file).set_index("ID")
        return z_df
    


def get_model(polynomial_degree=10):

    from main_autoencoder_cardiac import get_coma_args
    
    partitions = {
      "left_atrium" : ("LA", "MVP", "PV1", "PV2", "PV3", "PV4", "PV5"),
      "right_atrium" : ("RA", "TVP", "PV6", "PV7"),
      "left_ventricle" : ("LV", "AVP", "MVP"),
      "right_ventricle" : ("RV", "PVP", "TVP"),
      "biventricle" : ("LV", "AVP", "MVP", "RV", "PVP", "TVP"),
      "aorta" : ("aorta",)
    }
    
    PARTITION = "left_ventricle"
    FACES_FILE = "utils/CardioMesh/data/faces_and_downsampling_mtx_frac_0.1_LV.pkl"
    MEAN_ACROSS_CYCLE_FILE = f"utils/CardioMesh/data/cached/mean_shape_time_avg__{PARTITION}.npy"
    PROCRUSTES_FILE = f"utils/CardioMesh/data/cached/procrustes_transforms_{PARTITION}.pkl"    
    SUBSETTING_MATRIX_FILE = f"utils/CardioMesh/data/cached/subsetting_matrix_{PARTITION}.pkl" 
    MESHES_PATH = "/home/rodrigo/01_repos/CardiacMotionRL/data/cardio/meshes"
    
    subsetting_matrix = pkl.load(open(SUBSETTING_MATRIX_FILE, "rb"))
    
    ID = "1000511"
    fhm_mesh = Cardiac3DMesh(
       filename=f"/home/rodrigo/doctorado/data/meshes/Results/{ID}/models/FHM_res_0.1_time001.npy",
       faces_filename="/home/rodrigo/01_repos/CardioMesh/data/faces_fhm_10pct_decimation.csv",
       subpart_id_filename="/home/rodrigo/01_repos/CardioMesh/data/subpartIDs_FHM_10pct.txt"
    )
    mean_shape = np.load(MEAN_ACROSS_CYCLE_FILE)
    faces = fhm_mesh[partitions[PARTITION]].f
    template = EasyDict({ "v": mean_shape, "f": faces })
    
    N_subj = 10
    NT = 10
    PHASES = 1+(50/NT)*np.array(range(NT)) # 1, 6, 11, 16, 21...
    
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
       
    POLYNOMIAL_DEGREE = polynomial_degree
    DOWNSAMPLING = 3
    
    config = load_yaml_config("config_folded_c_and_s.yaml")
    config.network_architecture.convolution.parameters.polynomial_degree = [POLYNOMIAL_DEGREE] * 4
    config.network_architecture.pooling.parameters.downsampling_factors = [3, 3, 2, 2] # * 4
    config.network_architecture.latent_dim_c = 8 
    config.network_architecture.latent_dim_s = 8
    config.loss.regularization.weight = 0
    
    coma_args = get_coma_args(config)
    coma_matrices = get_coma_matrices(config, template, PARTITION)
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
    decoder_style = DecoderStyle(decoder_config_s, phase_embedding_method="exp_v1")
    t_decoder = DecoderTemporalSequence(decoder_content, decoder_style, is_variational=coma_args.is_variational)
        
    t_ae = AutoencoderTemporalSequence(encoder=t_encoder, decoder=t_decoder, is_variational=coma_args.is_variational)
    t_ae.decoder._mode = "inference"
    
    return t_ae