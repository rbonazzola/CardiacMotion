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

mlflow_uri = f"{os.environ['HOME']}/01_repos/CardiacMotion/mlruns/"
mlflow.set_tracking_uri(mlflow_uri)

runs_df = mlflow.search_runs(experiment_ids=['3', '4', '5', '6', '7', '8'])
    
if len(runs_df) == 0:
    raise ValueError(f"No runs found under URI {mlflow_uri} and experiment {experiment_ids}.")

runs_df = runs_df[runs_df["metrics.val_recon_loss_c"] < 3]
runs_df = runs_df.set_index(["experiment_id", "run_id"], drop=False)
# print(runs_df)

runs_df.artifact_uri = runs_df.artifact_uri.apply(lambda x: x.replace("/home/rodrigo/CISTIB/repos/", "/mnt/data/workshop/workshop-user1/output/"))
runs_df.artifact_uri = runs_df.artifact_uri.apply(lambda x: x.replace("/home/home01/scrb/01_repos/", "/mnt/data/workshop/workshop-user1/output/"))    
runs_df.artifact_uri = runs_df.artifact_uri.apply(lambda x: x.replace("/1/", "/3/"))
# runs_df.artifact_uri = runs_df.artifact_uri.apply(lambda x: x.replace("/user/", "/rodrigo/"))

################################################################################

MESHES_PATH = "/mnt/data/workshop/workshop-user1/datasets/meshes/Results_Yan/"

fhm_mesh = Cardiac3DMesh(
  filename=f"{MESHES_PATH}/1000511/models/FHM_res_0.1_time001.npy",
  faces_filename=f"{os.environ['HOME']}/01_repos/CardioMesh/data/faces_fhm_10pct_decimation.csv",
  subpart_id_filename=f"{os.environ['HOME']}/01_repos/CardioMesh/data/subpartIDs_FHM_10pct.txt"
)

################################################################################



################################################################################

class Run():
    
    def __init__(self, runinfo, load_model=True, load_dataloader=True):
        
        self.exp_id = runinfo.experiment_id
        self.run_id = runinfo.run_id

        self._partition_mapping = {   
            3: "right_ventricle",
            4: "left_ventricle",
            5: "biventricle",
            6: "left_atrium",
            7: "right_atrium",
            8: "aorta"
        }
        
        self.partition = self._partition_mapping[int(self.exp_id)]  
        self._FACES_FILE = "utils/CardioMesh/data/faces_and_downsampling_mtx_frac_0.1_LV.pkl"
        self._MEAN_ACROSS_CYCLE_FILE = f"utils/CardioMesh/data/cached/mean_shape_time_avg__{self.partition}.npy"
        self._PROCRUSTES_FILE = f"utils/CardioMesh/data/cached/procrustes_transforms_{self.partition}.pkl"
        self._SUBSETTING_MATRIX_FILE = f"/home/user/01_repos/CardioMesh/data/cached/subsetting_matrix_{self.partition}.pkl"
        self.mean_shape = np.load(self._MEAN_ACROSS_CYCLE_FILE)
        self.faces = fhm_mesh[partitions[self.partition]].f
        
        self.model_weights = None
        self.z_df = None
        
        if load_model:
            self.load_model()
            
        if load_dataloader:
            self.load_dataloader()
    
    
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
            k: v for k, v in checkpoint_locations.items() if v is not None
        }
        
        return checkpoint_locations[self.run_id]

    
    def load_weights(self):
        
        ckpt_path = self.get_ckpt_path()
        model_weights = torch.load(ckpt_path, map_location=torch.device('cpu'))["state_dict"]
        print(f"Loaded weights from checkpoint:\n {ckpt_path}")
        model_weights = EasyDict({k.replace("model.", ""): v for k, v in model_weights.items()})        
        model_weights = EasyDict({k.replace("z_aggr_function", "z_aggr_function_mu"): v for k, v in model_weights.items()})
        self.model_weights = model_weights
        return model_weights
    
    
    def load_dataloader(self):
        
        NT = 10 # config.dataset.parameters.T

        subsetting_matrix = pkl.load(open(self._SUBSETTING_MATRIX_FILE, "rb"))
    
        template = EasyDict({
          "v": np.load(self._MEAN_ACROSS_CYCLE_FILE),
          "f": fhm_mesh[partitions[self.partition]].f
        })
        
        print("Loading datamodule...")
        cardiac_dataset = CardiacMeshPopulationDataset(
            root_path=f"{os.environ['HOME']}/01_repos/CardiacMotion/data/cardio/Results", 
            procrustes_transforms=self._PROCRUSTES_FILE,
            faces=template.f,
            subsetting_matrix=subsetting_matrix,
            template_mesh=template,
            N_subj=None,
            phases_filter=1+(50/NT)*np.array(range(NT))
        )
        
        print(f"Length of dataset: {len(cardiac_dataset)}")  
        mesh_dl = torch.utils.data.DataLoader(cardiac_dataset, batch_size=128, num_workers=16)
        
        self.dataloader = mesh_dl
        
    
    def build_model_from_ckpt(self):
        
        polynomial_degree = self.get_polynomial_degree()
    
        config.network_architecture.convolution.parameters.polynomial_degree = [polynomial_degree] * 4
        ################################################
      
        subsetting_matrix = pkl.load(open(SUBSETTING_MATRIX_FILE, "rb"))
    
        template = EasyDict({
          "v": np.load(MEAN_ACROSS_CYCLE_FILE),
          "f": fhm_mesh[partitions[PARTITION]].f
        })

        
    def _get_z_path(self):
        
        z_file = f"{mlflow_uri}/{self.exp_id}/{self.run_id}/artifacts/latent_vector.csv"
            
        return z_file

    
    def generate_z_df(self):
        
        zfile = self._get_z_path()
        
        if not os.path.exists(zfile):    
        
            torch.cuda.empty_cache()
        
            zs = []
            
            for i, x in enumerate(self.dataloader):
                
                # if (i % 10) == 0:
                # print(i)
                    
                if i < (len(zs)-1):
                    continue
                
                x['s_t'] = x['s_t'].to("cuda:0")
                z = self.model.encoder(x['s_t'])
                z = z['mu'].detach().cpu().numpy()
                zs.append(z)
                
                # zs.append(z)
                torch.cuda.empty_cache() 
            
            zs_concat = np.concatenate(zs)
            z_df = pd.DataFrame(zs_concat, index=self.dataloader.dataset.ids)
            del zs_concat, zs
            
            # colnames before: 0, 1, 2, 3
            z_df.columns = [ f"z{str(i).zfill(3)}" for i in range(16) ]
            # colnames after: z000, z001, z002, z003
            
            z_df = z_df.reset_index().rename({"index": "ID"}, axis=1)
            z_df.to_csv(zfile, index=False)        
    
    
    def get_z_df(self):
        
        z_file = self._get_z_path()
        if not os.path.exists(z_file):
            raise FileNotFoundError(f"{z_file} does not exist. Skipping...")
                
        if self.z_df is None:
            z_df = pd.read_csv(z_file).set_index("ID")
            return z_df
        else:
            return self.z_df
    
    
    def get_polynomial_degree(self):
        
        regex = re.compile("encoder.encoder_3d_mesh.layers.layer_0.graph_conv.lins.(.*).weight")
        return max([int(regex.match(x).group(1)) for x in self.model_weights.keys() if regex.match(x)]) + 1
    
    
    def get_n_channels(self):
        
        return [self.model_weights[f"encoder.encoder_3d_mesh.layers.layer_{l}.graph_conv.lins.0.weight"].shape[0] for l in range(0, 4)]
    
    
    def load_model(self):
        
        model_weights = self.load_weights()
        
        pol_degree = self.get_polynomial_degree()
        n_channels = self.get_n_channels()
        
        t_ae = get_model(partition=self.partition, polynomial_degree=pol_degree, n_channels=n_channels)
        t_ae.load_state_dict(model_weights, strict=False)
        
        self.model = t_ae

    
def get_model(partition="left_ventricle", polynomial_degree=10, n_channels=[16, 16, 32, 32]):

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
    NT = 10
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
    config.network_architecture.convolution.channels_enc = n_channels
    config.network_architecture.convolution.channels_dec_c = n_channels
    config.network_architecture.convolution.channels_dec_s = n_channels
    config.network_architecture.convolution.parameters.polynomial_degree = [POLYNOMIAL_DEGREE] * 4
    config.network_architecture.pooling.parameters.downsampling_factors = [3, 3, 2, 2] # * 4
    
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
    decoder_style = DecoderStyle(decoder_config_s, phase_embedding_method="exp_v1", n_timeframes=50)
    t_decoder = DecoderTemporalSequence(decoder_content, decoder_style, is_variational=coma_args.is_variational)
        
    t_ae = AutoencoderTemporalSequence(encoder=t_encoder, decoder=t_decoder, is_variational=coma_args.is_variational)
    t_ae.decoder._mode = "inference"
    
    if torch.cuda.is_available():
        t_ae = t_ae.to("cuda:0")
    
    return t_ae