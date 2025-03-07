import os, sys
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

from data.DataModules import CardiacMeshPopulationDM, CardiacMeshPopulationDataset
from cardio_mesh import Cardiac3DMesh
from cardio_mesh.procrustes import transform_mesh

from cardiac_motion import PKG_DIR, MLFLOW_URI

from .image_helpers import generate_gif, merge_gifs_horizontally

from config.load_config import load_yaml_config

from cardiac_motion import (
    Encoder3DMesh,
    FCN_Aggregator,
    EncoderTemporalSequence,
    DecoderContent,
    DecoderStyle,
    DecoderTemporalSequence,    
    AutoencoderTemporalSequence,
    ENCODER_ARGS,
    DECODER_C_ARGS,
    DECODER_S_ARGS,        
)

from lightning_modules.ComaLightningModule import CoMA_Lightning
from lightning_modules.EncoderLightningModule import TemporalEncoderLightning

################################################################################


expid_to_partition_mapping = { 3: "RV", 4: "LV", 5: "BV", 6: "LA", 7: "RA", 8: "aorta" }
partition_to_expid_mapping = { v: k for k, v in expid_to_partition_mapping.items() }

all_exp_ids = [ str(k) for k in expid_to_partition_mapping.keys() ]

################################################################################

def patch_artifact_uri(runs_df):

    runs_df.artifact_uri = runs_df.artifact_uri.apply(lambda x: x.replace("/home/rodrigo/CISTIB/repos/", "/mnt/data/workshop/workshop-user1/output/"))
    runs_df.artifact_uri = runs_df.artifact_uri.apply(lambda x: x.replace("/home/home01/scrb/01_repos/", "/mnt/data/workshop/workshop-user1/output/"))    
    runs_df.artifact_uri = runs_df.artifact_uri.apply(lambda x: x.replace("/1/", "/3/"))
    runs_df.artifact_uri = runs_df.artifact_uri.apply(lambda x: x.replace("/user/", "/rodrigo/"))

    return runs_df


def get_runs(exp_ids='all', metric="metrics.test_recon_loss_c", metric_threshold=3):
   
    if exp_ids == "all":
        exp_ids = all_exp_ids

    mlflow.set_tracking_uri(MLFLOW_URI)
    
    runs_df = mlflow.search_runs(experiment_ids=exp_ids)

    assert len(runs_df) > 0, f"No runs found under URI {MLFLOW_URI} and experiment {exp_ids}."

    runs_df = runs_df[runs_df[metric] < metric_threshold]
    runs_df = runs_df.set_index(["experiment_id", "run_id"], drop=False)
    runs_df = patch_artifact_uri(runs_df)
        
    return runs_df


################################################################################

import cardio_mesh
ONE_RANDOM_ID = "1000511"; END_DIASTOLE = 1
template_fhm_mesh = cardio_mesh.load_full_heart_mesh(ONE_RANDOM_ID, timeframe=END_DIASTOLE)

################################################################################

class Run():
    
    def __init__(self, runinfo, load_model=True, load_dataloader=True):
        
        self.exp_id = runinfo.experiment_id
        self.run_id = runinfo.run_id

        self.partition = partition_mapping[int(self.exp_id)]

        # self._FACES_FILE = "utils/CardioMesh/data/faces_and_downsampling_mtx_frac_0.1_LV.pkl"
        self._MEAN_ACROSS_CYCLE_FILE = f"{cardio_mesh.BASEDIR}/data/cached/mean_shape_time_avg__{self.partition}.npy"
        self._PROCRUSTES_FILE = f"{cardio_mesh.BASEDIR}/data/cached/procrustes_transforms_{self.partition}.pkl"
        
        self.mean_shape = cardio_mesh.paths.get_mean_shape(partition)
        self._SUBSETTING_MATRIX_FILE = cardio_mesh.paths.get_subsetting_matrix(self.partition)
        
        # np.load(self._MEAN_ACROSS_CYCLE_FILE)
        self.faces = template_fhm_mesh[partitions[self.partition]].f
        
        self.model_weights = None
        self.z_df = None
        
        self.template = EasyDict({
          "v": np.load(self._MEAN_ACROSS_CYCLE_FILE),
          "f": fhm_mesh[partitions[self.partition]].f
        })

        if load_model:      self.load_model()
        if load_dataloader: self.load_dataloader()
    
    
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
        
        NT = 10

        subsetting_matrix = pkl.load(open(self._SUBSETTING_MATRIX_FILE, "rb"))
    
        print("Loading datamodule...")
        cardiac_dataset = CardiacMeshPopulationDataset(
            root_path=cardio_mesh.MESHES_DIR, 
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
        
    
    # def build_model_from_ckpt(self):
    #     
    #     polynomial_degree = self.get_polynomial_degree()
    # 
    #     config.network_architecture.convolution.parameters.polynomial_degree = [polynomial_degree] * 4
    #     ################################################
    #   
    #     subsetting_matrix = pkl.load(open(SUBSETTING_MATRIX_FILE, "rb"))
    # 
    #     template = EasyDict({
    #       "v": np.load(MEAN_ACROSS_CYCLE_FILE),
    #       "f": fhm_mesh[partitions[PARTITION]].f
    #     })

        
    def _get_z_path(self):
        
        z_file = f"{MLFLOW_URI}/{self.exp_id}/{self.run_id}/artifacts/latent_vector.csv"
            
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

    
def get_model(partition="LV", polynomial_degree=10, n_channels=[16, 16, 32, 32]):

    from main_autoencoder_cardiac import get_coma_args
    
    partitions = cardio_mesh.Constants.closed_partitions
    
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
    
    model = AutoencoderTemporalSequence.build_from_config(config, mesh_template, partition, n_timeframes)

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


def fetch_loci_mapping():

    import requests
    from io import StringIO
    # https://docs.google.com/spreadsheets/d/1LbILFyaTHeRPit8v3gwx2Db4uS1Hnx6dibeGHK9zXcU/edit?usp=sharing
    # LINK = 'https://docs.google.com/spreadsheet/ccc?key=1LbILFyaTHeRPit8v3gwx2Db4uS1Hnx6dibeGHK9zXcU&output=csv'
    LINK = 'https://docs.google.com/spreadsheet/ccc?key=1XvVDFZSvcWWyVaLaQuTpglOqrCGB6Kdf6c78JJxymYw&output=csv'
    response = requests.get(LINK)
    assert response.status_code == 200, 'Wrong status code'
    loci_mapping_df = pd.read_csv(
        StringIO(response.content.decode()),
        sep=","
    ).set_index("region")
    
    return loci_mapping_df