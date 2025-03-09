import os, sys
import mlflow
from mlflow.tracking import MlflowClient

from tqdm import tqdm

import re
import pickle as pkl
from easydict import EasyDict

import numpy as np
import pandas as pd 

import torch
from torch import Tensor
from torch.utils.data import TensorDataset, DataLoader, random_split, SubsetRandomSampler

from typing import Union, List, Optional

from data.DataModules import CardiacMeshPopulationDM, CardiacMeshPopulationDataset

import cardio_mesh
from cardio_mesh import (
    Cardiac3DMesh,
    close_chamber
)

from cardio_mesh.paths import (
    get_mean_shape,
    get_procrustes_file,
    get_procrustes_transforms,
    get_subsetting_matrix
)

from cardio_mesh.procrustes import transform_mesh

from cardiac_motion import PKG_DIR, MLFLOW_URI
from cardiac_motion import AutoencoderTemporalSequence

from .image_helpers import generate_gif, merge_gifs_horizontally

from config.load_config import load_yaml_config


################################################################################

def patch_artifact_uri(runs_df):

    runs_df.artifact_uri = runs_df.artifact_uri.apply(lambda x: x.replace("/home/rodrigo/CISTIB/repos/", "/mnt/data/workshop/workshop-user1/output/"))
    runs_df.artifact_uri = runs_df.artifact_uri.apply(lambda x: x.replace("/home/home01/scrb/01_repos/", "/mnt/data/workshop/workshop-user1/output/"))    
    runs_df.artifact_uri = runs_df.artifact_uri.apply(lambda x: x.replace("/1/", "/3/"))
    runs_df.artifact_uri = runs_df.artifact_uri.apply(lambda x: x.replace("/user/", "/rodrigo/"))
            
    return runs_df

################################################################################

class Run():

    ONE_RANDOM_ID = "1000511"; END_DIASTOLE = 1
    template_fhm_mesh: Cardiac3DMesh = cardio_mesh.load_full_heart_mesh(ONE_RANDOM_ID, timeframe=END_DIASTOLE)

    expid_to_partition_mapping = { 3: "RV", 4: "LV", 5: "BV", 6: "LA", 7: "RA", 8: "aorta" }
    partition_to_expid_mapping = { v: k for k, v in expid_to_partition_mapping.items() }

    all_exp_ids = [ str(k) for k in expid_to_partition_mapping.keys() ]

    mlflow.set_tracking_uri(MLFLOW_URI := MLFLOW_URI)
    
    runs_df = None
    checkpoint_locations = None

    @classmethod
    def get_runs(cls, exp_ids='all', metric="metrics.test_recon_loss_c", metric_threshold=3):
         
        if exp_ids == "all":
            exp_ids = cls.all_exp_ids
            
        runs_df = mlflow.search_runs(experiment_ids=exp_ids)
    
        assert len(runs_df) > 0, f"No runs found under URI {MLFLOW_URI} and experiment {exp_ids}."
    
        runs_df = runs_df[runs_df[metric] < metric_threshold]
        runs_df = runs_df.set_index(["experiment_id", "run_id"], drop=False)
        runs_df = patch_artifact_uri(runs_df)
            
        return runs_df


    @staticmethod
    def list_runs(exp_ids='all'):
        
        runs_df = Run.get_runs(exp_ids)
        return runs_df
    
 
    def __init__(self, exp_id, run_id, load_model=True, load_dataloader=True):
        
        Run.runs_df = Run.get_runs()

        self.exp_id = exp_id
        self.run_id = run_id

        self.RUN_BASE_DIR = f"{MLFLOW_URI}/{self.exp_id}/{self.run_id}"

        assert os.path.exists(self.RUN_BASE_DIR), f"Run directory {self.RUN_BASE_DIR} does not exist."

        self.partition = self.get_heart_partition()
        self.template = Run.get_template(self.partition)
        
        self.model_weights = None
        self.z_df = None

        self._static_variables = None
        self._dynamic_variables = None

        self.model = None if not load_model else self.build_model_from_checkpoint()
        self.dataloader = None if not load_dataloader else self.load_dataloader()
            

    def get_heart_partition(self):
        return Run.expid_to_partition_mapping[int(self.exp_id)]


    @staticmethod
    def get_template(partition):
        
        template = EasyDict({
          "v": get_mean_shape(partition),
          "f": Run.template_fhm_mesh[close_chamber(partition)].f
        })

        return template


    @classmethod
    def from_dict(cls, runinfo, load_model=False, load_dataloader=False):
        return cls(exp_id=runinfo['exp_id'], run_id=runinfo['run_id'], load_model=load_model, load_dataloader=load_dataloader)


    @classmethod
    def get_all_ckpt_paths(cls):
        
        checkpoint_locations = {}
    
        for i, row in cls.runs_df.iterrows():
            
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
        
        cls.checkpoint_locations = checkpoint_locations
        return checkpoint_locations


    def get_ckpt_path(self):        
        return Run.checkpoint_locations.get(self.run_id, None)
        
    
    def load_weights(self):
        
        if Run.checkpoint_locations is None:
            Run.get_all_ckpt_paths()

        ckpt_path = self.get_ckpt_path()
        model_weights = torch.load(ckpt_path, map_location=torch.device('cpu'))["state_dict"]
        print(f"Loaded weights from checkpoint:\n {ckpt_path}")
        model_weights = EasyDict({k.replace("model.", ""): v for k, v in model_weights.items()})        
        model_weights = EasyDict({k.replace("z_aggr_function", "z_aggr_function_mu"): v for k, v in model_weights.items()})
        self.model_weights = model_weights
        return model_weights
    
    
    def load_dataloader(self):
                
        subsetting_matrix = get_subsetting_matrix(self.partition)

        from utils.helpers import get_n_equispaced_timeframes
    
        print("Loading datamodule...")
        cardiac_dataset = CardiacMeshPopulationDataset(
            root_path=cardio_mesh.MESHES_DIR, 
            procrustes_transforms=get_procrustes_file(self.partition),
            faces=self.template.f,
            subsetting_matrix=subsetting_matrix,
            template_mesh=self.template,
            N_subj=None,
            phases_filter=get_n_equispaced_timeframes(self.get_n_timeframes())
        )
        
        print(f"Length of dataset: {len(cardiac_dataset)}")  
        mesh_dl = torch.utils.data.DataLoader(cardiac_dataset, batch_size=128, num_workers=16)
        
        return mesh_dl
        
    
    def _get_z_path(self):
       
        return f"{self.RUN_BASE_DIR}/artifacts/latent_vector.csv"
            
    
    def generate_z_df(self, overwrite=False):
        
        zfile = self._get_z_path()
        
        if not os.path.exists(zfile) or overwrite:
            torch.cuda.empty_cache()        
            zs = []
            for i, x in enumerate(self.dataloader):
                if i < (len(zs)-1):
                    continue
                x['s_t'] = x['s_t'].to("cuda:0")
                z = self.model.encoder(x['s_t'])
                z = z['mu'].detach().cpu().numpy()
                zs.append(z)
                torch.cuda.empty_cache() 
            
            zs_concat = np.concatenate(zs)
            z_df = pd.DataFrame(zs_concat, index=self.dataloader.dataset.ids)
            del zs_concat, zs
            
            z_df.columns = self.static_variables + self.dynamic_variables
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
    
    
    def get_n_timeframes(self):
        return self.model_weights["encoder.z_aggr_function_mu_mu.fcn.weight"].shape[1] // self.model_weights['decoder.decoder_style.decoder_3d.dec_lin.weight'].shape[0]


    def get_downsampling_factors(self):
        file = f"{self.RUN_BASE_DIR}/params/reduction_factors"
        assert os.path.exists(file), f"File {file} does not exist."        
        return [int(x) for x in open(file, "rt").read().replace("[", "").replace("]", "").split(",")]
    

    def build_model_config(self):

        config = load_yaml_config("config_folded_c_and_s.yaml")
        config.network_architecture.latent_dim_c = self.get_latent_dim_c()
        config.network_architecture.latent_dim_s = self.get_latent_dim_s()
        config.network_architecture.convolution.channels_enc = self.get_n_channels()
        config.network_architecture.convolution.channels_dec_c = self.get_n_channels()
        config.network_architecture.convolution.channels_dec_s = self.get_n_channels()
        config.network_architecture.convolution.parameters.polynomial_degree = [self.get_polynomial_degree()] * 4
        config.network_architecture.pooling.parameters.downsampling_factors = self.get_downsampling_factors()
        config.loss.regularization.weight = 0
        
        return config


    def build_model_from_checkpoint(self):
        
        model_weights = self.load_weights()
        model = self.build_model_architecture()
        model.load_state_dict(model_weights, strict=False)        
        return model

    
    def build_model_architecture(self):

        config = self.build_model_config()
        n_timeframes = self.get_n_timeframes()

        model = AutoencoderTemporalSequence.build_from_config(config, self.template, self.partition, n_timeframes)
        return model


    def get_latent_dim_c(self):

        latent_dim_c_file = f"{self.RUN_BASE_DIR}/params/latent_dim_c"
        assert os.path.exists(latent_dim_c_file), f"File {latent_dim_c_file} does not exist."
        
        latent_dim_c = open(latent_dim_c_file, "rt").read()        
        return int(latent_dim_c)
    

    def get_latent_dim_s(self):
        
        latent_dim_s_file = f"{self.RUN_BASE_DIR}/params/latent_dim_s"
        assert os.path.exists(latent_dim_s_file), f"File {latent_dim_s_file} does not exist."
        
        latent_dim_s = open(latent_dim_s_file, "rt").read()
        return int(latent_dim_s)

    
    @property
    def static_variables(self):
        if self._static_variables is None:
            self._static_variables = [ f"z{str(i).zfill(3)}" for i in range(self.get_latent_dim_c()) ]
        return self._static_variables
    
    
    @property
    def dynamic_variables(self):
        if self._dynamic_variables is None:
            self._dynamic_variables = [ f"z{str(i).zfill(3)}" for i in range(self.get_latent_dim_c(), self.get_latent_dim_c() + self.get_latent_dim_s()) ]
        return self._dynamic_variables
        



# def get_model(partition="LV", polynomial_degree=10, n_channels=[16, 16, 32, 32], n_timeframes=10):
# 
#     config = load_yaml_config("config_folded_c_and_s.yaml")
#     
#     mesh_template = 
# 
#     model = AutoencoderTemporalSequence.build_from_config(config, mesh_template, partition, n_timeframes)
# 
#     x = 
#        
#     
#   
#     if torch.cuda.is_available():
#         model = model.to("cuda:0")
#     
#     return model