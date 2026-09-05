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
import os
os.environ["CARDIAC_MOTION_REPO"] = f"{os.environ['HOME']}/01_repos/CardiacMotion"
repo_dir = os.environ.get("CARDIAC_MOTION_REPO", "kk")
os.chdir(repo_dir)

# %%
import numpy as np

from utils import mesh_operations
from utils.helpers import *
from config.load_config import load_yaml_config
from models.Model4D import AutoencoderTemporalSequence

from main_autoencoder_cardiac import get_coma_matrices, get_datamodule

import pytorch_lightning as pl
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import MLFlowLogger

from easydict import EasyDict
from models.Model4D import ENCODER_ARGS, DECODER_C_ARGS, DECODER_S_ARGS

import mlflow
from mlflow.tracking import MlflowClient

from config.cli_args import CLI_args, overwrite_config_items
from config.load_config import load_yaml_config, to_dict

from utils.mlflow_helpers import get_mlflow_parameters, get_mlflow_dataset_params
from utils.CardioMesh.CardiacMesh import Cardiac3DMesh

# %%
from data.DataModules import CardiacMeshPopulationDataset, CardiacMeshPopulationDM
from data.SyntheticDataModules import SyntheticMeshesDM

# %%
partitions = {
  "left_atrium" : ("LA", "MVP", "PV1", "PV2", "PV3", "PV4", "PV5"),
  "right_atrium" : ("RA", "TVP", "PV6", "PV7"),
  "left_ventricle" : ("LV", "AVP", "MVP"),
  "right_ventricle" : ("RV", "PVP", "TVP"),
  "biventricle" : ("LV", "AVP", "MVP", "RV", "PVP", "TVP"),
  "aorta" : ("aorta",)
}

# %%
PARTITION = "left_ventricle" # args.partition
FACES_FILE = "CardioMesh/data/faces_and_downsampling_mtx_frac_0.1_LV.pkl"
MEAN_ACROSS_CYCLE_FILE = f"CardioMesh/data/cached/mean_shape_time_avg__{PARTITION}.npy"
PROCRUSTES_FILE = f"CardioMesh/data/cached/procrustes_transforms_{PARTITION}.pkl"    
SUBSETTING_MATRIX_FILE = f"CardioMesh/data/cached/subsetting_matrix_{PARTITION}.pkl" 
MESHES_PATH = "CardiacMotion/data/cardio/Results"

subsetting_matrix = pkl.load(open(SUBSETTING_MATRIX_FILE, "rb"))

ID = "1000511"
fhm_mesh = Cardiac3DMesh(
   filename=f"/mnt/data/workshop/workshop-user1/datasets/meshes/Results_Yan/{ID}/models/FHM_res_0.1_time001.npy",
   faces_filename="CardioMesh/data/faces_fhm_10pct_decimation.csv",
   subpart_id_filename="CardioMesh/data/subpartIDs_FHM_10pct.txt"
)
mean_shape = np.load(MEAN_ACROSS_CYCLE_FILE)
faces = fhm_mesh[partitions[PARTITION]].f
template = EasyDict({ "v": mean_shape, "f": faces })

N_subj = 100

NT = 50
PHASES = 1+(50/NT)*np.array(range(NT)) # e.g. 1, 6, 11, 16, 21... if NT == 10

cardiac_dataset = CardiacMeshPopulationDataset(
    root_path=MESHES_PATH, 
    procrustes_transforms=PROCRUSTES_FILE,
    faces=faces,
    subsetting_matrix=subsetting_matrix,
    template_mesh= EasyDict({ "v": mean_shape, "f": faces }),
    N_subj=N_subj,
    phases_filter=PHASES,
    static_shape="end_diastole" # "end_diastole"
)

# %%
cardiac_dataset[0].d_content
