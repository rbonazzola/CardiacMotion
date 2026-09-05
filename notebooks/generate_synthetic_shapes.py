#!/usr/bin/env python
# coding: utf-8

import os
import sys
import glob
import re
import pickle as pkl
from pprint import pprint
from typing import Union, List, Optional
from itertools import product
from easydict import EasyDict

import numpy as np
import pandas as pd
import torch
from torch import Tensor
from torch.utils.data import TensorDataset, DataLoader, random_split, SubsetRandomSampler

import mlflow
from mlflow.tracking import MlflowClient
from tqdm import tqdm
import ipywidgets as widgets
from ipywidgets import interact

# Set up paths and environment
CARDIAC_MOTION = os.path.join(os.environ['HOME'], "01_repos/CardiacMotion")
sys.path.extend([CARDIAC_MOTION, os.path.join(CARDIAC_MOTION, "utils")])
os.chdir(CARDIAC_MOTION)

# Import custom modules
from data.DataModules import CardiacMeshPopulationDM, CardiacMeshPopulationDataset
from utils.CardioMesh.CardiacMesh import Cardiac3DMesh, transform_mesh
from utils.image_helpers import generate_gif, merge_gifs_horizontally
from utils.run_helpers import Run, get_model

# MLflow setup
mlflow_uri = os.path.join(os.environ['HOME'], "01_repos/CardiacMotion/mlruns/")
mlflow.set_tracking_uri(mlflow_uri)

# Load and filter runs
runs_df = mlflow.search_runs(experiment_ids=['4'])
if runs_df.empty:
    raise ValueError(f"No runs found under URI {mlflow_uri} and experiment {experiment_ids}.")

runs_df = runs_df[runs_df["metrics.test_recon_loss"] < 3]
runs_df = runs_df.set_index(["experiment_id", "run_id"], drop=False)

# Adjust artifact URIs
runs_df.artifact_uri = runs_df.artifact_uri.str.replace(
    "/home/rodrigo/CISTIB/repos/", "/mnt/data/workshop/workshop-user1/output/"
).str.replace(
    "/home/home01/scrb/01_repos/", "/mnt/data/workshop/workshop-user1/output/"
).str.replace("/1/", "/3/").str.replace("/user/", "/rodrigo/")

# Define partitions
partitions = {
    "left_atrium": ("LA", "MVP", "PV1", "PV2", "PV3", "PV4", "PV5"),
    "right_atrium": ("RA", "TVP", "PV6", "PV7"),
    "left_ventricle": ("LV", "AVP", "MVP"),
    "right_ventricle": ("RV", "PVP", "TVP"),
    "biventricle": ("LV", "AVP", "MVP", "RV", "PVP", "TVP"),
    "aorta": ("aorta",)
}

# Paths configuration
class Paths:
    PARTITION = "left_ventricle"
    FACES_FILE = os.path.join(CARDIAC_MOTION, "utils/CardioMesh/data/faces_and_downsampling_mtx_frac_0.1_LV.pkl")
    MEAN_ACROSS_CYCLE_FILE = os.path.join(CARDIAC_MOTION, f"utils/CardioMesh/data/cached/mean_shape_time_avg__{PARTITION}.npy")
    PROCRUSTES_FILE = os.path.join(CARDIAC_MOTION, f"utils/CardioMesh/data/cached/procrustes_transforms_{PARTITION}.pkl")
    SUBSETTING_MATRIX_FILE = os.path.join(CARDIAC_MOTION, f"utils/CardioMesh/data/cached/subsetting_matrix_{PARTITION}.pkl")
    MESHES_PATH = os.path.join(os.environ['HOME'], "01_repos/CardiacMotion/data/cardio/meshes")

# Function to generate synthetic shapes
def generate_synthetic_shape(run, z_var, value, resolution=50):
    z_df = run.get_z_df()
    z_mean = z_df.mean()
    z_std = z_df.std()

    z = z_mean + value * np.diag(z_std)[z_var]
    z = torch.Tensor(z).unsqueeze(0)
    z = EasyDict({"mu": z, "log_var": None})
    
    output_mesh = run.model.decoder(z)[1][0].detach().numpy()
    return output_mesh

# Main execution
z_vars = list(range(8, 16))
z_values = [-3, 3]

fhm_mesh = Cardiac3DMesh(
    filename=os.path.join(os.environ['HOME'], "doctorado/data/meshes/Results/1000511/models/FHM_res_0.1_time001.npy"),
    faces_filename=os.path.join(os.environ['HOME'], "01_repos/CardioMesh/data/faces_fhm_10pct_decimation.csv"),
    subpart_id_filename=os.path.join(os.environ['HOME'], "01_repos/CardioMesh/data/subpartIDs_FHM_10pct.txt")
)

previous_run, previous_expid = None, None
filenames = []

for (exp_id, run_id), z_var, z_value in product(runs_df.index, z_vars, z_values):
    if run_id != previous_run:
        run = Run(run_id=run_id, exp_id=exp_id)
        model_weights = run.load_weights()
        
        try:
            t_ae = get_model(polynomial_degree=10)
            t_ae.load_state_dict(model_weights, strict=False)
        except:
            t_ae = get_model(polynomial_degree=12)
            t_ae.load_state_dict(model_weights, strict=False)
        run.model = t_ae
        
    if exp_id != previous_expid:
        mean_shape = np.load(run._MEAN_ACROSS_CYCLE_FILE)
        faces = fhm_mesh[partitions[run._PARTITION]].f

    previous_expid, previous_run = exp_id, run_id
    
    try:
        filename = f"./{run.run_id}_z{str(z_var).zfill(3)}_{z_value}.gif"
        output_mesh = generate_synthetic_shape(run, z_var, z_value)
        generate_gif(output_mesh, faces, filename, camera_position="xz")
    except FileNotFoundError as e:
        print(e)
        continue
        
    filenames.append(filename)
    
    if z_value == z_values[-1]:
        ofilename = f"{run.run_id}_z{str(z_var).zfill(3)}.gif"
        merge_gifs_horizontally(*filenames, ofilename)
        filenames = []
