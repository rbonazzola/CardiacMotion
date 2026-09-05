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
HOME = os.environ["HOME"]
CARDIAC_GWAS_REPO = f"{HOME}/01_repos/CardiacGWAS"
CARDIAC_COMA_REPO = f"{HOME}/01_repos/CardiacCOMA"
CARDIAC_MOTION_REPO = f"{HOME}/01_repos/CardiacMotion"
MLRUNS_DIR = f"{CARDIAC_MOTION_REPO}/mlruns"
os.chdir(CARDIAC_MOTION_REPO)

from easydict import EasyDict

import re
import glob

import mlflow
from mlflow.tracking import MlflowClient

import torch
import torch.nn.functional as F

import ipywidgets as widgets
from ipywidgets import interact, interactive, fixed, interact_manual
from IPython.display import Image
from IPython import embed

import numpy as np
import pandas as pd
import shlex
from subprocess import check_output

import pickle as pkl
import pytorch_lightning as ptl

from argparse import Namespace
import matplotlib.pyplot as plt

from copy import copy, deepcopy
from pprint import pprint
from tqdm import tqdm

sys.path.insert(0, '..')

from config.cli_args import overwrite_config_items
import pyvista as pv
# from utils.mlflow_helpers import get_model_pretrained_weights


# %%
from utils.CardioMesh.CardiacMesh import Cardiac3DMesh, transform_mesh

from models.Model3D import Autoencoder3DMesh
from models.lightning.ComaLightningModule import CoMA_Lightning
from config.load_config import load_yaml_config, to_dict

from scipy.linalg import orthogonal_procrustes
from typing import Dict, List
from IPython import embed
import logging

from trimesh import Trimesh
import ipywidgets as widgets
from ipywidgets import interact


# %%
def mse(s1, s2=None):
    if s2 is None:
        s2 = torch.zeros_like(s1)
    return ((s1-s2)**2).sum(-1).mean(-1)

def get_3d_mesh(ids, root_folder):
    
    for id in ids:
        npy_file = f"{root_folder}/{id}/models/fhm_time001.npy"
        pc  = np.load(npy_file)
        yield id, pc
        

def get_4d_mesh(ids, root_folder, timepoints=list(range(1,51))):
    
    for id in ids:
        for t in timepoints:
            npy_file = f"{root_folder}/{id}/models/fhm_time{str(t).zfill(3)}.npy"
            pc  = np.load(npy_file)
        yield id, pc

        
def generalisedProcrustes(point_clouds: np.array, ids: List, template_mesh=None, scaling=False, logger=logging.getLogger()):


    logger.info("Performing Procrustes analysis with scaling")
    if template_mesh is None:
        template_mesh = point_clouds[0]

    old_disparity, disparity = 0, 1  # random values
    it_count = 0
    
    transforms = {}            

    centroids = point_clouds.mean(axis=1)
    for i, id in enumerate(ids):
        point_clouds[i] -= centroids[i] 
        transforms[id] = {}
        transforms[id]["traslation"] = centroids[i]


    while abs(old_disparity - disparity) / disparity > 1e-2 and disparity:

        old_disparity = disparity
        disparity = []

        for i, id in enumerate(ids):

            # Docs: https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.procrustes.html
            if scaling:
                mtx1, mtx2, _disparity = procrustes(template_mesh, point_clouds[i])
                point_clouds[i] = np.array(mtx2)  # if self.procrustes_scaling else np.array(mtx1)

            else:
                # Docs: https://docs.scipy.org/doc/scipy/reference/generated/scipy.linalg.orthogonal_procrustes.html
                # Note that the arguments are swapped with respect to the previous @procrustes function
                R, s = orthogonal_procrustes(point_clouds[i], template_mesh)
                # Rotate
                point_clouds[i] = np.dot(point_clouds[i], R)  # * s
                # Mean point-wise MSE
                _disparity = mse(point_clouds[i], template_mesh) 
                disparity.append(_disparity)

                if it_count == 0:
                    transforms[id]["rotation"] = R #, "scaling": s}
                else:
                    transforms[id]["rotation"] = R.dot(transforms[id]["rotation"]) #, "scaling": transforms[i]["scaling"] * s}

        template_mesh = point_clouds.mean(axis=0)
        disparity = np.array(disparity).mean(axis=0)
        it_count += 1
        
    #self.procrustes_aligned = True
    logger.info(
        "Generalized Procrustes analysis with scaling performed after %s iterations"
        % it_count
    )

    return transforms

# %%
# PROCRUSTES_ORIGINAL_FILE = f"{CARDIAC_COMA_REPO}/data/procrustes_transforms_35k.pkl"
# MESHES_ORIGINAL_FILE = f"{CARDIAC_COMA_REPO}/data/LV_meshes_at_ED_35k.pkl"
# 
# MESHES_REPLICATION_FILE = f"{CARDIAC_COMA_REPO}/data/LV_meshes_at_ED_replication_25k.pkl"
#     
# procrustes_original = pkl.load(open(PROCRUSTES_ORIGINAL_FILE, "rb"))
# meshes_original = pkl.load(open(MESHES_ORIGINAL_FILE, "rb"))
# meshes_original_aligned = np.array([transform_mesh(meshes_original[id], **procrustes_original[id]) for id in meshes_original])
# 
# template_mesh = meshes_original_aligned.mean(axis=0)
# 
# PROCRUSTES_REPLICATION_FILE = f"{CARDIAC_COMA_REPO}/data/procrustes_transforms_LV_25k_new_meshes.pkl"
# procrustes_replication = pkl.load(open(PROCRUSTES_REPLICATION_FILE, "rb"))
# 
# procrustes_all = { k:v for k,v in procrustes_replication.items() if k not in procrustes_original }
# procrustes_all.update(procrustes_original)
# procrustes_all

# %% [markdown]
# # Generate reference shapes for each partition

# %% [markdown]
# 1. Generate reference shape for FHM mesh.
# 2. Read subpart IDs for each part. Transform it into a downsampling matrix.
# 3. For each FHM point cloud, extract the partitions.

# %%
## ALREADY CACHED

# MESHES_FHM_FILE = "/home/user/01_repos/CardiacCOMA/data/FHM_meshes_at_ED_all_my_segmentation_61225.pkl"
# meshes = pkl.load(open(MESHES_FHM_FILE, "rb"))
# 
PROCRUSTES_FILE = "utils/CardioMesh/data/procrustes_transforms_FHM_61k.pkl"
procrustes_transforms = pkl.load(open(PROCRUSTES_FILE, "rb"))

# meshes_original_aligned = []
# for id in tqdm(procrustes_transforms):
#    meshes_original_aligned.append(transform_mesh(meshes[id], **procrustes_transforms[id]))

# MESHES_ALIGNED_FHM_FILE = "/home/user/01_repos/CardiacCOMA/data/FHM_meshes_at_ED_all_my_segmentation_61225_aligned.pkl"
# meshes_original_aligned = np.array(meshes_original_aligned)
# pkl.dump(meshes_original_aligned, open(MESHES_ALIGNED_FHM_FILE, "wb"))

# %%
MESHES_ALIGNED_FHM_FILE = "/home/user/01_repos/CardiacCOMA/data/FHM_meshes_at_ED_all_my_segmentation_61225_aligned.pkl"
meshes_original_aligned = pkl.load(open(MESHES_ALIGNED_FHM_FILE, "rb"))

# %%
ID = "1000511"

fhm_mesh = Cardiac3DMesh(
    filename=f"/mnt/data/workshop/workshop-user1/datasets/meshes/Results_Yan/{ID}/models/FHM_res_0.1_time001.npy",
    faces_filename="/home/user/01_repos/CardioMesh/data/faces_fhm_10pct_decimation.csv",
    subpart_id_filename="/home/user/01_repos/CardioMesh/data/subpartIDs_FHM_10pct.txt"
)

fhm_mesh.points = transform_mesh(fhm_mesh.points, **procrustes_transforms["1000511"])

# %% [markdown]
# ### Subsetting matrices
# Generadas por `MeshPartitioning.py`. Cargar desde caché:

# %%
partitions = {
  "left_atrium" : ("LA", "MVP", "PV1", "PV2", "PV3", "PV4", "PV5"),
  "right_atrium" : ("RA", "TVP", "PV6", "PV7"),
  "left_ventricle" : ("LV", "AVP", "MVP"),
  "right_ventricle" : ("RV", "PVP", "TVP"),
  "biventricle" : ("LV", "AVP", "MVP", "RV", "PVP", "TVP"),
  "aorta" : ("aorta",)
}

PARTITION = "left_ventricle"
subsetting_mtx_file = f"/home/user/01_repos/CardioMesh/data/cached/subsetting_matrix_{PARTITION}.pkl"
subsetting_mtx = pkl.load(open(subsetting_mtx_file, "rb"))

# %%
# Procrustes transforms computados en compute_temporal_mean.py. Cargar meshes para visualización:
MESHES_FHM_FILE = "/home/user/01_repos/CardiacCOMA/data/FHM_meshes_at_ED_all_my_segmentation_61225.pkl"
fhm_meshes = pkl.load(open(MESHES_FHM_FILE, "rb"))
ids = list(fhm_meshes.keys())
meshes = list(fhm_meshes.values())

# %% [markdown]
# ___

# %%
# sphere = vedo.Sphere(res=params["mesh_resolution"]).to_trimesh()
# conn = sphere.faces # connectivity
# conn = np.c_[np.ones(conn.shape[0]) * 3, conn].astype(int)  # add column of 3, as required by PyVista

import random

pv.set_plot_theme("document")

procrustes_transforms = pkl.load(open(f"/home/user/01_repos/CardioMesh/data/cached/procrustes_transforms_{partition}.pkl", "rb"))
# faces, _ = pkl.load(open("faces_and_downsampling_mtx_frac_0.1_LV.pkl", "rb")).values()
faces = fhm_mesh[partitions[partition]].f
faces = np.c_[np.ones(faces.shape[0]) * 3, faces].astype(int)

color_palette = list(pv.colors.color_names.values())
random.shuffle(color_palette)

subsetting_mtx = pkl.load(open(subsetting_mtx_file, "rb"))

# %%
# Trimesh(mesh, faces[:,1:4]).show()

# %%
pl = pv.Plotter(notebook=True, off_screen=False, polygon_smoothing=False)

for i, id in enumerate(ids[:20]):
    mesh = meshes[i]
    mesh = subsetting_mtx * mesh
    mesh = transform_mesh(mesh, **procrustes_transforms[id])    
    # mesh = pv.PolyData(mesh, faces)
    pl.add_mesh(mesh, show_edges=False, point_size=1.5, color=color_palette[i], opacity=0.5)

    pl.show(interactive=True, interactive_update=True)    


# %%
def f(ids, rotated, traslated):
                
    pl = pv.Plotter(notebook=True, off_screen=False, polygon_smoothing=False)
    
    for i, id in enumerate(ids):
      
      mesh = meshes[i]
      mesh = subsetting_mtx * mesh
      if rotated and not traslated:
        mesh = transform_mesh(mesh, rotation=procrustes_transforms[id]["rotation"])
      elif traslated and not rotated:
        mesh = transform_mesh(mesh, traslation=procrustes_transforms[id]["traslation"])
      elif traslated and rotated:
        mesh = transform_mesh(mesh, **procrustes_transforms[id])
      mesh = pv.PolyData(mesh, faces)
        
      pl.add_mesh(mesh, show_edges=False, point_size=1.5, color=color_palette[i], opacity=0.5)
    pl.show(interactive=True, interactive_update=True)
    
#interact(f, i=widgets.IntSlider(min=1,max=N))
interact(f, 
  ids=widgets.SelectMultiple(options=ids[:20]),
  rotated=widgets.ToggleButton(),
  traslated=widgets.ToggleButton()
);

# %% [markdown]
# ___

# %% [markdown]
# Test transforms

# %%
transforms = pkl.load(open("/home/user/01_repos/CardioMesh/data/cached/procrustes_transforms_biventricle.pkl", "rb"))

# %%
subsetting_mtx_file = f"/home/user/01_repos/CardioMesh/data/cached/subsetting_matrix_biventricle.pkl"
subsetting_mtx = pkl.load(open(subsetting_mtx_file, "rb"))

# %%
ID = '2653145'

# %%
subsetted_mesh = subsetting_mtx * fhm_meshes[ID]

# %% [markdown]
# ___

# %%
faces = fhm_mesh[partitions[PARTITION]].f
i = 13
Trimesh(subsetted_meshes_aligned[:,i,:], faces).show()

# %% [markdown]
# ___

# %%
extract_partition_and_downsample = lambda shape: subsetting_mtx * shape.reshape(-1, 3)


# %%
@interact
# def show_partition(partition=widgets.Select(options=partitions, value=('LV'), 'AVP', 'MVP'))):
def show_partition(partition=widgets.Select(options=partitions, value=partitions["left_ventricle"])):
    
    print(partition)
    partition_mesh = fhm_mesh[partition]
    return Trimesh(partition_mesh.v, partition_mesh.f).show()


# %% [markdown]
# ___

# %% [markdown]
# ### Medias temporales poblacionales
# Computadas en `compute_temporal_mean.py`.
