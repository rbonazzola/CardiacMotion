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

try:
    os.environ["CARDIAC_MOTION_REPO"] = f"{os.environ['HOME']}/01_repos/CardiacMotion"
    repo_dir = os.environ.get("CARDIAC_MOTION_REPO")
    os.chdir(repo_dir)
except FileNotFoundError:
    os.environ["HOME"] = "/root"
    os.environ["CARDIAC_MOTION_REPO"] = f"{os.environ['HOME']}/01_repos/CardiacMotion"
    repo_dir = os.environ.get("CARDIAC_MOTION_REPO")
    os.chdir(repo_dir)

import torch
from torch import Tensor

import os, sys
import glob
import re
import pickle as pkl

import numpy as np

from torch.utils.data import TensorDataset, DataLoader, random_split
from typing import Union, Dict, List, Optional

import pytorch_lightning as pl
from utils.CardioMesh.CardiacMesh import transform_mesh

from easydict import EasyDict
from tqdm import tqdm

import ipywidgets as widgets
from ipywidgets import interact

from main_autoencoder_cardiac import CardiacMeshPopulationDataset, CardiacMeshPopulationDM

# %% [markdown]
# `CardioMesh.CardiacMeshPopulation.py`
#
# Example of usage:
#     
# ```
# mesh_population = CardiacMeshPopulation(
#   root_path = "data/cardio/Results",
#   N_subj = None
# )
#
# ```

# %% [markdown]
# `(ID, phase) -> path`

# %% [markdown]
# I am assuming LV is being used:

# %%
root_path = "data/cardio/Results/"

# %% [markdown]
# # Compute Procrustes transforms for additional meshes

# %%
from scipy.linalg import orthogonal_procrustes
from IPython import embed
import logging

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

        for i, id in tqdm(enumerate(ids)):

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
PROCRUSTES_FILE = "utils/CardioMesh/data/procrustes_transforms_FHM_35k.pkl"
procrustes_transforms = pkl.load(open(PROCRUSTES_FILE, "rb"))

# %%
# import re
# paths = glob.glob("/mnt/data/workshop/workshop-user1/datasets/meshes/Results_*/*/models/FHM_res_0.1_time001.npy")
# regex = "/mnt/data/workshop/workshop-user1/datasets/meshes/Results_.*/(.*)/models/FHM_res_0.1_time001.npy"
# regex = re.compile(regex)
# meshes_fhmed = {regex.match(path).group(1): np.load(path) for path in paths}
# pkl.dump(meshes_fhmed, open("/home/user/01_repos/CardiacCOMA/data/FHM_meshes_at_ED_all_my_segmentation_61225.pkl", "wb"))

# %%
MESHES_FHM_FILE = "/home/user/01_repos/CardiacCOMA/data/FHM_meshes_at_ED_all_my_segmentation_61225.pkl"
meshes = pkl.load(open(MESHES_FHM_FILE, "rb"))

# %%
common_ids = set(meshes.keys()).intersection(set(procrustes_transforms.keys()))
procrustes_transforms = {k:procrustes_transforms[k] for k in common_ids}

# %%
meshes_original_aligned = [transform_mesh(meshes[id], **procrustes_transforms[id]) for id in procrustes_transforms]

# %%
meshes_original_aligned = np.array(meshes_original_aligned)

# %%
template_mesh = meshes_original_aligned.mean(axis=0)

# %%
template_mesh.shape

# %%
meshes_npy = np.array(list(meshes.values()))

# %%
ids = list(meshes.keys())

# %%
procrustes_transforms = generalisedProcrustes(
    point_clouds=meshes_npy,
    ids=ids,
    template_mesh=template_mesh
)

PROCRUSTES_FHM_FULL = "utils/VTKHelpers/data/procrustes_transforms_FHM_61k.pkl"
pkl.dump(procrustes_transforms, open(PROCRUSTES_FHM_FULL, "wb"))

# %%
faces = EasyDict(
    pkl.load(open("utils/VTKHelpers/data/faces_and_downsampling_mtx_frac_0.1_LV.pkl", "rb"))
).new_faces

template = EasyDict({
   "v": transform_mesh(np.load(
       f"{root_path}/1000215/models/FHM_time001.npy"), 
       **procrustes_transforms["1000215"]
   ),
   "f": faces
})

# %% jupyter={"outputs_hidden": true}
cardiac_dataset = CardiacMeshPopulationDataset(
    root_path, 
    procrustes_transforms="utils/VTKHelpers/data/procrustes_transforms_FHM_35k.pkl",
    faces=faces,
    template_mesh=template
)

# %% [markdown]
# ### Compute mean across timepoints and across subjects

# %%
pp = []
for i, k in enumerate(cardiac_dataset):
    print(i)
    pp.append(k["time_avg_s"])    

# %%
s_popmean = torch.stack(pp).mean(0).numpy()
POPMEAN_SHAPE = "data/LV_shape_mean_across_timepoints.npy"
np.save(POPMEAN_SHAPE, s_popmean)

# %%
t_avg_s = [ k["time_avg_s"] in cardiac_dataset ]

# %%
cardiac_mesh_dm = CardiacMeshPopulationDM(cardiac_dataset, batch_size=32)
cardiac_mesh_dm.setup()
len(cardiac_dataset)

# %%
subj_idx_w = widgets.IntSlider(min=1, max=len(cardiac_dataset))

def generate_gif(mesh4D, faces, filename, camera_position='xy', show_edges=False, **kwargs):
        
        '''
        Produces a gif file representing the motion of the input mesh.
        
        params:
          ::mesh4D:: a sequence of Trimesh mesh objects.
          ::faces:: array of F x 3 containing the indices of the mesh's triangular faces.
          ::filename:: the name of the output gif file.
          ::camera_position:: camera position for pyvista plotter (check relevant docs)
          
        return:
          None, only produces the gif file.
        '''

        import pyvista as pv
        
        connectivity = np.c_[np.ones(faces.shape[0]) * 3, faces].astype(int)
                
        pv.set_plot_theme("document")
        os.makedirs(os.path.dirname("./"+filename) , exist_ok=True)
        
        # plotter = pv.Plotter(shape=(1, len(camera_positions)), notebook=False, off_screen=True)
        pv.start_xvfb()
        plotter = pv.Plotter(notebook=False, off_screen=True)
            
        # Open a gif
        plotter.open_gif(filename)

        try:
            # if mesh3D is torch.Tensor, this your should run OK
            mesh4D = mesh4D.cpu().numpy()[0].astype("float32")
        except AttributeError:
            pass

        kk = pv.PolyData(mesh4D[0], connectivity)
        # plotter.add_mesh(kk, smooth_shading=True, opacity=0.5 )#, show_edges=True)
        plotter.add_mesh(kk, show_edges=show_edges, opacity=0.5, color="red") 
        
        for t, _ in enumerate(mesh4D):
            kk = pv.PolyData(mesh4D[t], connectivity)
            plotter.camera_position = camera_position
            plotter.update_coordinates(kk.points, render=False)
            plotter.render()             
            plotter.write_frame()
        
        plotter.close()
        
        return filename

@interact
def show_gif(subj_idx=subj_idx_w):
    
    from IPython.display import HTML
    import base64
        
    # subj_id = cardiac_dataset.ids[subj_id]                
    
    gifpath = generate_gif(
        torch.stack(pp).mean(0), #cardiac_dataset[subj_idx], 
        faces, filename="kk.gif", camera_position="xz"
    )
    
    b64 = base64.b64encode(
        open(gifpath,'rb').read()
    ).decode('ascii')
    
    display(HTML(f'<img src="data:image/gif;base64,{b64}" />'))
