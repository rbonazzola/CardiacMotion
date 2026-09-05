# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.17.3
#   kernelspec:
#     display_name: cardiac_motion
#     language: python
#     name: python3
# ---

# %%
import os
import numpy as np
import pickle as pkl

# %%
PROCRUSTES_FILE = "/home/rodrigo/01_repos/CardioMesh/data/cached/procrustes_transforms_biventricle.pkl"
procrustes_transforms = pkl.load(open(PROCRUSTES_FILE, "rb"))

# %%
procrustes_transforms

# %%
MESHES_FOLDER = "/home/rodrigo/01_repos/CardiacSegmentation/FHM/"
os.listdir()
