# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.17.3
#   kernelspec:
#     display_name: cardio
#     language: python
#     name: cardio
# ---

# %% [markdown]
# # Mesh Decimation
#
# Precomputes a **fused sparse transform** that maps PDM b-values directly to
# a decimated mesh, without ever materialising the full-resolution intermediate.
#
# Pipeline:
# ```
# b  →  shape_full = mean_ + b @ components_   (PDM, linear)
#     →  shape_part = S @ shape_full            (partition subsetting, sparse)
#     →  shape_dec  = D @ shape_part            (QSlim decimation, sparse)
# ```
# All three steps are linear, so we precompute:
# ```
# P_3d   = kron(D @ S, I_3)
# mean_dec  = P_3d @ mean_
# modes_dec = (P_3d @ components_.T).T
# ```
# Runtime (per timeframe): `shape_dec = mean_dec + b @ modes_dec`
#
# In addition, the graph-CNN hierarchy matrices (D/U/A) are computed once
# from the decimated mesh topology.

# %% [markdown]
# ## Configuration

# %%
import os
import pickle as pkl
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import scipy.sparse as sp
from scipy.sparse import kron, eye as speye
import joblib

# --- paths ---
PDM_FILE           = os.path.expanduser("~/repos/CardiacSegmentation/PDM/fhm_pdm.joblib")
MEAN_SHAPE_VTK     = os.path.expanduser("~/repos/CardiacSegmentation/PDM/mean_shape.vtk")

# CardioMesh stores the QSlim decimation matrices
CARDIO_MESH_DATA   = "/mnt/data/01_repos/CardioMesh/data"
DOWNSAMPLING_FILES = {
    "FHM": os.path.join(CARDIO_MESH_DATA, "faces_and_downsampling_mtx_frac_0.1_full_heart.pkl"),
    "LV":  os.path.join(CARDIO_MESH_DATA, "faces_and_downsampling_mtx_frac_0.1_LV.pkl"),
}

OUTPUT_DIR = "/mnt/data/01_repos/CardiacMotion/data/cached/matrices"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Which partition to process
PARTITION = "FHM"   # "FHM" | "LV" | ...

# Decimation factors for the graph-CNN hierarchy (applied to the already-decimated mesh)
HIERARCHY_FACTORS = [4, 4, 4, 4]

# %% [markdown]
# ## 1. Load PDM

# %%
pdm = joblib.load(PDM_FILE)

n_verts_full = pdm.n_features_in_ // 3
n_modes      = pdm.n_components_

print(f"PDM  — vertices: {n_verts_full:,}  modes: {n_modes}")
print(f"mean_       shape: {pdm.mean_.shape}")
print(f"components_ shape: {pdm.components_.shape}")

# %% [markdown]
# ## 2. Load decimation matrix

# %%
dd = pkl.load(open(DOWNSAMPLING_FILES[PARTITION], "rb"))
D_mtx     = dd["mtx"]        # (n_dec, n_full) sparse
new_faces = dd["new_faces"]  # (F, 3)

n_verts_dec = D_mtx.shape[0]
print(f"Downsampling  — {D_mtx.shape[1]:,} → {n_verts_dec:,} vertices")
print(f"Decimated faces: {new_faces.shape}")
print(f"Sparsity: {D_mtx.nnz / (D_mtx.shape[0] * D_mtx.shape[1]):.2e}")

# %% [markdown]
# ## 3. Build subsetting matrix S
#
# For FHM the subsetting matrix is the identity (all vertices are kept).
# For a sub-partition (LV, RV, …) we select the relevant vertex indices from
# the mean-shape VTK using the `subpartID` attribute of `Cardiac3DMesh`.

# %%
if PARTITION == "FHM":
    # S is the identity — P = D
    S = speye(n_verts_full, format="csc")
else:
    from cardio_mesh import Cardiac3DMesh
    mesh_full = Cardiac3DMesh(MEAN_SHAPE_VTK)
    col_ind   = [i for i, x in enumerate(mesh_full.subpartID) if x == PARTITION]
    row_ind   = list(range(len(col_ind)))
    S = sp.csc_matrix(
        (np.ones(len(col_ind)), (row_ind, col_ind)),
        shape=(len(col_ind), n_verts_full)
    )
    print(f"Subsetting: {n_verts_full:,} → {S.shape[0]:,} vertices")

# Combined spatial transform: (n_dec, n_full_or_part)
P = (D_mtx @ S).tocsc()

# Extend to 3D with Kronecker product
# kron(P, I_3) maps a flattened (x1,y1,z1,x2,…) vector of size n_full*3
# to size n_dec*3, applying P to each coordinate independently.
P_3d = kron(P, speye(3), format="csc")
print(f"\nP      shape: {P.shape}")
print(f"P_3d   shape: {P_3d.shape}")

# %% [markdown]
# ## 4. Transform PDM basis
#
# `mean_dec`  replaces `pdm.mean_`  in the decimated space.
# `modes_dec` replaces `pdm.components_` in the decimated space.
#
# Reconstruction: `shape_dec = (mean_dec + b @ modes_dec).reshape(n_dec, 3)`

# %%
print("Computing mean_dec …")
mean_dec = P_3d @ pdm.mean_          # (n_dec*3,)

print("Computing modes_dec …")
# pdm.components_.T  : (n_full*3, n_modes)
# P_3d @ …          : (n_dec*3,  n_modes)  — sparse × dense
modes_dec = (P_3d @ pdm.components_.T).T   # (n_modes, n_dec*3)

print(f"mean_dec  shape: {mean_dec.shape}")
print(f"modes_dec shape: {modes_dec.shape}")

# %% [markdown]
# ## 5. Verification
#
# With `b = 0` the transformed PDM should reproduce the decimated mean shape.
# Compare against directly applying the decimation matrix to `pdm.mean_`.

# %%
shape_via_pdm = mean_dec.reshape(n_verts_dec, 3)
shape_direct  = (P @ pdm.mean_.reshape(n_verts_full, 3))

max_err = np.abs(shape_via_pdm - shape_direct).max()
print(f"Max reconstruction error (b=0): {max_err:.2e}  {'✓' if max_err < 1e-6 else '✗'}")

# Non-zero b: pick a random unit vector in mode space
rng = np.random.default_rng(42)
b_test = rng.standard_normal(n_modes)

shape_fused  = (mean_dec + b_test @ modes_dec).reshape(n_verts_dec, 3)
shape_twostep = (P @ (pdm.mean_ + b_test @ pdm.components_).reshape(n_verts_full, 3))

max_err_b = np.abs(shape_fused - shape_twostep).max()
print(f"Max reconstruction error (random b): {max_err_b:.2e}  {'✓' if max_err_b < 1e-6 else '✗'}")

# %% [markdown]
# ## 6. Graph-CNN hierarchy matrices
#
# Starting from the already-decimated mesh, compute the multi-resolution
# downsampling (D), upsampling (U), and adjacency (A) matrices used by CoMA.

# %%
import sys
sys.path.insert(0, "/mnt/data/01_repos/CardiacMotion/cardiac_motion")
from utils.mesh_operations import generate_transform_matrices, Mesh

dec_mesh = Mesh(v=mean_dec.reshape(n_verts_dec, 3), f=new_faces)
print(f"Starting mesh: {dec_mesh.v.shape[0]:,} vertices, {dec_mesh.f.shape[0]:,} faces")

M, A, D_hier, U = generate_transform_matrices(dec_mesh, HIERARCHY_FACTORS)

print("\nHierarchy levels:")
for i, m in enumerate(M):
    print(f"  Level {i}: {m.v.shape[0]:,} vertices")

# %% [markdown]
# ## 7. Save

# %%
output_fname = os.path.join(OUTPUT_DIR, f"pdm_fused_{PARTITION.lower()}.pkl")

out = {
    # Fused PDM (decimated space)
    "mean_dec":   mean_dec,           # (n_dec*3,)
    "modes_dec":  modes_dec,          # (n_modes, n_dec*3)
    "P":          P,                  # sparse (n_dec, n_full) — kept for applying to existing meshes
    "new_faces":  new_faces,          # (F, 3) — topology of the decimated mesh
    "n_verts_dec": n_verts_dec,
    "n_modes":    n_modes,
    # Graph-CNN hierarchy
    "M": M, "A": A, "D": D_hier, "U": U,
}

pkl.dump(out, open(output_fname, "wb"))
print(f"Saved → {output_fname}")

# %% [markdown]
# ## 8. Usage in reconstruct_meshes.py
#
# Load the fused PKL once, then per subject/timeframe:
#
# ```python
# out = pkl.load(open("pdm_fused_fhm.pkl", "rb"))
# mean_dec, modes_dec, P = out["mean_dec"], out["modes_dec"], out["P"]
#
# # b_values: (n_modes,) array from the CSV
# shape_dec = (mean_dec + b_values @ modes_dec).reshape(-1, 3)
#
# # Apply subject-specific rigid body transform (still needed, per-subject)
# shape_transformed = transform_shape(shape_dec, ...)
#
# np.save(output_filename, shape_transformed)
# ```
#
# Alternatively, to decimate an **existing** full-resolution mesh `.npy`:
# ```python
# shape_full = np.load("fhm_full_time001.npy").reshape(-1, 3)   # (n_full, 3)
# shape_dec  = (P @ shape_full)                                   # (n_dec, 3)
# np.save("FHM_res_0.1_time001.npy", shape_dec)
# ```
