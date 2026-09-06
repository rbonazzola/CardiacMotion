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

# %% [markdown]
# # Compress 4D cardiac meshes into PCs + translations (+ optional Procrustes rotations)
#
# Runs the full pipeline for each cardiac chamber independently.
#
# **Pipeline per (chamber, subject, timepoint):**
# 1. Subsetting: `x_sub = S @ x`  ->  (N_sub, 3)
# 2. Centroid:   `t = mean(x_sub, axis=0)`  ->  stored as translation
# 3. Center:     `x_c = x_sub - t`
# 4. PCA:        `z = x_c.flatten() @ V.T`  ->  stored as PCA code
#
# Optionally, per subject:
# 5. Procrustes: orthogonal rotation `R` aligning the subject's temporal-mean shape
#    to the population mean  ->  stored separately
#
# **GPU acceleration (PyTorch + CUDA):**
# - Subsetting via torch.sparse matrix multiplication
# - PCA via streaming covariance accumulation on GPU -> single torch.linalg.eigh
# - Reconstruction-error curve via analytic formula (no loop over k)
# - Encoding via batched X @ V.T on GPU
#
# **Output layout:**
#   OUTPUT_DIR/
#     {chamber}/
#       pca.npz               -- V (k x D), mu (D,), eigenvalues (k,)
#       subsetting_matrix.npy -- copy of S used
#       meta.pkl              -- config + rms_errors curve + chosen k
#       {subject_id}.npz      -- z (T x k), t (T x 3), [R (3 x 3)]

# %%
import os
import re
import glob
import pickle as pkl

import numpy as np
import torch
from scipy.linalg import orthogonal_procrustes
from tqdm.auto import tqdm
import matplotlib.pyplot as plt

import cardio_mesh

# %% [markdown]
# ## Global configuration

# %%
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {DEVICE}")
if DEVICE == "cuda":
    print(f"  {torch.cuda.get_device_name(0)}  --  "
          f"{torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

MESHES_DIR         = cardio_mesh.MESHES_DIR
CARDIO_MESH_CACHE  = "/mnt/data/01_repos/CardioMesh/data/cached"
OUTPUT_BASE_DIR    = "/mnt/data/01_repos/CardiacMotion/data/compressed"

N_TIMEPOINTS       = 50
TRAIN_FRACTION     = 0.8
MAX_COMPONENTS     = 100
ERROR_THRESHOLD_MM = 0.2
COMPUTE_PROCRUSTES = True

# One subsetting matrix file per chamber (all from CardioMesh cache)
CHAMBERS = {
    "LV":          os.path.join(CARDIO_MESH_CACHE, "subsetting_matrix_LV.npy"),
    "RV":          os.path.join(CARDIO_MESH_CACHE, "subsetting_matrix_RV.npy"),
    "LA":          os.path.join(CARDIO_MESH_CACHE, "subsetting_matrix_LA.npy"),
    "RA":          os.path.join(CARDIO_MESH_CACHE, "subsetting_matrix_RA.npy"),
    "biventricle": os.path.join(CARDIO_MESH_CACHE, "subsetting_matrix_biventricle.npy"),
    "aorta":       os.path.join(CARDIO_MESH_CACHE, "subsetting_matrix_aorta.npy"),
}

# Covariance matrix for biventricle is ~3.5 GB float32; falls back to CPU if needed
MAX_COV_GB_GPU = 4.0

# %% [markdown]
# ## Discover subjects with exactly N_TIMEPOINTS frames (done once)

# %%
_regex = re.compile(
    rf"{re.escape(MESHES_DIR)}/.*/models/FHM_res_0\.1_time0(\d\d)\.npy"
)

def get_subject_paths(root, n_timepoints=50):
    result = {}
    for sid in sorted(os.listdir(root)):
        paths = sorted(glob.glob(os.path.join(root, sid, "models", "FHM*.npy")))
        matched = [p for p in paths if _regex.match(p)]
        if len(matched) == n_timepoints:
            result[sid] = matched
    return result

print("Scanning subjects ...")
all_paths = get_subject_paths(MESHES_DIR, N_TIMEPOINTS)
all_ids   = list(all_paths.keys())
print(f"Subjects with {N_TIMEPOINTS} timepoints: {len(all_ids)}")

# %% [markdown]
# ## Train / validation split (same split for all chambers)

# %%
rng      = np.random.default_rng(42)
shuffled = rng.permutation(len(all_ids))
n_train  = int(TRAIN_FRACTION * len(all_ids))

train_ids = [all_ids[i] for i in shuffled[:n_train]]
val_ids   = [all_ids[i] for i in shuffled[n_train:]]
print(f"Train: {len(train_ids)}  |  Val: {len(val_ids)}")

# %% [markdown]
# ## Helper functions

# %%
def make_sparse_gpu(S_scipy, device):
    S_coo     = S_scipy.tocoo()
    indices   = torch.tensor(np.vstack([S_coo.row, S_coo.col]), dtype=torch.long)
    values    = torch.tensor(S_coo.data, dtype=torch.float32)
    return (
        torch.sparse_coo_tensor(indices, values, S_coo.shape)
        .to_sparse_csr()
        .to(device)
    )

def load_subject_gpu(paths, S, device=DEVICE):
    """
    Returns
    -------
    X_flat : (T, N_sub*3)  float32, centered
    T_cent : (T, 3)        float32, per-frame centroids
    """
    frames, centroids = [], []
    for p in paths:
        x     = torch.tensor(np.load(p, allow_pickle=True),
                              dtype=torch.float32, device=device)
        x_sub = torch.mm(S, x)
        t     = x_sub.mean(dim=0)
        frames.append((x_sub - t).reshape(-1))
        centroids.append(t)
    return torch.stack(frames), torch.stack(centroids)

def decode(z, t, V, mu, N_sub):
    """
    z   : (T, k)  numpy
    t   : (T, 3)  numpy
    Returns x_sub : (T, N_sub, 3)
    """
    x_c = (z @ V) + mu
    return x_c.reshape(len(z), N_sub, 3) + t[:, None, :]

# %% [markdown]
# ## Main loop: one PCA model per chamber

# %%
rms_curves = {}   # chamber -> rms_errors array, for the summary plot

for chamber, subsetting_file in CHAMBERS.items():

    print(f"\n{'='*60}")
    print(f"  Chamber: {chamber}")
    print(f"{'='*60}")

    out_dir = os.path.join(OUTPUT_BASE_DIR, chamber)
    os.makedirs(out_dir, exist_ok=True)

    # -- Load subsetting matrix --
    S_scipy       = np.load(subsetting_file, allow_pickle=True)
    N_sub, N_full = S_scipy.shape
    D             = N_sub * 3
    print(f"  {N_full} -> {N_sub} vertices  (D = {D})")

    S_sparse = make_sparse_gpu(S_scipy, DEVICE)

    # -- Decide where to accumulate covariance (GPU vs CPU) --
    cov_size_gb = D * D * 4 / 1e9
    cov_device  = DEVICE if cov_size_gb <= MAX_COV_GB_GPU else "cpu"
    print(f"  Covariance matrix: {cov_size_gb:.2f} GB  ->  accumulating on {cov_device}")

    # -- Pass 1: global mean --
    print("  Pass 1/2 -- global mean ...")
    running_sum = torch.zeros(D, device=DEVICE, dtype=torch.float64)
    n_total     = 0

    for sid in tqdm(train_ids, desc="  Mean", leave=False):
        X_flat, _ = load_subject_gpu(all_paths[sid], S_sparse)
        running_sum += X_flat.to(torch.float64).sum(0)
        n_total     += X_flat.shape[0]

    mu = (running_sum / n_total).to(torch.float32)

    # -- Pass 2: covariance --
    print("  Pass 2/2 -- covariance ...")
    C = torch.zeros(D, D, device=cov_device, dtype=torch.float32)

    for sid in tqdm(train_ids, desc="  Cov", leave=False):
        X_flat, _ = load_subject_gpu(all_paths[sid], S_sparse)
        X_c        = (X_flat - mu).to(cov_device)
        C         += X_c.T @ X_c

    C /= n_total - 1

    # -- Eigen-decomposition --
    print("  Eigen-decomposition ...")
    C_gpu = C.to(DEVICE)
    del C
    eigenvalues, eigenvectors = torch.linalg.eigh(C_gpu)
    del C_gpu

    idx         = torch.argsort(eigenvalues, descending=True)
    eigenvalues = eigenvalues[idx[:MAX_COMPONENTS]]
    V           = eigenvectors[:, idx[:MAX_COMPONENTS]].T.contiguous()
    del eigenvectors
    torch.cuda.empty_cache()

    print(f"  Top-3 explained variance: {eigenvalues[:3].cpu().numpy()}")

    # -- Analytic reconstruction-error curve --
    print("  Reconstruction-error curve (validation) ...")
    total_sq     = 0.0
    component_sq = torch.zeros(MAX_COMPONENTS, device=DEVICE, dtype=torch.float64)
    n_val_frames = 0

    for sid in tqdm(val_ids, desc="  Val", leave=False):
        X_flat, _ = load_subject_gpu(all_paths[sid], S_sparse)
        X_c        = X_flat - mu
        Z          = X_c @ V.T
        total_sq  += (X_c.to(torch.float64) ** 2).sum().item()
        component_sq += (Z.to(torch.float64) ** 2).sum(0)
        n_val_frames += X_flat.shape[0]

    cumvar     = torch.cumsum(component_sq, dim=0).cpu().numpy()
    unexplained = np.maximum(total_sq - cumvar, 0.0)
    rms_errors  = np.sqrt(unexplained / (n_val_frames * N_sub))
    rms_curves[chamber] = rms_errors

    # -- Choose k --
    below = np.where(rms_errors <= ERROR_THRESHOLD_MM)[0]
    if len(below) == 0:
        print(f"  WARNING: {ERROR_THRESHOLD_MM} mm threshold not reached. Using k = {MAX_COMPONENTS}.")
        k = MAX_COMPONENTS
    else:
        k = int(below[0]) + 1
        print(f"  Chosen k = {k}  (RMS = {rms_errors[k-1]:.3f} mm)")

    V_k     = V[:k]
    V_k_gpu = V_k.to(DEVICE)

    # -- Procrustes (optional) --
    if COMPUTE_PROCRUSTES:
        pop_mean = torch.zeros(N_sub, 3, device=DEVICE, dtype=torch.float64)
        n_proc   = 0
        for sid in tqdm(train_ids, desc="  Procrustes mean", leave=False):
            X_flat, _ = load_subject_gpu(all_paths[sid], S_sparse)
            pop_mean  += X_flat.mean(0).to(torch.float64).reshape(N_sub, 3)
            n_proc    += 1
        pop_mean_np = (pop_mean / n_proc).cpu().numpy().astype(np.float32)

    # -- Encode all subjects --
    print(f"  Encoding {len(all_ids)} subjects ...")
    for sid in tqdm(all_ids, desc="  Encode", leave=False):
        paths          = all_paths[sid]
        X_flat, T_cent = load_subject_gpu(paths, S_sparse)
        z = ((X_flat - mu) @ V_k_gpu.T).cpu().numpy().astype(np.float32)
        t = T_cent.cpu().numpy().astype(np.float32)

        out = dict(z=z, t=t)
        if COMPUTE_PROCRUSTES:
            subj_mean  = X_flat.mean(0).cpu().numpy().reshape(N_sub, 3)
            R, _       = orthogonal_procrustes(subj_mean, pop_mean_np)
            out["R"]   = R.astype(np.float32)

        np.savez_compressed(os.path.join(out_dir, sid), **out)

    # -- Save PCA model + metadata --
    np.savez_compressed(
        os.path.join(out_dir, "pca"),
        V=V_k.cpu().numpy().astype(np.float32),
        mu=mu.cpu().numpy().astype(np.float32),
        eigenvalues=eigenvalues[:k].cpu().numpy().astype(np.float32),
    )
    np.save(os.path.join(out_dir, "subsetting_matrix.npy"), S_scipy)
    with open(os.path.join(out_dir, "meta.pkl"), "wb") as f:
        pkl.dump(dict(
            chamber=chamber,
            k_chosen=k,
            max_components=MAX_COMPONENTS,
            n_timepoints=N_TIMEPOINTS,
            n_subjects_total=len(all_ids),
            n_subjects_train=len(train_ids),
            n_subjects_val=len(val_ids),
            error_threshold_mm=ERROR_THRESHOLD_MM,
            rms_errors=rms_errors,
            n_vertices_full=N_full,
            n_vertices_sub=N_sub,
            subsetting_file=subsetting_file,
            compute_procrustes=COMPUTE_PROCRUSTES,
        ), f)

    print(f"  Saved to {out_dir}")

    # -- Quick verification on one val subject --
    sid_ex  = val_ids[0]
    data_ex = np.load(os.path.join(out_dir, f"{sid_ex}.npz"))
    X_hat   = decode(data_ex["z"].astype(np.float64),
                     data_ex["t"].astype(np.float64),
                     V_k.cpu().numpy().astype(np.float64),
                     mu.cpu().numpy().astype(np.float64),
                     N_sub)
    X_flat_ex, T_ex = load_subject_gpu(all_paths[sid_ex], S_sparse)
    X_orig = X_flat_ex.cpu().numpy().reshape(-1, N_sub, 3) + T_ex.cpu().numpy()[:, None, :]
    err    = np.sqrt(((X_orig - X_hat) ** 2).sum(-1))
    floats_compressed = k * N_TIMEPOINTS + N_TIMEPOINTS * 3
    floats_raw        = N_full * N_TIMEPOINTS * 3
    print(f"  Verification ({sid_ex}): mean err = {err.mean():.3f} mm | "
          f"compression {floats_raw:,} -> {floats_compressed:,} "
          f"({floats_compressed/floats_raw*100:.1f}%)")

    del V, V_k, V_k_gpu, mu, eigenvalues, S_sparse
    torch.cuda.empty_cache()

print("\nAll chambers done.")

# %% [markdown]
# ## Summary: reconstruction-error curves for all chambers

# %%
plt.figure(figsize=(9, 5))
for chamber, rms in rms_curves.items():
    plt.plot(np.arange(1, len(rms) + 1), rms, label=chamber)
plt.axhline(ERROR_THRESHOLD_MM, color="k", linestyle="--",
            label=f"{ERROR_THRESHOLD_MM} mm threshold")
plt.xlabel("Number of PCs (k)")
plt.ylabel("RMS per-vertex error (mm)")
plt.title("Reconstruction error vs. number of PCs — all chambers")
plt.legend()
plt.tight_layout()
plt.show()
