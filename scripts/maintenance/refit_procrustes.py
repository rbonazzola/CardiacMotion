"""
Generalized Procrustes Analysis (GPA), refit from scratch against the new
b-values pipeline's own reconstructions, for all 6 cached partitions.

Writes procrustes_transforms_{partition}_refit.pkl into CardioMesh's cache
dir (NOT overwriting the existing procrustes_transforms_{partition}.pkl) so
the result can be validated before swapping it in.

Algorithm, per partition:
  1. Reconstruct each sampled subject's raw ED shape: PCA(b-values) + rigid
     pose (translation/qrotation/scale) -- same as
     cardio_mesh.pdm_reconstruction.reconstruct_shapes_from_bvalues, i.e.
     exactly what CardiacMeshFromBValuesDataset feeds into training, before
     any Procrustes step.
  2. Iterate: center each subject on its own centroid (that centroid is the
     "traslation" transform_mesh expects), solve the orthogonal Procrustes
     rotation (Kabsch) aligning it to the current reference template, update
     the reference to the mean of all aligned shapes, repeat.
  3. Reference starts at pca_mean (the PDM's own canonical mean shape) so
     the fit shrinks toward something the network's mean_shape already
     assumes, rather than an arbitrary subject.
Output schema matches cardio_mesh.procrustes.transform_mesh exactly:
  {subject_id: {"traslation": (3,) ndarray, "rotation": (3,3) ndarray}}
"""
import os
import pickle
import random
import sys
import time
from concurrent.futures import ThreadPoolExecutor

REPO_ROOT = "/net/scratch/t19767rb/src/CardiacMotion"
sys.path.insert(0, REPO_ROOT)
os.chdir(REPO_ROOT)

import numpy as np
from scipy.spatial.transform import Rotation

from cardio_mesh import paths as cardio_mesh_paths
from cardio_mesh.pdm_reconstruction import reconstruct_shapes_from_bvalues

PARAMS_DIR = "/net/scratch/t19767rb/src/CardiacSegmentation/.cache/params"
CACHE_DIR = "/net/scratch/t19767rb/src/CardioMesh/data/cached"
N_SUBJECTS = None  # None = use the full cohort (~96k), not a sample
N_LOAD_WORKERS = 32  # threads for the .npz reads -- I/O-bound, not CPU-bound
N_ITERS = 10
SEED = 0
PARTITIONS = ["left_ventricle", "right_ventricle", "biventricle", "left_atrium", "right_atrium", "aorta"]


def kabsch_rotation(P, Q):
    """Orthogonal R (det=+1) minimizing ||P @ R - Q||_F, P/Q already centered (N,3)."""
    M = P.T @ Q
    U, S, Vt = np.linalg.svd(M)
    d = np.sign(np.linalg.det(U @ Vt))
    D = np.diag([1.0, 1.0, d])
    return U @ D @ Vt


def spread(centroids):
    return np.linalg.norm(centroids - centroids.mean(0), axis=1).mean()


def _load_one(sid):
    try:
        d = np.load(os.path.join(PARAMS_DIR, f"{sid}.npz"))
        return sid, d["bvals"][0], d["translation"][0], d["qrotation"][0], d["scale"][0]
    except Exception as e:
        return sid, None, None, None, str(e)


def main():
    t0 = time.perf_counter()
    all_ids = sorted(f[:-4] for f in os.listdir(PARAMS_DIR) if f.endswith(".npz"))
    if N_SUBJECTS is None:
        subject_ids = all_ids
        print(f"Using the full cohort: {len(subject_ids)} subjects")
    else:
        random.seed(SEED)
        subject_ids = random.sample(all_ids, N_SUBJECTS)
        print(f"Sampled {len(subject_ids)} subjects (seed={SEED}) from {len(all_ids)} available")

    # Load each subject's ED (frame 0) b-values + pose once, reused across all
    # partitions -- parallelized (ThreadPoolExecutor releases the GIL during
    # each file's disk read, so this is a real speedup, not just overhead).
    bvals, translation, qrotation, scale, kept_ids = [], [], [], [], []
    n_done, n_failed = 0, 0
    with ThreadPoolExecutor(max_workers=N_LOAD_WORKERS) as ex:
        for sid, bv, tr, qr, sc in ex.map(_load_one, subject_ids):
            if bv is None:
                n_failed += 1
                if n_failed <= 10:
                    print(f"  skip {sid}: {sc}")
            else:
                bvals.append(bv)
                translation.append(tr)
                qrotation.append(qr)
                scale.append(sc)
                kept_ids.append(sid)
            n_done += 1
            if n_done % 10000 == 0:
                elapsed = time.perf_counter() - t0
                print(f"  loaded {n_done}/{len(subject_ids)} ({elapsed:.0f}s elapsed, "
                      f"~{elapsed / n_done * 1000:.2f}ms/subject, {n_failed} failed so far)")
    bvals = np.array(bvals)
    translation = np.array(translation)
    qrotation = np.array(qrotation)
    scale = np.array(scale)
    print(f"Loaded ED params for {len(kept_ids)} subjects in {time.perf_counter() - t0:.1f}s")

    for partition in PARTITIONS:
        t_p = time.perf_counter()
        pca_components = cardio_mesh_paths.get_pca_components(partition)
        pca_mean = cardio_mesh_paths.get_pca_mean(partition)
        n_verts = pca_mean.shape[0] // 3

        # Vectorized raw PCA reconstruction (T=1 per subject here, so reuse the
        # (T,...) function with T=n_subjects by treating subjects as the "time" axis).
        raw = reconstruct_shapes_from_bvalues(
            bvals, translation, qrotation, scale, pca_components, pca_mean
        )  # (n_subjects, n_verts, 3)

        raw_centroids = raw.mean(axis=1)
        print(f"\n=== {partition} (n_verts={n_verts}) ===")
        print(f"  raw spread (no Procrustes): {spread(raw_centroids):.1f} mm")

        reference = pca_mean.reshape(n_verts, 3).copy()
        reference -= reference.mean(0)

        translations = None
        rotations = None
        for it in range(N_ITERS):
            translations = raw.mean(axis=1)  # (n_subj, 3), each subject's own centroid
            centered = raw - translations[:, None, :]
            rotations = np.empty((len(kept_ids), 3, 3))
            aligned = np.empty_like(raw)
            for i in range(len(kept_ids)):
                R = kabsch_rotation(centered[i], reference)
                rotations[i] = R
                aligned[i] = centered[i] @ R
            new_reference = aligned.mean(axis=0)
            shift = np.linalg.norm(new_reference - reference, axis=1).mean()
            reference = new_reference
            aligned_centroids = aligned.mean(axis=1)  # should be ~0 by construction, sanity check
            print(f"  iter {it}: mean reference shift = {shift:.4f} mm, "
                  f"aligned-centroid residual = {np.abs(aligned_centroids).mean():.2e}")
            if shift < 1e-3:
                print("  converged")
                break

        # NOTE: post-fit centroid spread is NOT a meaningful check here -- translation
        # is defined as each subject's own centroid, so it's exactly 0 by construction
        # regardless of fit quality (unlike the earlier audit of the *cached* files,
        # where stored translation was independent of the raw reconstruction and a
        # nonzero centroid spread was real evidence of misalignment). The real
        # question is whether the *shape* converged -- measure per-vertex spread of
        # the aligned population around the final reference.
        aligned_all = np.array([
            (raw[i] - translations[i]).dot(rotations[i]) for i in range(len(kept_ids))
        ])
        per_vertex_rms = np.sqrt(((aligned_all - reference) ** 2).sum(-1).mean())
        print(f"  per-vertex RMS distance to converged reference: {per_vertex_rms:.2f} mm "
              f"(population shape variability after alignment)")

        out = {
            sid: {"traslation": translations[i], "rotation": rotations[i]}
            for i, sid in enumerate(kept_ids)
        }
        out_path = os.path.join(CACHE_DIR, f"procrustes_transforms_{partition}_refit.pkl")
        with open(out_path, "wb") as f:
            pickle.dump(out, f)
        print(f"  wrote {out_path} ({time.perf_counter() - t_p:.1f}s)")

    print(f"\nTotal: {time.perf_counter() - t0:.1f}s")


if __name__ == "__main__":
    main()
