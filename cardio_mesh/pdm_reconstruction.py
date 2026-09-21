"""
Reconstructs cardiac meshes directly from PDM b-values + per-frame rigid/scale
transform + per-subject population Procrustes transform, skipping the
full-resolution mesh entirely.

This composes, into a single small affine map per (subject, frame):
  1. PCA reconstruction (b-values -> shape), already collapsed at decimation
     time to the final (decimated + partitioned) vertex set - see
     cardio_mesh/paths.py:get_pca_components / get_pca_mean, and the
     precompute step that produced those cached arrays.
  2. The per-frame rigid + isotropic-scale transform predicted alongside the
     b-values (translation, qrotation, scale), inverted back to real-world
     space. Ported verbatim (semantics, not just shape) from
     CardiacSegmentation/pycardiox/pdm/shapes.py::transform_shape (inverse
     path) so reconstructions match the original pipeline's outputs exactly
     (validated empirically: RMSE ~1e-6 relative to decimating the pipeline's
     own full-resolution reconstruction).

Decimation and partition selection do not appear here at all: both are pure
vertex-selection matrices (confirmed empirically: exactly one nonzero, equal
to 1, per row), so they were folded into the cached PCA basis once, globally,
instead of being applied per subject.
"""
import numpy as np
from scipy.spatial.transform import Rotation

try:
    import torch
except ImportError:  # torch is an optional dependency for this module
    torch = None


def quaternion_rotate(points: np.ndarray, qrotate) -> np.ndarray:
    """
    Rotate an (N, 3) array of points by a quaternion [q1, q2, q3, q4].

    Ported from CardiacSegmentation/pycardiox/pdm/quaternion.py::quaternion_rotate.
    """
    return Rotation.from_quat(qrotate).apply(points)


def reconstruct_shapes_from_bvalues(
    bvalues: np.ndarray,
    translation: np.ndarray,
    qrotation: np.ndarray,
    scale: np.ndarray,
    pca_components: np.ndarray,
    pca_mean: np.ndarray,
) -> np.ndarray:
    """
    Reconstruct a subject's decimated/partitioned mesh for every frame.

    Args:
        bvalues: (T, n_components) PDM coefficients, one row per frame.
        translation: (T, 3) per-frame translation.
        qrotation: (T, 4) per-frame rotation quaternion.
        scale: (T,) per-frame isotropic scale.
        pca_components: (n_components, n_verts*3) cached, already restricted
            to the decimated + partitioned vertex set.
        pca_mean: (n_verts*3,) cached mean shape, same restriction.

    Returns:
        (T, n_verts, 3) reconstructed vertices, in the same space the
        population Procrustes transform (cardio_mesh.procrustes.transform_mesh)
        expects as input.
    """
    T = bvalues.shape[0]
    n_verts = pca_mean.shape[0] // 3

    # Step 1: PCA reconstruction, already at final resolution. Affine in bvalues.
    shapes = bvalues @ pca_components + pca_mean  # (T, n_verts*3)
    shapes = shapes.reshape(T, n_verts, 3)

    # Step 2: per-frame rigid + scale transform, inverted back to real-world space.
    out = np.empty_like(shapes)
    for t in range(T):
        out[t] = quaternion_rotate(shapes[t] / scale[t], qrotation[t]) + translation[t]

    return out


def _quat_to_rotmat_torch(q: "torch.Tensor") -> "torch.Tensor":
    """
    Batched quaternion (..., 4) in [x, y, z, w] convention (scipy default) ->
    rotation matrix (..., 3, 3). Matches scipy.spatial.transform.Rotation
    exactly (same convention, same normalization).
    """
    q = q / q.norm(dim=-1, keepdim=True)
    x, y, z, w = q.unbind(-1)
    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z
    R = torch.stack([
        1 - 2 * (yy + zz), 2 * (xy - wz), 2 * (xz + wy),
        2 * (xy + wz), 1 - 2 * (xx + zz), 2 * (yz - wx),
        2 * (xz - wy), 2 * (yz + wx), 1 - 2 * (xx + yy),
    ], dim=-1)
    return R.reshape(*q.shape[:-1], 3, 3)


def reconstruct_shapes_from_bvalues_torch(
    bvalues: "torch.Tensor",
    translation: "torch.Tensor",
    qrotation: "torch.Tensor",
    scale: "torch.Tensor",
    pca_components: "torch.Tensor",
    pca_mean: "torch.Tensor",
) -> "torch.Tensor":
    """
    Batched, GPU-friendly equivalent of reconstruct_shapes_from_bvalues, for an
    extra leading batch dimension B (e.g. B subjects in one training batch,
    each with T frames). Numerically matches the numpy/scipy version (same
    quaternion convention, same operation order) -- see
    tests/test_pdm_reconstruction.py::TestReconstructShapesFromBValuesTorch.

    Args:
        bvalues: (B, T, n_components)
        translation: (B, T, 3)
        qrotation: (B, T, 4), [x, y, z, w] convention
        scale: (B, T)
        pca_components: (n_components, n_verts*3)
        pca_mean: (n_verts*3,)

    Returns:
        (B, T, n_verts, 3)
    """
    B, T, n_components = bvalues.shape
    n_verts = pca_mean.shape[0] // 3

    shapes = bvalues.reshape(B * T, n_components) @ pca_components + pca_mean
    shapes = shapes.reshape(B, T, n_verts, 3)
    shapes = shapes / scale.unsqueeze(-1).unsqueeze(-1)

    R = _quat_to_rotmat_torch(qrotation)  # (B, T, 3, 3)
    # per (b, t): shapes[b, t] @ R[b, t].T  -- matches Rotation.from_quat(q).apply(points)
    rotated = torch.einsum("btvi,btji->btvj", shapes, R)
    return rotated + translation.unsqueeze(2)
