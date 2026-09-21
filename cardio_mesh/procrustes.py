from copy import copy

import numpy as np

try:
    import torch
except ImportError:  # torch is an optional dependency for this module
    torch = None


def transform_mesh(
    mesh,
    rotation: np.ndarray | None = None,
    traslation: np.ndarray | None = None,
):
    mesh = copy(mesh)

    if traslation is not None:
        mesh = mesh - traslation

    if rotation is not None:
        centroid = mesh.mean(axis=0)
        mesh -= centroid
        mesh = mesh.dot(rotation)
        mesh += centroid

    return mesh


def transform_mesh_torch(
    s_t: "torch.Tensor",
    rotation: "torch.Tensor",
    traslation: "torch.Tensor",
) -> "torch.Tensor":
    """
    Batched, GPU-friendly equivalent of transform_mesh, applying one
    (rotation, traslation) pair per subject to every frame of that subject
    (same transform across T -- this only removes per-subject pose, it must
    not touch genuine per-frame motion). Centroid for the rotation step is
    still computed per-frame, matching transform_mesh's semantics exactly
    (it centers whatever single (V,3) array it's given, and the numpy path
    calls it once per frame).

    Args:
        s_t: (B, T, n_verts, 3)
        rotation: (B, 3, 3)
        traslation: (B, 3)

    Returns:
        (B, T, n_verts, 3)
    """
    mesh = s_t - traslation.unsqueeze(1).unsqueeze(1)
    centroid = mesh.mean(dim=2, keepdim=True)  # (B, T, 1, 3), per-frame
    mesh = mesh - centroid
    mesh = torch.einsum("btvi,bij->btvj", mesh, rotation)
    return mesh + centroid
