from copy import copy

import numpy as np


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
