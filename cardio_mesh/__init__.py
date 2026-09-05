import os
from dataclasses import dataclass
from typing import Iterable

import numpy as np

from . import paths

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
MESHES_DIR = os.path.join(REPO_ROOT, "data", "cardio", "meshes")


_CLOSED_PARTITIONS = {
    "LA_closed": ("LA", "MVP", "PV1", "PV2", "PV3", "PV4", "PV5"),
    "RA_closed": ("RA", "TVP", "PV6", "PV7"),
    "LV_closed": ("LV", "AVP", "MVP"),
    "RV_closed": ("RV", "PVP", "TVP"),
    "BV_closed": ("LV", "AVP", "MVP", "RV", "PVP", "TVP"),
    "aorta": ("aorta",),
}

_PARTITION_ALIASES = {
    "left_ventricle": ("LV",),
    "right_ventricle": ("RV",),
    "biventricle": ("LV", "RV"),
    "left_atrium": ("LA",),
    "right_atrium": ("RA",),
    "aorta": ("aorta",),
    **_CLOSED_PARTITIONS,
}


@dataclass
class Cardiac3DMesh:
    v: np.ndarray
    f: np.ndarray
    subpart_id: np.ndarray | None = None

    @property
    def points(self):
        return self.v

    @property
    def triangles(self):
        return self.f

    def __getitem__(self, partition):
        labels = _resolve_partition(partition)
        if self.subpart_id is None:
            raise ValueError("Cannot extract a partition without subpart ids.")

        keep = np.isin(self.subpart_id, labels)
        old_vertex_ids = np.flatnonzero(keep)
        old_to_new = {old: new for new, old in enumerate(old_vertex_ids)}

        face_mask = keep[self.f].all(axis=1)
        faces = self.f[face_mask]
        remapped_faces = np.array(
            [[old_to_new[int(vertex_id)] for vertex_id in face] for face in faces],
            dtype=np.int64,
        )

        return Cardiac3DMesh(
            v=self.v[keep],
            f=remapped_faces,
            subpart_id=self.subpart_id[keep],
        )


class CardiacMeshPopulation:
    pass


def close_chamber(partition: str) -> tuple[str, ...]:
    return _resolve_partition(partition)


def load_full_heart_mesh(subject_id: str, timeframe: int = 1) -> Cardiac3DMesh:
    mesh_path = os.path.join(
        MESHES_DIR,
        str(subject_id),
        "models",
        f"FHM_res_0.1_time{timeframe:03d}.npy",
    )
    if not os.path.exists(mesh_path):
        mesh_path = os.path.join(
            MESHES_DIR,
            str(subject_id),
            "models",
            f"FHM_time{timeframe:03d}.npy",
        )

    vertices = np.load(mesh_path)
    faces = np.loadtxt(paths.get_fhm_faces_file(), delimiter=",", dtype=np.int64)
    subpart_id = np.array(paths.get_fhm_subpart_ids())
    return Cardiac3DMesh(vertices, faces, subpart_id)


def _resolve_partition(partition) -> tuple[str, ...]:
    if isinstance(partition, str):
        return _PARTITION_ALIASES.get(partition, (partition,))

    if isinstance(partition, Iterable):
        labels = []
        for item in partition:
            labels.extend(_resolve_partition(item))
        return tuple(labels)

    raise TypeError(f"Unsupported partition type: {type(partition)!r}")
