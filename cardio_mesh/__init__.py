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


_CLOSED_CHAMBER_MAP = {
    "left_ventricle": "LV_closed", "LV": "LV_closed",
    "right_ventricle": "RV_closed", "RV": "RV_closed",
    "left_atrium": "LA_closed", "LA": "LA_closed",
    "right_atrium": "RA_closed", "RA": "RA_closed",
    "biventricle": "BV_closed", "BV": "BV_closed",
    "aorta": "aorta",
}


def close_chamber(partition: str) -> tuple[str, ...]:
    return _resolve_partition(_CLOSED_CHAMBER_MAP.get(partition, partition))


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


def load_fhm_topology() -> Cardiac3DMesh:
    """
    Loads only the (subject-independent) topology of the full heart mesh -
    faces and per-vertex partition labels - without reading any subject's
    reconstructed mesh from disk. Vertex positions are a placeholder (zeros):
    callers that need this topology to select per-partition faces (via
    Cardiac3DMesh.__getitem__) only use `.f`/`.subpart_id`, never `.v`, from
    the result.
    """
    faces = np.loadtxt(paths.get_fhm_faces_file(), delimiter=",", dtype=np.int64)
    subpart_id = np.array(paths.get_fhm_subpart_ids())
    vertices = np.zeros((len(subpart_id), 3))
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
