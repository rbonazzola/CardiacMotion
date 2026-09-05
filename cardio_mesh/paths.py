import os
import pickle

import numpy as np

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
CARDIOMESH_DIR = os.environ.get(
    "CARDIOMESH_DIR",
    os.path.abspath(os.path.join(REPO_ROOT, os.pardir, "CardioMesh")),
)
CACHE_DIR = os.path.join(CARDIOMESH_DIR, "data", "cached")
DATA_DIR = os.path.join(CARDIOMESH_DIR, "data")

_CACHE_NAMES = {
    "left_ventricle": "left_ventricle",
    "right_ventricle": "right_ventricle",
    "biventricle": "biventricle",
    "left_atrium": "left_atrium",
    "right_atrium": "right_atrium",
    "aorta": "aorta",
}


def get_subsetting_matrix(partition: str):
    return _load_cached(f"subsetting_matrix_{_cache_name(partition)}", prefer_npy=True)


def get_mean_shape(partition: str):
    return _load_cached(f"mean_shape_{_cache_name(partition)}", prefer_npy=True)


def get_procrustes_file(partition: str) -> str:
    return _cached_file(f"procrustes_transforms_{_cache_name(partition)}", ".pkl")


def get_fhm_faces_file() -> str:
    return os.path.join(DATA_DIR, "faces_fhm_10pct_decimation.csv")


def get_fhm_subpart_ids() -> list[str]:
    with open(os.path.join(DATA_DIR, "subpartIDs_FHM_10pct.txt"), "rt") as f:
        return [line.strip() for line in f]


def _cache_name(partition: str) -> str:
    try:
        return _CACHE_NAMES[partition]
    except KeyError as exc:
        raise ValueError(f"Unknown cardiac partition: {partition}") from exc


def _cached_file(stem: str, suffix: str) -> str:
    path = os.path.join(CACHE_DIR, stem + suffix)
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    return path


def _load_cached(stem: str, prefer_npy: bool):
    suffixes = (".npy", ".pkl") if prefer_npy else (".pkl", ".npy")
    for suffix in suffixes:
        path = os.path.join(CACHE_DIR, stem + suffix)
        if not os.path.exists(path):
            continue
        if suffix == ".npy":
            return np.load(path, allow_pickle=True)
        with open(path, "rb") as f:
            return pickle.load(f)
    raise FileNotFoundError(f"No cached file found for {stem!r} in {CACHE_DIR}")
