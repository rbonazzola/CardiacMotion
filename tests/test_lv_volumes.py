'''
LV cavity volumes from endocardial convex hulls, and the end-systolic frame (minimum volume).
'''
import sys

import numpy as np

sys.path.insert(0, "cardiac_motion")
from utils.lv_volumes import endocardial_volumes, volume_table


def _sphere(radius_mm, n=2000, seed=0):
    points = np.random.default_rng(seed).normal(size=(n, 3))
    return radius_mm * points / np.linalg.norm(points, axis=1, keepdims=True)


def test_endocardial_volume_of_spheres_in_ml():
    radii = np.array([30.0, 25.0, 20.0])  # mm
    volumes = endocardial_volumes(np.stack([_sphere(r) for r in radii]))
    expected = 4 / 3 * np.pi * radii ** 3 / 1000  # ml
    np.testing.assert_allclose(volumes, expected, rtol=0.01)  # a dense hull of points on the sphere


def test_volume_table_frames_are_1_based_and_ef():
    frames = [1, 6, 11, 16]
    volumes = np.array([[120.0, 90.0, 50.0, 80.0],     # end systole at the 3rd frame (frame 11)
                        [100.0, 110.0, 70.0, 40.0]])   # max at frame 6, min at frame 16
    table = volume_table(["A", "B"], volumes, frames)
    assert list(table.columns[:5]) == ["subject_id", "vol_frame01", "vol_frame06", "vol_frame11", "vol_frame16"]
    assert table.es_frame.tolist() == [11, 16] and table.ed_frame.tolist() == [1, 6]
    np.testing.assert_allclose(table.ef, [(120 - 50) / 120, (110 - 40) / 110])
