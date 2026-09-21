"""
Tests for cardio_mesh/pdm_reconstruction.py, using synthetic data only
(no dependency on the real PDM / CardioMesh cache files).

Run with:  python -m unittest tests/test_pdm_reconstruction.py -v
"""
import unittest

import numpy as np

from cardio_mesh.pdm_reconstruction import quaternion_rotate, reconstruct_shapes_from_bvalues

try:
    import torch
    from scipy.spatial.transform import Rotation as _ScipyRotation
    from cardio_mesh.pdm_reconstruction import reconstruct_shapes_from_bvalues_torch
    from cardio_mesh.procrustes import transform_mesh, transform_mesh_torch
    _HAS_TORCH = True
except ImportError:
    _HAS_TORCH = False


IDENTITY_QUAT = np.array([0.0, 0.0, 0.0, 1.0])  # scipy convention: [x, y, z, w]


class TestQuaternionRotate(unittest.TestCase):

    def test_identity_quaternion_is_noop(self):
        points = np.array([[1.0, 2.0, 3.0], [-1.0, 0.5, 4.0]])
        rotated = quaternion_rotate(points, IDENTITY_QUAT)
        np.testing.assert_allclose(rotated, points)

    def test_90_degree_rotation_about_z(self):
        # Rotating [1, 0, 0] by 90 deg about z should give [0, 1, 0].
        quat_90_z = np.array([0.0, 0.0, np.sin(np.pi / 4), np.cos(np.pi / 4)])
        points = np.array([[1.0, 0.0, 0.0]])
        rotated = quaternion_rotate(points, quat_90_z)
        np.testing.assert_allclose(rotated, [[0.0, 1.0, 0.0]], atol=1e-10)

    def test_preserves_pairwise_distances(self):
        rng = np.random.default_rng(0)
        points = rng.normal(size=(5, 3))
        quat = rng.normal(size=4)
        quat /= np.linalg.norm(quat)
        rotated = quaternion_rotate(points, quat)
        d0 = np.linalg.norm(points[0] - points[1])
        d1 = np.linalg.norm(rotated[0] - rotated[1])
        self.assertAlmostEqual(d0, d1, places=10)


class TestReconstructShapesFromBValues(unittest.TestCase):
    """
    Uses a small synthetic PDM (n_components=3, n_verts=4) instead of the real
    (70, 583623) basis, so the composed affine map can be checked exactly
    against a plain per-frame reference computation.
    """

    def setUp(self):
        rng = np.random.default_rng(42)
        self.n_components = 3
        self.n_verts = 4
        self.T = 6

        self.pca_components = rng.normal(size=(self.n_components, self.n_verts * 3))
        self.pca_mean = rng.normal(size=(self.n_verts * 3,))

        self.bvalues = rng.normal(size=(self.T, self.n_components))
        self.translation = rng.normal(size=(self.T, 3)) * 10
        qrotation = rng.normal(size=(self.T, 4))
        self.qrotation = qrotation / np.linalg.norm(qrotation, axis=1, keepdims=True)
        self.scale = rng.uniform(0.5, 2.0, size=(self.T,))

    def _reference_reconstruction(self):
        """Same math, written as an explicit per-frame loop (no vectorization),
        to sanity-check reconstruct_shapes_from_bvalues independently."""
        out = np.empty((self.T, self.n_verts, 3))
        for t in range(self.T):
            shape_t = (self.bvalues[t] @ self.pca_components + self.pca_mean).reshape(self.n_verts, 3)
            out[t] = quaternion_rotate(shape_t / self.scale[t], self.qrotation[t]) + self.translation[t]
        return out

    def test_matches_reference_loop(self):
        result = reconstruct_shapes_from_bvalues(
            self.bvalues, self.translation, self.qrotation, self.scale,
            self.pca_components, self.pca_mean,
        )
        np.testing.assert_allclose(result, self._reference_reconstruction())

    def test_output_shape(self):
        result = reconstruct_shapes_from_bvalues(
            self.bvalues, self.translation, self.qrotation, self.scale,
            self.pca_components, self.pca_mean,
        )
        self.assertEqual(result.shape, (self.T, self.n_verts, 3))

    def test_identity_transform_is_plain_pca_reconstruction(self):
        T = 2
        identity_quat = np.tile(IDENTITY_QUAT, (T, 1))
        zero_translation = np.zeros((T, 3))
        unit_scale = np.ones(T)
        bvalues = self.bvalues[:T]

        result = reconstruct_shapes_from_bvalues(
            bvalues, zero_translation, identity_quat, unit_scale,
            self.pca_components, self.pca_mean,
        )
        expected = (bvalues @ self.pca_components + self.pca_mean).reshape(T, self.n_verts, 3)
        np.testing.assert_allclose(result, expected)

    def test_zero_bvalues_reconstructs_mean_shape(self):
        T = 1
        bvalues = np.zeros((T, self.n_components))
        identity_quat = np.tile(IDENTITY_QUAT, (T, 1))
        zero_translation = np.zeros((T, 3))
        unit_scale = np.ones(T)

        result = reconstruct_shapes_from_bvalues(
            bvalues, zero_translation, identity_quat, unit_scale,
            self.pca_components, self.pca_mean,
        )
        np.testing.assert_allclose(result[0], self.pca_mean.reshape(self.n_verts, 3))


@unittest.skipUnless(_HAS_TORCH, "torch not installed")
class TestReconstructShapesFromBValuesTorch(unittest.TestCase):
    """
    Batched torch reconstruction (reconstruct_shapes_from_bvalues_torch +
    transform_mesh_torch) must match the numpy/scipy per-subject, per-frame
    reference exactly (up to float32 precision) -- this is what
    CardiacMeshFromBValuesDataset.decode_batch relies on for the GPU-side
    reconstruction path.
    """

    def setUp(self):
        rng = np.random.default_rng(1)
        self.B, self.T, self.n_components, self.n_verts = 5, 7, 70, 40

        self.pca_components = rng.normal(size=(self.n_components, self.n_verts * 3)).astype(np.float32)
        self.pca_mean = rng.normal(size=(self.n_verts * 3,)).astype(np.float32)

        self.bvals = rng.normal(size=(self.B, self.T, self.n_components)).astype(np.float32)
        self.translation = (rng.normal(size=(self.B, self.T, 3)) * 10).astype(np.float32)
        q = rng.normal(size=(self.B, self.T, 4)).astype(np.float32)
        self.qrotation = (q / np.linalg.norm(q, axis=-1, keepdims=True)).astype(np.float32)
        self.scale = rng.uniform(0.5, 2.0, size=(self.B, self.T)).astype(np.float32)

        self.rotation = np.stack(
            [_ScipyRotation.random(random_state=i).as_matrix() for i in range(self.B)]
        ).astype(np.float32)
        self.traslation = (rng.normal(size=(self.B, 3)) * 5).astype(np.float32)

    def _numpy_reference(self):
        ref = np.empty((self.B, self.T, self.n_verts, 3), dtype=np.float32)
        for b in range(self.B):
            raw = reconstruct_shapes_from_bvalues(
                self.bvals[b], self.translation[b], self.qrotation[b], self.scale[b],
                self.pca_components, self.pca_mean,
            )
            ref[b] = np.stack([
                transform_mesh(raw[t], rotation=self.rotation[b], traslation=self.traslation[b])
                for t in range(self.T)
            ])
        return ref

    def test_matches_numpy_reference(self):
        raw_t = reconstruct_shapes_from_bvalues_torch(
            torch.from_numpy(self.bvals), torch.from_numpy(self.translation),
            torch.from_numpy(self.qrotation), torch.from_numpy(self.scale),
            torch.from_numpy(self.pca_components), torch.from_numpy(self.pca_mean),
        )
        aligned_t = transform_mesh_torch(
            raw_t, torch.from_numpy(self.rotation), torch.from_numpy(self.traslation)
        ).numpy()
        np.testing.assert_allclose(aligned_t, self._numpy_reference(), atol=1e-3)

    def test_output_shape(self):
        raw_t = reconstruct_shapes_from_bvalues_torch(
            torch.from_numpy(self.bvals), torch.from_numpy(self.translation),
            torch.from_numpy(self.qrotation), torch.from_numpy(self.scale),
            torch.from_numpy(self.pca_components), torch.from_numpy(self.pca_mean),
        )
        self.assertEqual(tuple(raw_t.shape), (self.B, self.T, self.n_verts, 3))


if __name__ == "__main__":
    unittest.main()
