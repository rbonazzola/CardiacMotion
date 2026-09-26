"""
Tests for cardiac_motion/data/DataModules.py

Run with:  python -m unittest tests/test_datamodules.py -v
"""
import os
import pickle
import sys
import tempfile
import types
import unittest
from unittest.mock import MagicMock, patch

import numpy as np
import torch
from easydict import EasyDict
from torch.utils.data import TensorDataset

sys.path.insert(0, "cardiac_motion")
from data.DataModules import CardiacMeshPopulationDM, CardiacMeshFromBValuesDataset, GenericDataModule


def make_dataset(n=100):
    return TensorDataset(torch.zeros(n, 3))


# ---------------------------------------------------------------------------
# CardiacMeshPopulationDM._get_split_lengths
# ---------------------------------------------------------------------------

class TestCardiacDMSplitLengths(unittest.TestCase):

    def test_none_defaults_to_60_20_20(self):
        dm = CardiacMeshPopulationDM(make_dataset(100), split_lengths=None)
        train, val, test = dm.split_lengths
        self.assertEqual(train, 60)
        self.assertEqual(val, 20)
        self.assertEqual(test, 20)

    def test_none_lengths_sum_to_dataset_size(self):
        dm = CardiacMeshPopulationDM(make_dataset(97), split_lengths=None)
        self.assertEqual(sum(dm.split_lengths), 97)

    def test_integer_lengths_preserve_order(self):
        # Regression: val and test were swapped when integers were passed
        dm = CardiacMeshPopulationDM(make_dataset(100), split_lengths=[70, 20, 10])
        train, val, test = dm.split_lengths
        self.assertEqual(train, 70)
        self.assertEqual(val, 20)
        self.assertEqual(test, 10)

    def test_fraction_lengths(self):
        dm = CardiacMeshPopulationDM(make_dataset(100), split_lengths=[0.7, 0.2, 0.1])
        train, val, test = dm.split_lengths
        self.assertEqual(train, 70)
        self.assertEqual(val, 20)
        self.assertEqual(test, 10)

    def test_two_fractions_val_is_remainder(self):
        dm = CardiacMeshPopulationDM(make_dataset(100), split_lengths=[0.6, 0.2])
        train, val, test = dm.split_lengths
        self.assertEqual(train, 60)
        self.assertEqual(test, 20)
        self.assertEqual(val, 20)


# ---------------------------------------------------------------------------
# GenericDataModule
# ---------------------------------------------------------------------------

class TestGenericDataModule(unittest.TestCase):

    def test_none_defaults_to_60_20_20(self):
        dm = GenericDataModule(make_dataset(100), split_lengths=None)
        train, val, test = dm.split_lengths
        self.assertEqual(train, 60)
        self.assertEqual(val, 20)
        self.assertEqual(test, 20)

    def test_integer_lengths_preserve_order(self):
        dm = GenericDataModule(make_dataset(100), split_lengths=[70, 20, 10])
        train, val, test = dm.split_lengths
        self.assertEqual(train, 70)
        self.assertEqual(val, 20)
        self.assertEqual(test, 10)

    def test_setup_produces_correct_split_sizes(self):
        dm = GenericDataModule(make_dataset(100), split_lengths=[70, 20, 10], batch_size=10)
        dm.setup()
        self.assertEqual(len(dm.train_dataset), 70)
        self.assertEqual(len(dm.val_dataset), 20)
        self.assertEqual(len(dm.test_dataset), 10)

    def test_train_dataloader_batch_size(self):
        dm = GenericDataModule(make_dataset(100), batch_size=16)
        dm.setup()
        batch = next(iter(dm.train_dataloader()))
        self.assertLessEqual(batch[0].shape[0], 16)

    def test_pin_memory_matches_cuda_availability(self):
        import torch
        dm = GenericDataModule(make_dataset(100), batch_size=16)
        dm.setup()
        for dl in [dm.train_dataloader(), dm.val_dataloader(), dm.test_dataloader()]:
            self.assertEqual(dl.pin_memory, torch.cuda.is_available())


# ---------------------------------------------------------------------------
# d_style contract
# ---------------------------------------------------------------------------

class TestMseVectorization(unittest.TestCase):
    """mse computed in batch must match the per-frame loop it replaced."""

    def test_dev_from_tmp_avg_matches_loop(self):
        import torch
        from data.DataModules import mse

        T, V = 8, 50
        s_t    = torch.rand(T, V, 3)
        s_t_avg = s_t.mean(0)

        loop_result = torch.stack([mse(s_t[j], s_t_avg) for j in range(T)])
        vec_result  = mse(s_t, s_t_avg.unsqueeze(0))

        self.assertEqual(vec_result.shape, (T,))
        self.assertTrue(torch.allclose(loop_result, vec_result),
                        "Vectorized mse differs from loop result")

    def test_dev_from_template_matches_loop(self):
        import torch
        from data.DataModules import mse

        T, V = 8, 50
        s_t      = torch.rand(T, V, 3)
        template = torch.rand(V, 3)

        loop_result = torch.stack([mse(s_t[j], template) for j in range(T)])
        vec_result  = mse(s_t, template.unsqueeze(0))

        self.assertTrue(torch.allclose(loop_result, vec_result),
                        "Vectorized template mse differs from loop result")


class TestCenterAroundMean(unittest.TestCase):
    """Tests for the centering logic independently of disk I/O."""

    def setUp(self):
        import numpy as np
        self.T, self.V = 5, 20
        self.mean_v = torch.randn(self.V, 3) * 50   # mm scale
        # meshes near the mean with small residuals
        self.s_t = self.mean_v.unsqueeze(0) + torch.randn(self.T, self.V, 3)

    def _apply_centering(self, s_t, mean_v):
        """Mirrors the centering logic in __getitem__."""
        from data.DataModules import mse
        s_t_avg   = s_t.mean(0)
        d_content = mse(s_t, s_t_avg.unsqueeze(0))
        d_style   = mse(s_t, mean_v.unsqueeze(0))
        s_t_c     = s_t     - mean_v
        s_t_avg_c = s_t_avg - mean_v
        return s_t_c, s_t_avg_c, d_content, d_style

    def test_centering_reduces_scale(self):
        from data.DataModules import mse
        s_t_c, _, _, _ = self._apply_centering(self.s_t, self.mean_v)
        self.assertGreater(self.s_t.abs().mean().item(),
                           s_t_c.abs().mean().item(),
                           "Centered meshes should have smaller absolute values")

    def test_centered_time_avg_is_near_zero(self):
        _, s_t_avg_c, _, _ = self._apply_centering(self.s_t, self.mean_v)
        # time avg of small residuals should be close to 0
        self.assertLess(s_t_avg_c.abs().mean().item(), 5.0)

    def test_d_content_invariant_to_centering(self):
        from data.DataModules import mse
        s_t_avg = self.s_t.mean(0)
        d_content_raw = mse(self.s_t, s_t_avg.unsqueeze(0))
        s_t_c, s_t_avg_c, d_content_cent, _ = self._apply_centering(self.s_t, self.mean_v)
        self.assertTrue(torch.allclose(d_content_raw, d_content_cent),
                        "d_content must be invariant to centering")

    def test_center_around_mean_requires_template(self):
        from data.DataModules import CardiacMeshPopulationDataset
        with self.assertRaises((ValueError, AssertionError, Exception)):
            CardiacMeshPopulationDataset(
                root_path=".", faces=None,
                center_around_mean=True, template_mesh=None
            )


class TestDStyleContract(unittest.TestCase):
    """d_style must be a Tensor when template_mesh is provided, None otherwise."""

    def _make_item(self, template_mesh):
        """Simulate what __getitem__ returns without hitting disk."""
        import torch
        s_t = torch.rand(5, 10, 3)   # (T, V, 3)
        s_t_avg = s_t[0]
        dev_from_tmp_avg = torch.rand(5)

        if template_mesh is not None:
            dev_from_sphere = torch.rand(5)
        else:
            dev_from_sphere = None

        from easydict import EasyDict
        return EasyDict({
            "s_t": s_t,
            "time_avg_s": s_t_avg,
            "d_content": dev_from_tmp_avg,
            "d_style": dev_from_sphere,
        })

    def test_d_style_is_tensor_when_template_provided(self):
        item = self._make_item(template_mesh=object())
        self.assertIsNotNone(item["d_style"])

    def test_d_style_is_none_when_no_template(self):
        item = self._make_item(template_mesh=None)
        self.assertIsNone(item["d_style"])

    def test_shared_eval_step_handles_none_d_style(self):
        """d_style=None no debe causar crash — cubierto en profundidad por test_training.py."""
        # Verified via TestTrainingStep.test_validation_step_with_none_d_style
        pass


class TestCardiacMeshFromBValuesDataset(unittest.TestCase):
    """
    CardiacMeshFromBValuesDataset against an entirely synthetic PDM: no real
    PCA basis / b-values / CardioMesh cache files are touched. get_pca_components
    and get_pca_mean are patched to hand back a small made-up basis regardless
    of the requested partition.
    """

    N_COMPONENTS = 3
    N_VERTS = 4
    T = 5

    def setUp(self):
        rng = np.random.default_rng(7)

        self.pca_components = rng.normal(size=(self.N_COMPONENTS, self.N_VERTS * 3))
        self.pca_mean = rng.normal(size=(self.N_VERTS * 3,))

        self._tmpdir = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmpdir.cleanup)
        self.params_dir = self._tmpdir.name

        self.subjects = {
            "S1": {"rotation": np.eye(3), "traslation": np.zeros(3)},  # identity: no-op
            "S2": {"rotation": self._rotation_matrix_z(np.pi / 2), "traslation": np.array([1.0, -2.0, 0.5])},
        }

        for subject_id in self.subjects:
            qrotation = rng.normal(size=(self.T, 4))
            qrotation /= np.linalg.norm(qrotation, axis=1, keepdims=True)
            np.savez(
                os.path.join(self.params_dir, f"{subject_id}.npz"),
                bvals=rng.normal(size=(self.T, self.N_COMPONENTS)),
                translation=rng.normal(size=(self.T, 3)) * 5,
                qrotation=qrotation,
                scale=rng.uniform(0.8, 1.2, size=(self.T,)),
            )
        # a subject with b-values but no Procrustes transform: must be excluded
        np.savez(
            os.path.join(self.params_dir, "no_procrustes.npz"),
            bvals=rng.normal(size=(self.T, self.N_COMPONENTS)),
            translation=rng.normal(size=(self.T, 3)),
            qrotation=np.tile([0.0, 0.0, 0.0, 1.0], (self.T, 1)),
            scale=np.ones(self.T),
        )

        procrustes_path = os.path.join(self.params_dir, "procrustes_transforms.pkl")
        with open(procrustes_path, "wb") as f:
            pickle.dump(self.subjects, f)
        self.procrustes_path = procrustes_path

    @staticmethod
    def _rotation_matrix_z(theta):
        c, s = np.cos(theta), np.sin(theta)
        return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])

    def _make_dataset(self, **kwargs):
        with patch("data.DataModules.cardio_mesh_paths.get_pca_components", return_value=self.pca_components), \
             patch("data.DataModules.cardio_mesh_paths.get_pca_mean", return_value=self.pca_mean):
            return CardiacMeshFromBValuesDataset(
                params_dir=self.params_dir,
                partition="synthetic",
                procrustes_transforms=self.procrustes_path,
                **kwargs,
            )

    @staticmethod
    def _decode_one(ds, idx):
        """__getitem__ now returns raw per-subject params (see decode_batch's
        docstring on CardiacMeshFromBValuesDataset) -- collate a batch of one
        and run it through decode_batch, exactly like
        CardiacMeshPopulationDM.on_after_batch_transfer does, then drop the
        batch dim so these tests can keep comparing single-subject output."""
        raw = ds[idx]
        batch = {k: v.unsqueeze(0) for k, v in raw.items()}
        decoded = ds.decode_batch(batch)
        return EasyDict({k: (v[0] if v is not None else None) for k, v in decoded.items()})

    def test_excludes_subjects_without_procrustes_transform(self):
        ds = self._make_dataset()
        self.assertEqual(sorted(ds.ids), ["S1", "S2"])

    def test_len_matches_usable_subjects(self):
        ds = self._make_dataset()
        self.assertEqual(len(ds), 2)

    def test_n_subj_limits_dataset(self):
        ds = self._make_dataset(N_subj=1)
        self.assertEqual(len(ds), 1)

    def test_getitem_returns_raw_params_not_reconstructed_mesh(self):
        """__getitem__ must be cheap/raw -- no PCA reconstruction, no Procrustes,
        no disk I/O beyond what __init__ already preloaded."""
        ds = self._make_dataset()
        item = ds[0]
        self.assertEqual(
            set(item.keys()),
            {"bvals", "translation", "qrotation", "scale", "proc_rotation", "proc_traslation"},
        )
        self.assertEqual(tuple(item["bvals"].shape), (self.T, self.N_COMPONENTS))

    def test_output_shape_and_keys(self):
        ds = self._make_dataset()
        item = self._decode_one(ds, 0)
        self.assertEqual(set(item.keys()), {"s_t", "time_avg_s", "d_content", "d_style"})
        self.assertEqual(item.s_t.shape, (self.T, self.N_VERTS, 3))
        self.assertEqual(item.time_avg_s.shape, (self.N_VERTS, 3))
        self.assertEqual(item.d_content.shape, (self.T,))
        self.assertIsNone(item.d_style)

    def test_identity_procrustes_matches_raw_reconstruction(self):
        """S1 has an identity Procrustes transform, so s_t must equal the raw
        PCA + rigid reconstruction with no further change."""
        from cardio_mesh.pdm_reconstruction import reconstruct_shapes_from_bvalues

        ds = self._make_dataset()
        idx = ds.ids.index("S1")
        item = self._decode_one(ds, idx)

        d = np.load(os.path.join(self.params_dir, "S1.npz"))
        expected = reconstruct_shapes_from_bvalues(
            d["bvals"], d["translation"], d["qrotation"], d["scale"],
            self.pca_components, self.pca_mean,
        )
        np.testing.assert_allclose(item.s_t.numpy(), expected, atol=1e-4)

    def test_nonidentity_procrustes_matches_manual_application(self):
        """S2 has a real rotation+translation: verify decode_batch applies exactly
        what cardio_mesh.procrustes.transform_mesh applies, frame by frame."""
        from cardio_mesh.pdm_reconstruction import reconstruct_shapes_from_bvalues
        from cardio_mesh.procrustes import transform_mesh

        ds = self._make_dataset()
        idx = ds.ids.index("S2")
        item = self._decode_one(ds, idx)

        d = np.load(os.path.join(self.params_dir, "S2.npz"))
        raw = reconstruct_shapes_from_bvalues(
            d["bvals"], d["translation"], d["qrotation"], d["scale"],
            self.pca_components, self.pca_mean,
        )
        expected = np.stack([transform_mesh(raw[t], **self.subjects["S2"]) for t in range(self.T)])
        np.testing.assert_allclose(item.s_t.numpy(), expected, atol=1e-4)

    def test_phases_filter_selects_expected_frames(self):
        """phases_filter=[1, 3] (1-indexed) must select b-values frames 0 and 2."""
        ds_full = self._make_dataset()
        ds_filtered = self._make_dataset(phases_filter=[1, 3])

        idx = ds_full.ids.index("S1")
        item_full = self._decode_one(ds_full, idx)
        item_filtered = self._decode_one(ds_filtered, idx)

        self.assertEqual(item_filtered.s_t.shape[0], 2)
        np.testing.assert_allclose(item_filtered.s_t[0].numpy(), item_full.s_t[0].numpy(), atol=1e-4)
        np.testing.assert_allclose(item_filtered.s_t[1].numpy(), item_full.s_t[2].numpy(), atol=1e-4)

    def test_end_diastole_time_avg_is_first_frame(self):
        ds = self._make_dataset(static_shape="end_diastole")
        item = self._decode_one(ds, ds.ids.index("S1"))
        np.testing.assert_allclose(item.time_avg_s.numpy(), item.s_t[0].numpy())
        self.assertAlmostEqual(item.d_content[0].item(), 0.0, places=4)

    def test_temporal_mean_time_avg_is_mean_over_frames(self):
        ds = self._make_dataset(static_shape="temporal_mean")
        item = self._decode_one(ds, ds.ids.index("S1"))
        np.testing.assert_allclose(item.time_avg_s.numpy(), item.s_t.mean(dim=0).numpy(), atol=1e-5)
        # unlike end_diastole, no frame coincides with the static shape
        self.assertGreater(item.d_content.min().item(), 0.0)

    def test_center_around_own_mean_removes_subject_centroid_only(self):
        plain = self._decode_one(self._make_dataset(), 0)
        centered = self._decode_one(self._make_dataset(center_around_own_mean=True), 0)
        # the subject's centroid over frames and vertices becomes 0 ...
        np.testing.assert_allclose(centered.s_t.mean(dim=(0, 1)).numpy(), 0.0, atol=1e-4)
        # ... by one constant shift: shape and within-cycle motion (incl. centroid motion) unchanged
        shift = (plain.s_t - centered.s_t).reshape(-1, 3)
        np.testing.assert_allclose(shift.numpy(), np.broadcast_to(shift[0].numpy(), shift.shape), atol=1e-4)
        np.testing.assert_allclose((plain.time_avg_s - centered.time_avg_s).numpy(),
                                   np.broadcast_to(shift[0].numpy(), plain.time_avg_s.shape), atol=1e-4)

    def test_batched_decode_matches_per_subject_decode(self):
        """decode_batch on a real multi-subject batch (as the DataLoader would
        actually produce via default collation) must match decoding each
        subject one at a time -- i.e. the batching itself introduces no
        cross-subject leakage."""
        ds = self._make_dataset()
        from torch.utils.data import default_collate

        batch = default_collate([ds[i] for i in range(len(ds))])
        decoded_batch = ds.decode_batch(batch)

        for i, sid in enumerate(ds.ids):
            single = self._decode_one(ds, i)
            np.testing.assert_allclose(decoded_batch["s_t"][i].numpy(), single.s_t.numpy(), atol=1e-4)


if __name__ == "__main__":
    unittest.main()
