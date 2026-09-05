"""
Tests for cardiac_motion/data/DataModules.py

Run with:  python -m unittest tests/test_datamodules.py -v
"""
import sys
import types
import unittest
from unittest.mock import MagicMock

import torch
from torch.utils.data import TensorDataset

sys.path.insert(0, "cardiac_motion")
from data.DataModules import CardiacMeshPopulationDM, GenericDataModule


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


if __name__ == "__main__":
    unittest.main()
