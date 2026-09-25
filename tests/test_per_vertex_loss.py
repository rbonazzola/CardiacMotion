'''
The reconstruction loss ("mse") is the squared Euclidean distance per vertex (summed over xyz),
averaged over vertices, frames and subjects -- and the translation/shape split stays exact.
'''
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, "cardiac_motion")
from lightning_modules.ComaLightningModule import losses_menu, per_vertex_mse, translation_shape_split_loss


def test_per_vertex_mse_is_mean_squared_vertex_distance():
    torch.manual_seed(0)
    real, recon = torch.randn(4, 10, 50, 3), torch.randn(4, 10, 50, 3)
    squared_distances = torch.linalg.vector_norm(real - recon, dim=-1) ** 2  # (B, T, V)
    torch.testing.assert_close(per_vertex_mse(real, recon), squared_distances.mean())
    torch.testing.assert_close(per_vertex_mse(real, recon), 3 * F.mse_loss(real, recon))
    assert losses_menu["mse"] is per_vertex_mse


def test_translation_shape_split_still_exact():
    torch.manual_seed(1)
    real = torch.randn(4, 10, 50, 3)
    recon = real + 0.3 * torch.randn(4, 10, 50, 3) + torch.tensor([1.0, -2.0, 0.5])  # includes a shift
    translation, shape = translation_shape_split_loss(per_vertex_mse, real, recon)
    torch.testing.assert_close(translation + shape, per_vertex_mse(real, recon))
    # translation term = squared distance between centroids
    centroid_dist2 = ((real.mean(-2) - recon.mean(-2)) ** 2).sum(-1).mean()
    torch.testing.assert_close(translation, centroid_dist2)
