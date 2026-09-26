'''
val/test_rec_ratio_to_time_mean_pooled = total reconstruction error / total deviation from the
static shape over the whole epoch (a ratio of totals, not a mean of per-frame ratios).
'''
import sys

import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, "cardiac_motion")
from lightning_modules.ComaLightningModule import mse, pooled_mean_vertex_dev, pooled_ratio
from test_training import build_model, load_fixture, make_batch, make_lit


def test_pooled_ratio_is_ratio_of_totals():
    outputs = [{"rec_err_sum": torch.tensor(1.0), "dev_static_sum": torch.tensor(10.0)},
               {"rec_err_sum": torch.tensor(3.0), "dev_static_sum": torch.tensor(10.0)}]
    assert abs(pooled_ratio(outputs).item() - 0.2) < 1e-6  # (1 + 3) / (10 + 10), not mean(0.1, 0.3)
    assert torch.isnan(pooled_ratio([{"rec_err_sum": torch.tensor(1.0), "dev_static_sum": torch.tensor(0.0)}]))


def test_validation_logs_pooled_ratio():
    A, D, U, n_nodes = load_fixture()
    lit = make_lit(build_model(A, D, U, n_nodes))
    torch.manual_seed(0)
    batches = [make_batch(n_nodes[0]) for _ in range(3)]

    trainer = pl.Trainer(logger=False, accelerator="cpu", enable_progress_bar=False, enable_model_summary=False)
    metrics = trainer.validate(lit, DataLoader(batches, batch_size=None), verbose=False)[0]

    lit.eval()  # validate() leaves the module in train mode (batch norm would use batch statistics)
    with torch.no_grad():
        rec_err = sum(mse(b["s_t"], lit(b["s_t"])[2]).sum() for b in batches)
    expected = rec_err / sum(b["d_content"].sum() for b in batches)
    assert abs(metrics["val_rec_ratio_to_time_mean_pooled"] - expected.item()) < 1e-4 * expected.item()


def test_pooled_mean_vertex_dev_weights_by_vertex_count():
    # batches of different size: exact mean over all vertices, not mean of per-batch means
    outputs = [{"vertex_dev_sum": torch.tensor(10.0), "vertex_count": torch.tensor(10.0)},   # mean 1
               {"vertex_dev_sum": torch.tensor(30.0), "vertex_count": torch.tensor(10.0)},   # mean 3
               {"vertex_dev_sum": torch.tensor(20.0), "vertex_count": torch.tensor(20.0)}]   # mean 1
    assert abs(pooled_mean_vertex_dev(outputs).item() - 60.0 / 40.0) < 1e-6


def test_validation_logs_mean_vertex_dev():
    A, D, U, n_nodes = load_fixture()
    lit = make_lit(build_model(A, D, U, n_nodes))
    torch.manual_seed(0)
    batches = [make_batch(n_nodes[0]) for _ in range(3)]

    trainer = pl.Trainer(logger=False, accelerator="cpu", enable_progress_bar=False, enable_model_summary=False)
    metrics = trainer.validate(lit, DataLoader(batches, batch_size=None), verbose=False)[0]

    lit.eval()
    with torch.no_grad():
        distances = torch.cat([torch.linalg.vector_norm(b["s_t"] - lit(b["s_t"])[2], dim=-1).flatten() for b in batches])
    assert abs(metrics["val_mean_vertex_dev"] - distances.mean().item()) < 1e-4 * distances.mean().item()
    # mean of distances <= RMS distance (sqrt of the per-vertex squared-distance loss)
    assert metrics["val_mean_vertex_dev"] <= (distances ** 2).mean().sqrt().item() + 1e-6


def test_validation_logs_rec_s_translation_shape_split():
    A, D, U, n_nodes = load_fixture()
    lit = make_lit(build_model(A, D, U, n_nodes))
    torch.manual_seed(0)
    batches = [make_batch(n_nodes[0]) for _ in range(3)]
    trainer = pl.Trainer(logger=False, accelerator="cpu", enable_progress_bar=False, enable_model_summary=False)
    metrics = trainer.validate(lit, DataLoader(batches, batch_size=None), verbose=False)[0]

    # translation_weight = shape_weight = 1: the two parts add up to rec_s exactly
    total = metrics["val_recon_loss_s_translation"] + metrics["val_recon_loss_s_shape"]
    assert abs(total - metrics["val_recon_loss_s"]) < 1e-4 * metrics["val_recon_loss_s"]

    lit.eval()
    with torch.no_grad():
        centroid_err = [((b["s_t"].mean(-2) - lit(b["s_t"])[2].mean(-2)) ** 2).sum(-1).mean() for b in batches]
    expected = torch.stack(centroid_err).mean().item()
    assert abs(metrics["val_recon_loss_s_translation"] - expected) < 1e-4 * expected
