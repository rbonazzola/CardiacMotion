'''
Wall thickness term (--w_thickness): nearest epi/endo vertex pairs computed on a template, and a
loss on the difference between real and reconstructed epi-endo distances.
'''
import sys

import numpy as np
import pytest
import pytorch_lightning as pl
import torch
from easydict import EasyDict
from torch.utils.data import DataLoader

sys.path.insert(0, "cardiac_motion")
from lightning_modules.ComaLightningModule import (
    CoMA_Lightning, aggregate_thickness, build_wall_thickness_pairs, wall_thickness_distances,
)


def _shells(n=200, r_endo=1.0, r_epi=1.5, seed=0):
    # endo and epi: concentric spheres sampled along the same directions, plus some "base" vertices
    rng = np.random.default_rng(seed)
    directions = rng.normal(size=(n, 3))
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)
    base = rng.normal(size=(10, 3)) * 5
    vertices = np.concatenate([r_epi * directions, r_endo * directions, base])
    labels = np.array(["epi"] * n + ["endo"] * n + ["base"] * 10)
    return vertices, labels, n


def test_pairs_match_radial_neighbours_and_cover_both_surfaces():
    vertices, labels, n = _shells()
    pairs = build_wall_thickness_pairs(vertices, labels)

    assert set(pairs[:, 0].tolist()) == set(range(n))              # every epi vertex paired
    assert set(pairs[:, 1].tolist()) == set(range(n, 2 * n))       # every endo vertex paired
    assert len(torch.unique(pairs, dim=0)) == len(pairs)
    # concentric shells: nearest neighbour across the wall is the radial one
    assert torch.equal(pairs[:, 1], pairs[:, 0] + n)
    distances = wall_thickness_distances(torch.as_tensor(vertices), pairs)
    torch.testing.assert_close(distances, torch.full_like(distances, 0.5))


def test_pairs_need_both_surfaces():
    vertices, labels, _ = _shells()
    with pytest.raises(ValueError):
        build_wall_thickness_pairs(vertices, np.where(labels == "endo", "base", labels))


def _thickness_error(real, recon, pairs):
    return ((wall_thickness_distances(recon, pairs) - wall_thickness_distances(real, pairs)) ** 2).mean()


def test_thickness_loss_invariant_to_rigid_motion_and_sensitive_to_thinning():
    vertices, labels, n = _shells()
    pairs = build_wall_thickness_pairs(vertices, labels)
    real = torch.as_tensor(vertices).expand(2, 3, -1, -1)  # (B, T, V, 3)

    angle = torch.tensor(0.7, dtype=torch.float64)
    rotation = torch.tensor([[torch.cos(angle), -torch.sin(angle), 0], [torch.sin(angle), torch.cos(angle), 0], [0, 0, 1]],
                            dtype=torch.float64)
    moved = real @ rotation.T + torch.tensor([3.0, -1.0, 2.0], dtype=torch.float64)
    assert _thickness_error(real, moved, pairs) < 1e-20

    thinned = real.clone()
    thinned[..., n:2 * n, :] *= 1.2  # endo pushed out: wall 0.5 -> 0.3
    torch.testing.assert_close(_thickness_error(real, thinned, pairs), torch.tensor(0.04, dtype=torch.float64))


def test_lightning_module_adds_weighted_thickness_to_val_loss():
    from test_training import build_model, load_fixture, make_batch

    A, D, U, n_nodes = load_fixture()
    model = build_model(A, D, U, n_nodes)
    rng = np.random.default_rng(0)
    pairs = torch.as_tensor(rng.choice(n_nodes[0], size=(50, 2), replace=False))

    def lit_with(w_thickness):
        loss_params = EasyDict({
            "reconstruction_c": EasyDict({"type": "mse", "weight": 1.0}),
            "reconstruction_s": EasyDict({"weight": 1.0}),
            "regularization": EasyDict({"weight": 0.0}),
            "thickness": EasyDict({"weight": w_thickness}),
        })
        optimizer_params = EasyDict({"algorithm": "Adam", "parameters": EasyDict({"lr": 1e-3})})
        return CoMA_Lightning(model=model, loss_params=loss_params, optimizer_params=optimizer_params,
                              additional_params=EasyDict({}), thickness_pairs=pairs)

    with pytest.raises(ValueError):
        CoMA_Lightning(model=model, loss_params=EasyDict({**lit_with(0).loss_params, "thickness": EasyDict({"weight": 1.0})}),
                       optimizer_params=EasyDict({"algorithm": "Adam", "parameters": EasyDict({"lr": 1e-3})}),
                       additional_params=EasyDict({}))

    torch.manual_seed(0)
    batches = [make_batch(n_nodes[0]) for _ in range(2)]
    loader = DataLoader(batches, batch_size=None)
    trainer = pl.Trainer(logger=False, accelerator="cpu", enable_progress_bar=False, enable_model_summary=False)

    m0 = trainer.validate(lit_with(0.0), loader, verbose=False)[0]
    m2 = trainer.validate(lit_with(2.0), loader, verbose=False)[0]
    assert m0["val_thickness_loss"] > 0
    assert abs(m2["val_loss"] - (m0["val_loss"] + 2.0 * m0["val_thickness_loss"])) < 1e-4 * m2["val_loss"]

    model.eval()
    with torch.no_grad():
        real = torch.cat([wall_thickness_distances(b["s_t"], pairs).flatten() for b in batches])
        recon = torch.cat([wall_thickness_distances(model(b["s_t"])[2], pairs).flatten() for b in batches])
    errors = recon - real
    assert abs(m0["val_thickness_mae"] - errors.abs().mean().item()) < 1e-4 * errors.abs().mean().item()
    rel_err = errors.abs().sum() / real.sum()
    assert abs(m0["val_thickness_rel_err"] - rel_err.item()) < 1e-4 * rel_err.item()
    n_batches = len(batches)
    real_btp = real.reshape(n_batches * batches[0]["s_t"].shape[0], batches[0]["s_t"].shape[1], -1)
    errors_btp = errors.reshape(real_btp.shape)
    nrmse = torch.sqrt((errors_btp ** 2).sum() / ((real_btp - real_btp.mean(0)) ** 2).sum())
    assert abs(m0["val_thickness_nrmse"] - nrmse.item()) < 1e-3 * nrmse.item()
    assert "thickness_pairs" not in lit_with(1.0).state_dict()  # not saved in checkpoints


def _batch_stats(real, recon):
    """Per-batch sums as CoMA_Lightning._wall_thickness_terms computes them; real/recon: (B, T, P)."""
    error = recon - real
    return {
        "thickness_loss": (error ** 2).mean(), "thickness_count": torch.tensor(float(error.numel())),
        "thickness_abs_err_sum": error.abs().sum(), "thickness_real_sum": real.sum(),
        "thickness_sse": (error.double() ** 2).sum(), "thickness_n_subjects": torch.tensor(float(real.shape[0]), dtype=torch.float64),
        "thickness_sum_d": real.double().sum(0), "thickness_sum_d2": (real.double() ** 2).sum(0),
    }


def test_aggregate_thickness_relative_error_is_ratio_of_totals():
    # batch 1: walls of 4 mm off by 1 mm; batch 2: walls of 8 mm off by 1 mm
    real = [torch.full((2, 1, 5), 4.0), torch.full((2, 1, 5), 8.0)]
    outputs = [_batch_stats(r, r + 1.0) for r in real]
    agg = aggregate_thickness(outputs)
    assert abs(agg["mae"].item() - 1.0) < 1e-6
    assert abs(agg["rel_err"].item() - 20.0 / 120.0) < 1e-6  # not mean(0.25, 0.125)

    no_pairs = [{"thickness_loss": torch.tensor(0.0), "thickness_count": torch.tensor(0.0)}]
    agg = aggregate_thickness(no_pairs)
    assert all(torch.isnan(agg[k]) for k in ("mae", "rel_err", "nrmse"))


def test_thickness_nrmse_against_population_mean_curve():
    torch.manual_seed(0)
    real = 8.0 + torch.randn(30, 4, 20, dtype=torch.float64)  # (subjects, frames, pairs)
    batches = [real[:12], real[12:]]                            # epoch split in batches of different size
    population_curve = real.mean(0, keepdim=True)

    perfect = aggregate_thickness([_batch_stats(r, r) for r in batches])
    mean_curve = aggregate_thickness([_batch_stats(r, population_curve.expand_as(r)) for r in batches])
    half_way = aggregate_thickness([_batch_stats(r, (r + population_curve) / 2) for r in batches])

    assert perfect["nrmse"].item() == 0.0
    assert abs(mean_curve["nrmse"].item() - 1.0) < 1e-9  # predicting the population curve scores 1
    assert abs(half_way["nrmse"].item() - 0.5) < 1e-9


def test_thickness_metrics_not_logged_without_pairs():
    from test_training import build_model, load_fixture, make_batch, make_lit

    A, D, U, n_nodes = load_fixture()
    lit = make_lit(build_model(A, D, U, n_nodes))  # no thickness pairs
    torch.manual_seed(0)
    loader = DataLoader([make_batch(n_nodes[0])], batch_size=None)
    trainer = pl.Trainer(logger=False, accelerator="cpu", enable_progress_bar=False, enable_model_summary=False)
    metrics = trainer.validate(lit, loader, verbose=False)[0]
    assert not any(k in metrics for k in ("val_thickness_mae", "val_thickness_rel_err", "val_thickness_nrmse"))
    assert metrics["val_thickness_loss"] == 0.0
