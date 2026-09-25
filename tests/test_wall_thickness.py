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
    CoMA_Lightning, build_wall_thickness_pairs, wall_thickness_distances,
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
        errors = torch.cat([(wall_thickness_distances(model(b["s_t"])[2], pairs) - wall_thickness_distances(b["s_t"], pairs)).flatten()
                            for b in batches])
    assert abs(m0["val_thickness_mae"] - errors.abs().mean().item()) < 1e-4 * errors.abs().mean().item()
    assert "thickness_pairs" not in lit_with(1.0).state_dict()  # not saved in checkpoints
