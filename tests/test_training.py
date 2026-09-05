"""
Integration test for the training loop.

Uses the mesh topology cached in tests/data/cached/matrices/72467.pkl
and synthetic random mesh data — no real cardiac data required.

Run with:  python -m unittest tests/test_training.py -v
"""
import sys
import pickle as pkl
import unittest
import torch
from easydict import EasyDict

sys.path.insert(0, "cardiac_motion")

from models.Model3D import Encoder3DMesh, Decoder3DMesh
from models.Model4D import (
    AutoencoderTemporalSequence,
    EncoderTemporalSequence,
    DecoderTemporalSequence,
    DecoderContent,
    DecoderStyle,
)
from models.TemporalAggregators import FCN_Aggregator
from lightning_modules.ComaLightningModule import CoMA_Lightning


FIXTURE      = "tests/data/cached/matrices/72467.pkl"
N_FEATURES   = 3
LATENT_DIM_C = 4
LATENT_DIM_S = 4
LATENT_DIM   = LATENT_DIM_C + LATENT_DIM_S
N_TIMEFRAMES = 4
BATCH_SIZE   = 2
FILTERS      = [8, 8, 8, 8]


def load_fixture():
    A, D, U, n_nodes = pkl.load(open(FIXTURE, "rb"))
    return A, D, U, n_nodes


def build_model(A, D, U, n_nodes):
    template = EasyDict({"v": torch.zeros(n_nodes[0], N_FEATURES).numpy()})

    encoder3d = Encoder3DMesh(
        phase_input=False,
        num_conv_filters_enc=FILTERS,
        num_features=N_FEATURES,
        cheb_polynomial_order=[6] * len(FILTERS),
        n_layers=len(FILTERS),
        n_nodes=n_nodes,
        is_variational=False,
        template=template,
        adjacency_matrices=A,
        downsample_matrices=D,
        latent_dim=None,
        n_timeframes=N_TIMEFRAMES,
    )

    x_example = torch.zeros(1, N_TIMEFRAMES, n_nodes[0], N_FEATURES)
    h = encoder3d.forward_conv_stack(x_example, preserve_graph_structure=False)
    features_in = N_TIMEFRAMES * h.shape[-1]

    encoder = EncoderTemporalSequence(
        encoder3d=encoder3d,
        z_aggr_function=FCN_Aggregator(features_in=features_in, features_out=LATENT_DIM),
        is_variational=False,
    )

    common = dict(
        num_features=N_FEATURES,
        n_layers=len(FILTERS),
        n_nodes=n_nodes,
        cheb_polynomial_order=[6] * len(FILTERS),
        is_variational=False,
        template=template,
        adjacency_matrices=A,
        upsample_matrices=U,
    )

    decoder_content = DecoderContent({
        **common,
        "num_conv_filters_dec_c": FILTERS,
        "latent_dim_content": LATENT_DIM_C,
    })

    decoder_style = DecoderStyle({
        **common,
        "num_conv_filters_dec_s": FILTERS,
        "latent_dim_content": LATENT_DIM_C,
        "latent_dim_style": LATENT_DIM_S,
    }, phase_embedding_method="exp_v1", n_timeframes=N_TIMEFRAMES)

    decoder = DecoderTemporalSequence(
        decoder_content=decoder_content,
        decoder_style=decoder_style,
        is_variational=False,
    )

    return AutoencoderTemporalSequence(encoder=encoder, decoder=decoder, is_variational=False)


def make_batch(n_nodes_0):
    s_t      = torch.randn(BATCH_SIZE, N_TIMEFRAMES, n_nodes_0, N_FEATURES)
    time_avg = s_t.mean(dim=1)
    d_content = torch.rand(BATCH_SIZE, N_TIMEFRAMES) + 1e-6
    d_style   = torch.rand(BATCH_SIZE, N_TIMEFRAMES) + 1e-6
    return {
        "s_t":        s_t,
        "time_avg_s": time_avg,
        "d_content":  d_content,
        "d_style":    d_style,
    }


def make_lit(model):
    loss_params = EasyDict({
        "reconstruction_c": EasyDict({"type": "mse", "weight": 1.0}),
        "reconstruction_s": EasyDict({"weight": 1.0}),
        "regularization":   EasyDict({"weight": 0.0}),
    })
    optimizer_params = EasyDict({
        "algorithm": "Adam",
        "parameters": EasyDict({"lr": 1e-3}),
    })
    return CoMA_Lightning(
        model=model,
        loss_params=loss_params,
        optimizer_params=optimizer_params,
        additional_params=EasyDict({}),
    )


class TestTrainingStep(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        A, D, U, n_nodes = load_fixture()
        cls.n_nodes = n_nodes
        cls.model   = build_model(A, D, U, n_nodes)
        cls.lit     = make_lit(cls.model)
        cls.lit.model.set_mode("training")

    def test_forward_returns_correct_shapes(self):
        batch = make_batch(self.n_nodes[0])
        z, avg_shape, shape_t = self.model(batch["s_t"])
        self.assertIn("mu", z)
        self.assertEqual(avg_shape.shape, (BATCH_SIZE, self.n_nodes[0], N_FEATURES))
        self.assertEqual(shape_t.shape,   (BATCH_SIZE, N_TIMEFRAMES, self.n_nodes[0], N_FEATURES))

    def test_training_step_returns_finite_loss(self):
        batch = make_batch(self.n_nodes[0])
        loss_dict = self.lit.training_step(batch, batch_idx=0)
        self.assertTrue(torch.isfinite(loss_dict["loss"]))

    def test_loss_decreases_over_steps(self):
        optimizer = self.lit.configure_optimizers()
        losses = []
        for step in range(15):
            optimizer.zero_grad()
            loss_dict = self.lit.training_step(make_batch(self.n_nodes[0]), batch_idx=step)
            loss_dict["loss"].backward()
            optimizer.step()
            losses.append(loss_dict["loss"].item())

        self.assertLess(losses[-1], losses[0],
                        f"Loss did not decrease: first={losses[0]:.4f}, last={losses[-1]:.4f}")

    def test_validation_step_returns_finite_loss(self):
        batch = make_batch(self.n_nodes[0])
        loss_dict = self.lit.validation_step(batch, batch_idx=0)
        self.assertTrue(torch.isfinite(loss_dict["val_loss"]))

    def test_validation_step_with_none_d_style(self):
        batch = make_batch(self.n_nodes[0])
        batch["d_style"] = None
        loss_dict = self.lit.validation_step(batch, batch_idx=0)
        self.assertTrue(torch.isfinite(loss_dict["val_loss"]),
                        "val_loss debe ser finito incluso cuando d_style es None")


if __name__ == "__main__":
    unittest.main()
