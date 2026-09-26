'''
--batch_norm: without batch norm in the encoder (whose ParallelBatchNorm1d has per-frame parameters),
an encoder with the transformer aggregator works for any number of frames -- e.g. to fine-tune a
10-frame model on 50 frames.
'''
import sys

import pytest
import torch
from easydict import EasyDict

sys.path.insert(0, "cardiac_motion")
from models.Model3D import Encoder3DMesh
from models.Model4D import EncoderTemporalSequence
from models.TemporalAggregators import TransformerAggregator
from test_training import load_fixture, N_FEATURES, LATENT_DIM, BATCH_SIZE, FILTERS


def _encoder(batch_normalization, n_timeframes):
    A, D, U, n_nodes = load_fixture()
    encoder3d = Encoder3DMesh(
        phase_input=False, num_conv_filters_enc=FILTERS, num_features=N_FEATURES,
        cheb_polynomial_order=[3] * len(FILTERS), n_layers=len(FILTERS), n_nodes=n_nodes,
        is_variational=False, template=EasyDict({"v": torch.zeros(n_nodes[0], N_FEATURES).numpy()}),
        adjacency_matrices=A, downsample_matrices=D, latent_dim=None,
        n_timeframes=n_timeframes, batch_normalization=batch_normalization,
    )
    h = encoder3d.forward_conv_stack(torch.zeros(1, n_timeframes, n_nodes[0], N_FEATURES), preserve_graph_structure=False)
    aggregator = TransformerAggregator(features_in=h.shape[-1], features_out=LATENT_DIM, n_timeframes=n_timeframes,
                                       d_model=16, n_heads=2, n_layers=1, d_ff=32)
    return EncoderTemporalSequence(encoder3d=encoder3d, z_aggr_function=aggregator, is_variational=False), n_nodes


def test_encoder_without_batch_norm_is_independent_of_n_timeframes():
    encoder, n_nodes = _encoder(batch_normalization=False, n_timeframes=10)
    encoder.eval()
    for T in (10, 50, 7):
        z = encoder(torch.randn(BATCH_SIZE, T, n_nodes[0], N_FEATURES))
        assert z.mu.shape == (BATCH_SIZE, LATENT_DIM)
    assert not any("batch_normalization" in k for k in encoder.state_dict())


def test_per_frame_batch_norm_ties_encoder_to_n_timeframes():
    encoder, n_nodes = _encoder(batch_normalization=True, n_timeframes=10)
    encoder.eval()
    encoder(torch.randn(BATCH_SIZE, 10, n_nodes[0], N_FEATURES))
    with pytest.raises(RuntimeError):
        encoder(torch.randn(BATCH_SIZE, 50, n_nodes[0], N_FEATURES))


def test_shared_batch_norm_pools_frames_and_works_for_any_number_of_frames():
    A, D, U, n_nodes = load_fixture()
    encoder3d = Encoder3DMesh(
        phase_input=False, num_conv_filters_enc=FILTERS, num_features=N_FEATURES,
        cheb_polynomial_order=[3] * len(FILTERS), n_layers=len(FILTERS), n_nodes=n_nodes,
        is_variational=False, template=EasyDict({"v": torch.zeros(n_nodes[0], N_FEATURES).numpy()}),
        adjacency_matrices=A, downsample_matrices=D, latent_dim=None,
        n_timeframes=10, batch_normalization=True, batch_norm_across_time=True,
    )
    bn = encoder3d.layers["layer_0"]["batch_normalization"].batch_norm
    assert bn.num_features == FILTERS[0]  # per channel, not frames x channels

    # training mode: each channel normalized with statistics over batch, frames and vertices together
    x = torch.randn(BATCH_SIZE, 6, n_nodes[0], FILTERS[0]) * 3 + 5
    y = encoder3d.layers["layer_0"]["batch_normalization"](x)
    torch.testing.assert_close(y.mean(dim=(0, 1, 2)), torch.zeros(FILTERS[0]), atol=1e-4, rtol=0)
    torch.testing.assert_close(y.var(dim=(0, 1, 2), unbiased=False), torch.ones(FILTERS[0]), atol=1e-3, rtol=0)

    encoder3d.eval()
    for T in (10, 50, 3):
        out = encoder3d.forward_conv_stack(torch.randn(BATCH_SIZE, T, n_nodes[0], N_FEATURES), preserve_graph_structure=False)
        assert out.shape[:2] == (BATCH_SIZE, T)
