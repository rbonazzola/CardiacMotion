'''
Fourier harmonics in the style decoder's phase embedding (--n_harmonics):
K = 1 must reproduce the original embedding exactly, and K > 1 must append
[sin(k theta) z, cos(k theta) z] blocks and widen the style decoder's input accordingly.
'''
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, "cardiac_motion")
from models.PhaseModule import PhaseTensor
from models.Model4D import DecoderStyle
from test_training import load_fixture, N_FEATURES, LATENT_DIM_C, LATENT_DIM_S, N_TIMEFRAMES, BATCH_SIZE, FILTERS

N, T, M = 2, 10, 4


def _legacy_version_1(x):
    # Verbatim logic of PhaseTensor(version_1) before harmonics were added
    sen_t = []; cos_t = []
    n_timeframes, rank = x.shape[1], x.dim()
    for i in range(n_timeframes):
        phase = 2 * np.pi * i / n_timeframes
        sen_t.append(np.sin(phase))
        cos_t.append(np.cos(phase))
    dims_to_expand = list(range(rank))
    dims_to_expand.remove(1)
    dims_to_expand = tuple(dims_to_expand)
    sen_t = torch.Tensor(np.expand_dims(np.array(sen_t), axis=dims_to_expand)).type_as(x)
    cos_t = torch.Tensor(np.expand_dims(np.array(cos_t), axis=dims_to_expand)).type_as(x)
    return torch.cat((sen_t * x, cos_t * x), dim=-1)


def test_single_harmonic_matches_legacy_exactly():
    x = torch.randn(N, T, M)
    assert torch.equal(PhaseTensor("version_1")(x), _legacy_version_1(x))
    assert torch.equal(PhaseTensor("version_1", n_harmonics=1)(x), _legacy_version_1(x))


@pytest.mark.parametrize("K", [2, 3])
def test_harmonic_blocks(K):
    x = torch.randn(N, T, M)
    out = PhaseTensor("version_1", n_harmonics=K)(x)
    assert out.shape == (N, T, 2 * K * M)

    theta = 2 * torch.pi * torch.arange(T) / T
    for k in range(1, K + 1):
        sin_block = out[..., (2 * k - 2) * M:(2 * k - 1) * M]
        cos_block = out[..., (2 * k - 1) * M:(2 * k) * M]
        torch.testing.assert_close(sin_block, torch.sin(k * theta)[None, :, None] * x)
        torch.testing.assert_close(cos_block, torch.cos(k * theta)[None, :, None] * x)


def test_invalid_configurations():
    with pytest.raises(ValueError):
        PhaseTensor("version_1", n_harmonics=0)
    with pytest.raises(NotImplementedError):
        PhaseTensor("version_2", n_harmonics=2)


@pytest.mark.parametrize("K", [1, 3])
def test_style_decoder_with_harmonics(K):
    from easydict import EasyDict

    A, D, U, n_nodes = load_fixture()
    template = EasyDict({"v": torch.zeros(n_nodes[0], N_FEATURES).numpy()})
    decoder_style = DecoderStyle({
        "num_features": N_FEATURES, "n_layers": len(FILTERS), "n_nodes": n_nodes,
        "cheb_polynomial_order": [6] * len(FILTERS), "is_variational": False, "template": template,
        "adjacency_matrices": A, "upsample_matrices": U,
        "num_conv_filters_dec_s": FILTERS, "latent_dim_content": LATENT_DIM_C, "latent_dim_style": LATENT_DIM_S,
    }, phase_embedding_method="exp_v1", n_timeframes=N_TIMEFRAMES, translation_head=True, n_harmonics=K)

    assert decoder_style.translation_head.in_features == LATENT_DIM_C + 2 * K * LATENT_DIM_S

    z_c = torch.randn(BATCH_SIZE, LATENT_DIM_C)
    z_s = torch.randn(BATCH_SIZE, LATENT_DIM_S)
    out = decoder_style(z_c, z_s, N_TIMEFRAMES)
    assert out.shape == (BATCH_SIZE, N_TIMEFRAMES, n_nodes[0], N_FEATURES)
