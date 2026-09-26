'''
The style decoder's translation head: a single zero-initialized Linear by default (same state
dict keys as before, so older checkpoints load), or an MLP with --translation_head_hidden.
'''
import sys

import torch
from easydict import EasyDict

sys.path.insert(0, "cardiac_motion")
from models.Model4D import DecoderStyle
from test_training import load_fixture, N_FEATURES, LATENT_DIM_C, LATENT_DIM_S, N_TIMEFRAMES, BATCH_SIZE, FILTERS


def _decoder_style(**kwargs):
    A, D, U, n_nodes = load_fixture()
    template = EasyDict({"v": torch.zeros(n_nodes[0], N_FEATURES).numpy()})
    decoder = DecoderStyle({
        "num_features": N_FEATURES, "n_layers": len(FILTERS), "n_nodes": n_nodes,
        "cheb_polynomial_order": [6] * len(FILTERS), "is_variational": False, "template": template,
        "adjacency_matrices": A, "upsample_matrices": U,
        "num_conv_filters_dec_s": FILTERS, "latent_dim_content": LATENT_DIM_C, "latent_dim_style": LATENT_DIM_S,
    }, phase_embedding_method="exp_v1", n_timeframes=N_TIMEFRAMES, **kwargs)
    return decoder, n_nodes


def test_default_head_is_the_original_linear():
    decoder, _ = _decoder_style(translation_head=True)
    assert isinstance(decoder.translation_head, torch.nn.Linear)
    keys = [k for k in decoder.state_dict() if k.startswith("translation_head.")]
    assert sorted(keys) == ["translation_head.bias", "translation_head.weight"]  # older checkpoints still load


def test_mlp_head_structure_and_zero_init():
    decoder, n_nodes = _decoder_style(translation_head=True, translation_head_hidden=[32, 16])
    head = decoder.translation_head
    linears = [m for m in head if isinstance(m, torch.nn.Linear)]
    assert [(l.in_features, l.out_features) for l in linears] == [(LATENT_DIM_C + 2 * LATENT_DIM_S, 32), (32, 16), (16, 3)]
    assert sum(isinstance(m, torch.nn.ReLU) for m in head) == 2
    assert torch.all(linears[-1].weight == 0) and torch.all(linears[-1].bias == 0)

    # starts as a no-op: same output as the decoder without a head
    z_c, z_s = torch.randn(BATCH_SIZE, LATENT_DIM_C), torch.randn(BATCH_SIZE, LATENT_DIM_S)
    no_head, _ = _decoder_style(translation_head=False)
    no_head.decoder_3d.load_state_dict(decoder.decoder_3d.state_dict())
    torch.testing.assert_close(decoder(z_c, z_s, N_TIMEFRAMES), no_head(z_c, z_s, N_TIMEFRAMES))


def test_mlp_head_learns_nonlinear_translation():
    # gradients reach every layer once the output layer is no longer zero
    decoder, _ = _decoder_style(translation_head=True, translation_head_hidden=[16])
    z_c, z_s = torch.randn(BATCH_SIZE, LATENT_DIM_C), torch.randn(BATCH_SIZE, LATENT_DIM_S)
    torch.nn.init.normal_(decoder.translation_head[-1].weight)
    decoder(z_c, z_s, N_TIMEFRAMES).sum().backward()
    assert all(p.grad is not None and p.grad.abs().sum() > 0 for p in decoder.translation_head.parameters())
