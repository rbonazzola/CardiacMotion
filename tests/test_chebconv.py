'''
Tests for ChebConv_Coma: the forward must use every Chebyshev term (lins.0 ... lins.K-1),
with or without bias, and legacy (v1) checkpoints must load so that outputs are unchanged.
'''
import itertools
import warnings
from collections import OrderedDict

import pytest
import torch

from cardiac_motion.models.layers import ChebConv_Coma

N_NODES, IN_CH, OUT_CH, BATCH = 10, 4, 5, 2


def _graph():
    # ring graph
    src = torch.arange(N_NODES)
    dst = (src + 1) % N_NODES
    edge_index = torch.cat([torch.stack([src, dst]), torch.stack([dst, src])], dim=1)
    edge_index, norm = ChebConv_Coma.norm(edge_index, N_NODES)
    return edge_index, norm


def _make_layer(K, bias, seed=0):
    torch.manual_seed(seed)
    layer = ChebConv_Coma(IN_CH, OUT_CH, K)
    for lin in layer.lins:
        torch.nn.init.normal_(lin.weight)
    if bias:
        torch.nn.init.normal_(layer.bias)
    else:
        layer.bias = None  # as done in Decoder3DMesh for the last layer
    return layer


def _chebyshev_terms(layer, x, edge_index, norm, n_terms):
    Tx = [x]
    if n_terms > 1:
        Tx.append(layer.propagate(edge_index, x=x, norm=norm))
    for _ in range(2, n_terms):
        Tx.append(2 * layer.propagate(edge_index, x=Tx[-1], norm=norm) - Tx[-2])
    return Tx


def _reference_forward(layer, x, edge_index, norm):
    Tx = _chebyshev_terms(layer, x, edge_index, norm, len(layer.lins))
    out = sum(t @ lin.weight.t() for t, lin in zip(Tx, layer.lins))
    return out + layer.bias if layer.bias is not None else out


def _legacy_forward(layer, x, edge_index, norm):
    # Verbatim logic of the v1 forward: weights taken by position from self.parameters()[1:7]
    weights = []
    for i in range(1, 7):
        try:
            weights.append(next(itertools.islice(layer.parameters(), i, None)).t())
        except StopIteration:
            pass
    Tx = _chebyshev_terms(layer, x, edge_index, norm, len(weights))
    out = sum(t @ w for t, w in zip(Tx, weights))
    return out + layer.bias if layer.bias is not None else out


def _legacy_state_dict(layer):
    sd = layer.state_dict()
    sd._metadata[""]["version"] = 1
    return sd


@pytest.mark.parametrize("K", [1, 3, 6, 10])
@pytest.mark.parametrize("bias", [True, False])
def test_forward_uses_all_chebyshev_terms(K, bias):
    layer = _make_layer(K, bias)
    edge_index, norm = _graph()
    x = torch.randn(BATCH, N_NODES, IN_CH)
    torch.testing.assert_close(layer(x, edge_index, norm), _reference_forward(layer, x, edge_index, norm))


@pytest.mark.parametrize("bias", [True, False])
def test_every_chebyshev_weight_receives_gradient(bias):
    layer = _make_layer(3, bias)
    edge_index, norm = _graph()
    layer(torch.randn(BATCH, N_NODES, IN_CH), edge_index, norm).sum().backward()
    for j, lin in enumerate(layer.lins):
        assert lin.weight.grad is not None and lin.weight.grad.abs().sum() > 0, f"lins.{j} unused"


@pytest.mark.parametrize("K", [1, 3, 6, 7, 10])
@pytest.mark.parametrize("bias", [True, False])
def test_legacy_checkpoint_reproduces_legacy_output(K, bias):
    if K == 1 and not bias:
        pytest.skip("legacy forward crashed for K=1 without bias (no weights at all)")
    old_layer = _make_layer(K, bias, seed=1)
    edge_index, norm = _graph()
    x = torch.randn(BATCH, N_NODES, IN_CH)
    expected = _legacy_forward(old_layer, x, edge_index, norm)

    new_layer = _make_layer(K, bias, seed=2)
    new_layer.load_state_dict(_legacy_state_dict(old_layer))
    torch.testing.assert_close(new_layer(x, edge_index, norm), expected)


def test_legacy_checkpoint_zeroes_unused_terms():
    old_layer = _make_layer(3, bias=False, seed=1)
    new_layer = _make_layer(3, bias=False, seed=2)
    new_layer.load_state_dict(_legacy_state_dict(old_layer))
    torch.testing.assert_close(new_layer.lins[0].weight, old_layer.lins[1].weight)
    torch.testing.assert_close(new_layer.lins[1].weight, old_layer.lins[2].weight)
    assert torch.all(new_layer.lins[2].weight == 0)


@pytest.mark.parametrize("bias", [True, False])
def test_current_checkpoint_roundtrip_is_unchanged(bias):
    layer = _make_layer(6, bias, seed=1)
    other = _make_layer(6, bias, seed=2)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        other.load_state_dict(layer.state_dict())
    for a, b in zip(layer.lins, other.lins):
        torch.testing.assert_close(a.weight, b.weight)


def test_missing_metadata_is_treated_as_legacy_with_warning():
    old_layer = _make_layer(3, bias=False, seed=1)
    sd = OrderedDict(old_layer.state_dict())  # plain copy, metadata dropped
    new_layer = _make_layer(3, bias=False, seed=2)
    with pytest.warns(UserWarning, match="legacy"):
        new_layer.load_state_dict(sd)
    torch.testing.assert_close(new_layer.lins[0].weight, old_layer.lins[1].weight)


def test_legacy_conversion_inside_nested_module():
    # Prefixes and metadata keys must work when the conv is a submodule, as in the real models
    old = torch.nn.Sequential(OrderedDict(conv=_make_layer(3, bias=False, seed=1)))
    sd = old.state_dict()
    sd._metadata["conv"]["version"] = 1
    new = torch.nn.Sequential(OrderedDict(conv=_make_layer(3, bias=False, seed=2)))
    new.load_state_dict(sd)
    torch.testing.assert_close(new.conv.lins[0].weight, old.conv.lins[1].weight)
    assert torch.all(new.conv.lins[2].weight == 0)
