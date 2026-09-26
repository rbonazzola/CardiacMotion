'''
--init_from_checkpoint: initialize a model from a pretrained checkpoint, possibly trained with a
different number of timeframes (the encoder's per-frame batch norm is remapped by cardiac phase).
'''
import sys
from collections import OrderedDict

import pytest
import torch
from easydict import EasyDict

sys.path.insert(0, "cardiac_motion")
from models.Model3D import Encoder3DMesh
from models.Model4D import (AutoencoderTemporalSequence, DecoderContent, DecoderStyle, DecoderTemporalSequence,
                            EncoderTemporalSequence)
from models.TemporalAggregators import FCN_Aggregator, TransformerAggregator
from utils.pretrained import load_pretrained, phase_index_map, remap_per_frame_batch_norm, resolve_checkpoint
from test_training import load_fixture, N_FEATURES, LATENT_DIM_C, LATENT_DIM_S, LATENT_DIM, BATCH_SIZE, FILTERS


def _model(n_timeframes, aggregator="transformer"):
    A, D, U, n_nodes = load_fixture()
    template = EasyDict({"v": torch.zeros(n_nodes[0], N_FEATURES).numpy()})
    common = dict(num_features=N_FEATURES, n_layers=len(FILTERS), n_nodes=n_nodes, cheb_polynomial_order=[3] * len(FILTERS),
                  is_variational=False, template=template, adjacency_matrices=A)
    encoder3d = Encoder3DMesh(phase_input=False, num_conv_filters_enc=FILTERS, downsample_matrices=D, latent_dim=None,
                              n_timeframes=n_timeframes, **common)
    h = encoder3d.forward_conv_stack(torch.zeros(1, n_timeframes, n_nodes[0], N_FEATURES), preserve_graph_structure=False)
    z_aggr = (TransformerAggregator(features_in=h.shape[-1], features_out=LATENT_DIM, n_timeframes=n_timeframes,
                                    d_model=16, n_heads=2, n_layers=1, d_ff=32)
              if aggregator == "transformer" else FCN_Aggregator(features_in=n_timeframes * h.shape[-1], features_out=LATENT_DIM))
    decoder = DecoderTemporalSequence(
        decoder_content=DecoderContent({**common, "upsample_matrices": U, "num_conv_filters_dec_c": FILTERS,
                                        "latent_dim_content": LATENT_DIM_C}),
        decoder_style=DecoderStyle({**common, "upsample_matrices": U, "num_conv_filters_dec_s": FILTERS,
                                    "latent_dim_content": LATENT_DIM_C, "latent_dim_style": LATENT_DIM_S},
                                   phase_embedding_method="exp_v1", n_timeframes=n_timeframes, translation_head=True, n_harmonics=2),
        is_variational=False)
    encoder = EncoderTemporalSequence(encoder3d=encoder3d, z_aggr_function=z_aggr, is_variational=False)
    return AutoencoderTemporalSequence(encoder=encoder, decoder=decoder, is_variational=False), n_nodes


def _save_lightning_checkpoint(model, path):
    # as saved by CoMA_Lightning: keys prefixed with "model."
    torch.manual_seed(0)
    with torch.no_grad():
        for p in model.parameters():
            p.normal_()
        for name, buffer in model.named_buffers():
            if "running" in name:
                buffer.uniform_(0.5, 2.0)
    # keep the per-module version metadata, as Lightning does (ChebConv_Coma relies on it to tell
    # current checkpoints from legacy v1 ones)
    state_dict = model.state_dict()
    prefixed = OrderedDict(("model." + k, v) for k, v in state_dict.items())
    prefixed._metadata = OrderedDict((("model." + k) if k else "model", v) for k, v in state_dict._metadata.items())
    torch.save({"state_dict": prefixed}, path)


def test_phase_index_map_10_to_50():
    mapping = phase_index_map(10, 50)
    assert mapping[::5].tolist() == list(range(10))  # frames at the same phase keep their own parameters
    assert mapping[:3].tolist() == [0, 0, 0] and mapping[3:8].tolist() == [1] * 5
    assert mapping[-2:].tolist() == [0, 0]  # circular: the last frames are closest to phase 0


def test_remap_per_frame_batch_norm_layout():
    n_channels = 3
    old = torch.arange(10 * n_channels, dtype=torch.float32)  # value = frame * C + channel
    new = remap_per_frame_batch_norm(old, 10, 50).reshape(50, n_channels)
    for frame in range(50):
        torch.testing.assert_close(new[frame], old.reshape(10, n_channels)[phase_index_map(10, 50)[frame]])


def test_load_pretrained_same_and_different_number_of_frames(tmp_path):
    pretrained, n_nodes = _model(n_timeframes=4)
    ckpt = tmp_path / "pretrained.ckpt"
    _save_lightning_checkpoint(pretrained, ckpt)
    old = pretrained.state_dict()

    same, _ = _model(n_timeframes=4)
    load_pretrained(same, str(ckpt), n_timeframes=4)
    for k, v in same.state_dict().items():
        torch.testing.assert_close(v, old[k])

    finetune, _ = _model(n_timeframes=8)
    info = load_pretrained(finetune, str(ckpt), n_timeframes=8)
    assert info["n_timeframes_pretrained"] == 4
    remapped = 0
    for k, v in finetune.state_dict().items():
        if k.startswith("encoder.encoder_3d_mesh.layers.") and "batch_norm." in k and "num_batches" not in k:
            torch.testing.assert_close(v, remap_per_frame_batch_norm(old[k], 4, 8))
            remapped += 1
        else:
            torch.testing.assert_close(v, old[k])  # everything else, decoders' batch norm included, unchanged
    assert remapped == 4 * len(FILTERS)  # weight, bias, running mean and var of every encoder layer

    finetune.eval()
    z, avg_s, shat_t = finetune(torch.randn(BATCH_SIZE, 8, n_nodes[0], N_FEATURES))
    assert shat_t.shape == (BATCH_SIZE, 8, n_nodes[0], N_FEATURES)


def test_load_pretrained_rejects_fcn_and_mismatched_architectures(tmp_path):
    fcn, _ = _model(n_timeframes=4, aggregator="fcn")
    ckpt = tmp_path / "fcn.ckpt"
    _save_lightning_checkpoint(fcn, ckpt)
    with pytest.raises(ValueError, match="FCN"):
        load_pretrained(_model(n_timeframes=8, aggregator="fcn")[0], str(ckpt), n_timeframes=8)

    transformer_ckpt = tmp_path / "transformer.ckpt"
    _save_lightning_checkpoint(_model(n_timeframes=4)[0], transformer_ckpt)
    with pytest.raises(ValueError, match="differ"):
        load_pretrained(fcn, str(transformer_ckpt), n_timeframes=4)  # different aggregator


def test_resolve_checkpoint_from_mlflow_run_id(tmp_path):
    from mlflow.tracking import MlflowClient
    tracking_uri = str(tmp_path / "mlruns")
    client = MlflowClient(tracking_uri=tracking_uri)
    run = client.create_run(client.create_experiment("test"))
    artifacts = tmp_path / "mlruns" / run.info.experiment_id / run.info.run_id / "artifacts" / "checkpoints"
    artifacts.mkdir(parents=True)
    (artifacts / "best_model.ckpt").write_bytes(b"")
    assert resolve_checkpoint(run.info.run_id, tracking_uri) == str(artifacts / "best_model.ckpt")
    with pytest.raises(FileNotFoundError):
        resolve_checkpoint("not-a-run-nor-a-file", tracking_uri)
