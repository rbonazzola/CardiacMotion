'''
Initialize a model from a pretrained checkpoint, possibly trained with a different number of
timeframes (e.g. fine-tune a 10-frame model on 50 frames).

With the transformer aggregator, the only parameters tied to n_timeframes are those of the
encoder's per-frame batch norm (ParallelBatchNorm1d, features laid out as frame * n_channels +
channel). They are remapped by cardiac phase: each new frame takes the parameters/statistics of
the pretrained frame closest in phase (circularly). Everything else is loaded as is.
'''
import logging
import os
import re
from collections import OrderedDict
from urllib.parse import urlparse, unquote

import torch

logger = logging.getLogger(__name__)

_PER_FRAME_BN = re.compile(r"encoder\.encoder_3d_mesh\.layers\.layer_\d+\.batch_normalization\.batch_norm\.(weight|bias|running_mean|running_var)$")
_FCN_AGGREGATOR = re.compile(r"z_aggr_function(_mu|_log_var)?\.fcn\.weight$")


def resolve_checkpoint(spec: str, tracking_uri: str = None) -> str:
    '''
    A path to a .ckpt file, or an MLflow run ID in the store at `tracking_uri`: the run's
    checkpoints/best_model.ckpt, then the older layouts (checkpoints/ next to artifacts/).
    '''
    if os.path.isfile(spec):
        return spec
    from mlflow.tracking import MlflowClient
    try:
        run = MlflowClient(tracking_uri=tracking_uri).get_run(spec)
    except Exception as e:
        raise FileNotFoundError(f"{spec!r} is neither a checkpoint file nor an MLflow run ID ({e})") from e
    artifacts = unquote(urlparse(run.info.artifact_uri).path)
    candidates = [os.path.join(artifacts, "checkpoints", "best_model.ckpt")]
    legacy_dir = os.path.join(os.path.dirname(artifacts), "checkpoints")
    if os.path.isdir(legacy_dir):
        candidates += sorted(os.path.join(legacy_dir, f) for f in os.listdir(legacy_dir) if f.endswith(".ckpt"))
    for path in candidates:
        if os.path.isfile(path):
            return path
    raise FileNotFoundError(f"No checkpoint found for run {spec}; looked at: {candidates}")


def phase_index_map(n_old: int, n_new: int) -> torch.Tensor:
    '''For each of n_new equispaced phases, the index of the closest of n_old phases (circularly).'''
    return torch.round(torch.arange(n_new, dtype=torch.float64) * n_old / n_new).long() % n_old


def remap_per_frame_batch_norm(tensor: torch.Tensor, n_old: int, n_new: int) -> torch.Tensor:
    '''(n_old * C,) -> (n_new * C,), each new frame taking the closest pretrained frame's values.'''
    n_channels, rest = divmod(tensor.numel(), n_old)
    assert rest == 0, f"{tensor.numel()} features are not a multiple of {n_old} frames"
    return tensor.reshape(n_old, n_channels)[phase_index_map(n_old, n_new)].reshape(-1).clone()


def _model_state_dict_from_checkpoint(path: str) -> OrderedDict:
    '''The model's weights from a Lightning checkpoint (keys "model.<...>", maybe "_orig_mod." from torch.compile).'''
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    state_dict = checkpoint.get("state_dict", checkpoint)
    metadata = getattr(state_dict, "_metadata", None)

    def strip(key):
        key = key[len("model."):] if key.startswith("model.") else key
        return key.replace("_orig_mod.", "")

    stripped = OrderedDict((strip(k), v) for k, v in state_dict.items())
    if metadata is not None:  # module versions, e.g. for the legacy ChebConv_Coma conversion
        stripped._metadata = type(metadata)((strip(k), v) for k, v in metadata.items())
    return stripped


def load_pretrained(model: torch.nn.Module, spec: str, n_timeframes: int, tracking_uri: str = None) -> dict:
    '''
    Loads the checkpoint given by `spec` (path or MLflow run ID) into `model`, an
    AutoencoderTemporalSequence built for `n_timeframes` frames, remapping the encoder's
    per-frame batch norm if the checkpoint was trained with a different number of frames.
    Returns {"path", "n_timeframes_pretrained"}.
    '''
    path = resolve_checkpoint(spec, tracking_uri)
    state_dict = _model_state_dict_from_checkpoint(path)
    target = model.state_dict()

    if any(_FCN_AGGREGATOR.search(k) and k in target and state_dict[k].shape != target[k].shape for k in state_dict):
        raise ValueError("The FCN temporal aggregator's input size depends on n_timeframes, so it can't be "
                         "transferred to a different number of frames: use the transformer aggregator.")

    n_old = None
    for key, tensor in state_dict.items():
        if not _PER_FRAME_BN.search(key) or key not in target or tensor.shape == target[key].shape:
            continue
        n_channels, rest = divmod(target[key].numel(), n_timeframes)
        assert rest == 0, f"{key}: {target[key].numel()} features are not a multiple of {n_timeframes} frames"
        layer_n_old, rest = divmod(tensor.numel(), n_channels)
        if rest != 0 or (n_old is not None and layer_n_old != n_old):
            raise ValueError(f"{key}: can't infer the pretrained number of frames ({tensor.numel()} features, {n_channels} channels)")
        n_old = layer_n_old
        state_dict[key] = remap_per_frame_batch_norm(tensor, n_old, n_timeframes)

    mismatched = [f"{k}: checkpoint {tuple(state_dict[k].shape)} vs model {tuple(target[k].shape)}"
                  for k in state_dict if k in target and state_dict[k].shape != target[k].shape]
    if mismatched:
        raise ValueError("Checkpoint and model architectures differ (check the architecture flags):\n  " + "\n  ".join(mismatched))

    result = model.load_state_dict(state_dict, strict=False)
    if result.missing_keys or result.unexpected_keys:
        raise ValueError(f"Checkpoint and model architectures differ: missing={result.missing_keys}, unexpected={result.unexpected_keys}")

    if n_old is not None:
        logger.info("Remapped the encoder's per-frame batch norm from %d to %d frames (closest phase).", n_old, n_timeframes)
    logger.info("Initialized model from %s", path)
    return {"path": path, "n_timeframes_pretrained": n_old if n_old is not None else n_timeframes}
