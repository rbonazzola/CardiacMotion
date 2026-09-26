'''
mlflow_read_helpers: locating a run's checkpoint and renaming state dict keys.
(Replaces a script that instantiated a Run for a hard-coded run ID not present in this repo.)
'''
import os
import sys

import pandas as pd
import torch

sys.path.insert(0, "cardiac_motion")
import utils.mlflow_read_helpers as helpers
from utils.mlflow_read_helpers import Run, _rename_state_dict_keys


def _touch(path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    open(path, "wb").close()
    return str(path)


def test_checkpoint_lookup_prefers_best_model_then_latest_legacy_epoch(tmp_path, monkeypatch):
    monkeypatch.setattr(helpers, "MLFLOW_URI", str(tmp_path))
    best = _touch(tmp_path / "1" / "run_new" / "artifacts" / "checkpoints" / "best_model.ckpt")
    _touch(tmp_path / "1" / "run_new" / "checkpoints" / "epoch=3-step=10.ckpt")          # ignored: best_model wins
    _touch(tmp_path / "1" / "run_old" / "checkpoints" / "epoch=3-step=10.ckpt")
    latest = _touch(tmp_path / "1" / "run_old" / "checkpoints" / "epoch=12-step=40.ckpt")
    monkeypatch.setattr(Run, "runs_df", pd.DataFrame({"experiment_id": ["1", "1", "1"], "run_id": ["run_new", "run_old", "run_empty"]}),
                        raising=False)

    locations = Run.get_all_ckpt_paths()
    assert locations == {"run_new": best, "run_old": latest}  # runs without checkpoints are left out


def test_rename_state_dict_keys_keeps_module_metadata():
    state_dict = torch.nn.Sequential(torch.nn.Linear(2, 2)).state_dict()
    prefixed = type(state_dict)(("model." + k, v) for k, v in state_dict.items())
    prefixed._metadata = type(state_dict._metadata)((("model." + k) if k else "model", v) for k, v in state_dict._metadata.items())

    renamed = _rename_state_dict_keys(prefixed, "model.", "")
    assert list(renamed) == list(state_dict)
    assert renamed._metadata["0"] == state_dict._metadata["0"]  # module versions survive the renaming


def test_run_param_batch_norm_defaults_to_all_for_older_runs(tmp_path):
    run = Run.__new__(Run)  # without __init__, which loads the whole run
    run.RUN_BASE_DIR = str(tmp_path)
    assert run.get_param("batch_norm", default="all") == "all"
    os.makedirs(tmp_path / "params")
    (tmp_path / "params" / "batch_norm").write_text("shared")
    assert run.get_param("batch_norm", default="all") == "shared"
