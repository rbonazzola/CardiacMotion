'''
get_mlflow_parameters must log the loss-weight ramp settings, so runs with and without a ramp
can be told apart in MLflow.
'''
import sys

sys.path.insert(0, "cardiac_motion")
from config.load_config import load_yaml_config
from utils.mlflow_write_helpers import get_mlflow_parameters


def test_ramp_settings_are_logged():
    config = load_yaml_config("config_files/config_folded_c_and_s.yaml")
    config.loss.reconstruction_s.ramp_epochs = 20
    config.loss.reconstruction_s.start_weight = 0.1
    params = get_mlflow_parameters(config)
    assert params["w_s_ramp_epochs"] == 20 and params["w_s_start"] == 0.1
    for key in ("w_s_content_patience", "w_s_content_min_delta", "w_smooth_start", "w_smooth_ramp_epochs"):
        assert key in params

    config.loss.reconstruction_s.ramp_epochs = 0
    assert get_mlflow_parameters(config)["w_s_ramp_epochs"] == 0
