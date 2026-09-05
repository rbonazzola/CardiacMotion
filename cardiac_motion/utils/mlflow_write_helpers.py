from subprocess import check_output
import logging
import os
from pathlib import Path
from urllib.parse import urlparse

import mlflow

logger = logging.getLogger(__name__)

def get_mlflow_parameters(config):
    loss = config.loss

    net = config.network_architecture

    loss_params = {
        "w_kl": loss.regularization.weight,
        "w_s": loss.reconstruction_s.weight
    }
    net_params = {
        "latent_dim_s": net.latent_dim_s,
        "latent_dim_c": net.latent_dim_c,
        "z_aggr_function": net.z_aggr_function,
        "convolution_type": net.convolution.type,
        "n_channels_enc": net.convolution.channels_enc,
        "n_channels_dec_c": net.convolution.channels_dec_c,
        "n_channels_dec_s": net.convolution.channels_dec_s,
        "reduction_factors": net.pooling.parameters.downsampling_factors,
        "phase_input": net.phase_input
    }

    mlflow_parameters = {
        "platform": check_output(["hostname"]).strip().decode(),
        **loss_params,
        **net_params,
    }

    return mlflow_parameters


###
def get_mlflow_dataset_params(config):
    '''
    Returns a dictionary containing the dataset parameters, to be logged to MLflow.
    '''
    d = config.dataset

    if d.data_type == "cardiac":
        mlflow_dataset_params = {
          "dataset_type": "cardiac",
          "dataset_static_representative": d.static_representative
        }

    elif d.data_type == "synthetic":
        mlflow_dataset_params = {
          "dataset_type": "synthetic",
          "dataset_max_static_amplitude": d.parameters.amplitude_static_max,
          "dataset_max_dynamic_amplitude": d.parameters.amplitude_dynamic_max,
          "dataset_n_timeframes": d.parameters.T,
          "dataset_freq_max": d.parameters.freq_max,
          "dataset_l_max": d.parameters.l_max,
          "dataset_resolution": d.parameters.mesh_resolution,
          "dataset_complexity_c": (d.parameters.l_max + 1) ** 2,
          "dataset_complexity_s": ((d.parameters.l_max + 1) ** 2) * d.parameters.freq_max,
          "dataset_complexity": ((d.parameters.l_max + 1) ** 2) * (d.parameters.freq_max + 1),
          "dataset_random_seed": d.parameters.random_seed,
          "dataset_template": "icosphere",  # TODO: add this as parameter in the configuration
          "dataset_center_around_mean": d.preprocessing.center_around_mean}

    mlflow_dataset_params.update({
        "n_training": config.sample_sizes[0],
        "n_validation": config.sample_sizes[1],
        "n_test": config.sample_sizes[2],
        "dataset_center_around_mean": False
    })

    return mlflow_dataset_params


def print_auto_logged_info(r):

    tags = {k: v for k, v in r.data.tags.items() if not k.startswith("mlflow.")}
    artifacts = [f.path for f in MlflowClient().list_artifacts(r.info.run_id, "model")]
    print("run_id: {}".format(r.info.run_id))
    print("artifacts: {}".format(artifacts))
    print("params: {}".format(r.data.params))
    print("metrics: {}".format(r.data.metrics))
    print("tags: {}".format(tags))


def prepare_mlflow_config(mlflow_config):
    """Normalize MLflow paths and avoid stale experiments with unwritable artifact dirs."""

    if mlflow_config.tracking_uri is not None and "://" not in str(mlflow_config.tracking_uri):
        mlflow_config.tracking_uri = str(Path(mlflow_config.tracking_uri).resolve())

    if mlflow_config.artifact_location is not None and "://" not in str(mlflow_config.artifact_location):
        mlflow_config.artifact_location = str(Path(mlflow_config.artifact_location).resolve())

    mlflow.set_tracking_uri(mlflow_config.tracking_uri)
    experiment = mlflow.get_experiment_by_name(mlflow_config.experiment_name)

    if experiment is not None and not _artifact_location_is_writable(experiment.artifact_location):
        original_name = mlflow_config.experiment_name
        mlflow_config.experiment_name = f"{original_name}_local"
        mlflow_config.artifact_location = None
        logger.warning(
            "MLflow experiment %r uses unwritable artifact_location=%r. "
            "Using experiment %r instead.",
            original_name,
            experiment.artifact_location,
            mlflow_config.experiment_name,
        )

    return mlflow_config


def _artifact_location_is_writable(artifact_location):
    path = _local_artifact_path(artifact_location)
    if path is None:
        return True

    if str(path).startswith("/home/user"):
        return False

    probe = path if path.exists() else path.parent
    while not probe.exists() and probe != probe.parent:
        probe = probe.parent

    return os.access(probe, os.W_OK)


def _local_artifact_path(artifact_location):
    if artifact_location is None:
        return None

    artifact_location = str(artifact_location)
    parsed = urlparse(artifact_location)
    if parsed.scheme and parsed.scheme != "file":
        return None
    if parsed.scheme == "file":
        return Path(parsed.path)
    return Path(artifact_location)


def mlflow_startup(mlflow_config):
    
    '''
      Starts MLflow run      
      mlflow_config: Namespace including run_id, experiment_name, run_name, artifact_location            
    
    '''
    
    mlflow.pytorch.autolog(log_models=getattr(mlflow_config, "log_models", False))
 
    if mlflow_config.tracking_uri is not None:
        mlflow.set_tracking_uri(mlflow_config.tracking_uri)
    
    try:
        mlflow_config.exp_id = mlflow.create_experiment(mlflow_config.experiment_name, artifact_location=mlflow_config.artifact_location)
    except:
      # If the experiment already exists, we can just retrieve its ID
        experiment = mlflow.get_experiment_by_name(mlflow_config.experiment_name)
        logger.info("Using existing MLflow experiment: %s", experiment)
        mlflow_config.exp_id = experiment.experiment_id

    run_info = {
        "run_id": mlflow_config.run_id,
        "experiment_id": mlflow_config.exp_id,
        "run_name": mlflow_config.run_name,
    }
    
    mlflow.start_run(**run_info)
    
        
def mlflow_log_additional_params(config):
    
    '''
    Log additional parameters, extracted from config
    '''
        
    mlflow_params = get_mlflow_parameters(config)
    mlflow_dataset_params = get_mlflow_dataset_params(config)
    mlflow_params.update(mlflow_dataset_params)
    mlflow.log_params(mlflow_params)   


def log_computational_graph(model, x):
    
    from torchviz import make_dot
    yhat = model(x)
    make_dot(yhat, params=dict(list(model.named_parameters()))).render("comp_graph_network", format="png")
    mlflow.log_figure("comp_graph_network.png")
