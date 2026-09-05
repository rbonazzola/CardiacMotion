import os
import sys
import logging
import argparse
import pickle as pkl
from typing import Mapping
from easydict import EasyDict
import pprint
import numpy as np
from packaging import version

import torch
from torch import Tensor
from torch.utils.data import DataLoader

import pytorch_lightning as pl
from pytorch_lightning.loggers import MLFlowLogger

from pytorch_lightning.callbacks import (
    EarlyStopping, 
    ModelCheckpoint, 
    ModelSummary, 
    RichProgressBar
)

from pytorch_lightning.profilers import SimpleProfiler

import mlflow
from mlflow.tracking import MlflowClient

from IPython import embed

# Custom imports
from CardiacMotion.config.cli_args import CLI_args, overwrite_config_items
from CardiacMotion.config.load_config import load_yaml_config, to_dict
from CardiacMotion.utils.helpers import *
from CardiacMotion.utils.mlflow_helpers import get_mlflow_parameters, get_mlflow_dataset_params
from CardiacMotion.utils.CardioMesh.CardiacMesh import Cardiac3DMesh

from CardiacMotion.models.Model3D import (
    Encoder3DMesh, 
    Decoder3DMesh
)

from CardiacMotion.models.Model4D import (
    DECODER_C_ARGS, DECODER_S_ARGS, ENCODER_ARGS,
    DecoderStyle, DecoderContent, DecoderTemporalSequence,
    EncoderTemporalSequence, AutoencoderTemporalSequence
)

from CardiacMotion.models.TemporalAggregators import (
    TemporalAggregator, 
    FCN_Aggregator
)

from CardiacMotion.lightning.ComaLightningModule import CoMA_Lightning
from CardiacMotion.lightning.EncoderLightningModule import TemporalEncoderLightning
from CardiacMotion.data.DataModules import CardiacMeshPopulationDataset, CardiacMeshPopulationDM

from Constants import partitions

# Constants
REPOS_ROOT = f"{os.environ['HOME']}/01_repos"
sys.path.append(REPOS_ROOT)

# Logging configuration
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger()

# Profiler
profiler = SimpleProfiler(filename='simple_profiler_output.txt')

# Repositories
class Repos:
    CARDIAC_MOTION = f"{os.environ['HOME']}/01_repos/CardiacMotion"
    GWAS = f"{os.environ['HOME']}/01_repos/GWAS_pipeline/"
    CARDIAC = f"{os.environ['HOME']}/01_repos/CardiacCOMA/"
    CARDIOMESH = f"{os.environ['HOME']}/01_repos/CardioMesh/"


# MLflow Startup
def mlflow_startup(mlflow_config):
    mlflow.pytorch.autolog(log_models=True)
    if mlflow_config.tracking_uri is not None:
        mlflow.set_tracking_uri(mlflow_config.tracking_uri)
    
    try:
        exp_id = mlflow.create_experiment(mlflow_config.experiment_name, artifact_location=mlflow_config.artifact_location)
    except:
        experiment = mlflow.get_experiment_by_name(mlflow_config.experiment_name)
        exp_id = experiment.experiment_id
    
    run_info = {
        "run_id": trainer.logger.run_id,
        "experiment_id": exp_id,
        "run_name": mlflow_config.run_name,
    }
    mlflow.start_run(**run_info)


# MLflow Log Additional Parameters
def mlflow_log_additional_params(config):

    mlflow_params = get_mlflow_parameters(config)
    mlflow_dataset_params = get_mlflow_dataset_params(config)
    mlflow_params.update(mlflow_dataset_params)
    mlflow.log_params(mlflow_params)

# Get COMA Arguments
def get_coma_args(config: Mapping):

    net = config.network_architecture
    convs = net.convolution
    coma_args = {
        "num_features": net.n_features,
        "n_layers": len(convs.channels_enc),
        "num_conv_filters_enc": convs.channels_enc,
        "num_conv_filters_dec_c": convs.channels_dec_c,
        "num_conv_filters_dec_s": convs.channels_dec_s,
        "cheb_polynomial_order": convs.parameters.polynomial_degree,
        "latent_dim_content": net.latent_dim_c,
        "latent_dim_style": net.latent_dim_s,
        "is_variational": config.loss.regularization.weight != 0,
        "mode": "testing",
        "n_timeframes": config.dataset.parameters.T,
        "phase_input": net.phase_input,
        "z_aggr_function": net.z_aggr_function
    }
    return EasyDict(coma_args)

# Get COMA Matrices
def get_coma_matrices(config, template, partition, from_cached=True, cache=True):

    downsample_factors = config.network_architecture.pooling.parameters.downsampling_factors
    matrices_hash = hash((hash("1000215"), hash(tuple(downsample_factors)), hash(partition))) % 1000000
    cached_file = f"data/cached/matrices/{matrices_hash}.pkl"

    if from_cached and os.path.exists(cached_file):
        A_t, D_t, U_t, n_nodes = pkl.load(open(cached_file, "rb"))
    else:
        template_mesh = Mesh(template.v, template.f)
        M, A, D, U = mesh_operations.generate_transform_matrices(template_mesh, downsample_factors)
        n_nodes = [len(M[i].v) for i in range(len(M))]
        A_t, D_t, U_t = ([scipy_to_torch_sparse(x).float() for x in X] for X in (A, D, U))
        if cache:
            os.makedirs(os.path.dirname(cached_file), exist_ok=True)
            with open(cached_file, "wb") as ff:
                pkl.dump((A_t, D_t, U_t, n_nodes), ff)

    return {
        "downsample_matrices": D_t,
        "upsample_matrices": U_t,
        "adjacency_matrices": A_t,
        "n_nodes": n_nodes,
        "template": template
    }

# Memory Usage Callback
class MemoryUsageCallback(pl.Callback):
    def on_epoch_end(self, trainer, pl_module):
        print(f'Memory allocated: {torch.cuda.memory_allocated()} bytes')
        print(f'Memory cached: {torch.cuda.memory_reserved()} bytes')

# Model Checkpoint with Threshold
class ModelCheckpointWithThreshold(ModelCheckpoint):
    def __init__(self, monitor, threshold, mode='min', *args, **kwargs):
        super().__init__(monitor=monitor, mode=mode, *args, **kwargs)
        self.threshold = threshold

    def _should_save_checkpoint(self, trainer):
        current = trainer.callback_metrics.get(self.monitor)
        if current is None:
            return False
        if self.mode == 'min':
            return current < self.threshold
        else:
            return current > self.threshold

# Main Function
def main(model, datamodule, trainer, mlflow_config=None):
    if mlflow_config:
        mlflow_config.run_id = trainer.logger.run_id
        mlflow_startup(mlflow_config)
        mlflow_log_additional_params(config)
    
    trainer.fit(model, datamodule=datamodule)
    trainer.test(datamodule=datamodule, ckpt_path='best')
    mlflow.end_run()

# Add Trainer Arguments
def add_trainer_args(parser):
    parser_trainer_group = parser.add_argument_group("trainer")
    trainer_args = ["max_epochs", "min_epochs", "precision", "logger", "overfit_batches", "limit_test_batches"]

    parser_trainer_group.add_argument("--max_epochs", type=int, default=100, help="Number of epochs to train the model.")
    parser_trainer_group.add_argument("--min_epochs", type=int, default=1, help="Minimum number of epochs to train the model.")
    
    if version.parse(pl.__version__) < version.parse("2.0.0"):
        parser_trainer_group.add_argument("--gpus", type=int, default=1, help="Number of GPUs to use for training.")
        parser_trainer_group.add_argument("--auto_select_gpus", action='store_true', help="If enabled, auto select available GPUs.")
        parser_trainer_group.add_argument("--auto_scale_batch_size", action='store_true', help="If enabled, automatically scale the batch size.")
        trainer_args.extend(["gpus", "auto_select_gpus", "auto_scale_batch_size"])
    else:
        parser_trainer_group.add_argument("--devices", type=int, default=1, help="Number of devices (GPUs/TPUs) to use for training.")
        parser_trainer_group.add_argument("--accelerator", type=str, default='gpu', help="Accelerator type to use for training (e.g., 'cpu', 'gpu').")
        trainer_args.extend(["devices", "accelerator"])
    
    parser_trainer_group.add_argument("--precision", type=int, choices=[16, 32], default=32, help="Precision to use during training.")
    parser_trainer_group.add_argument("--logger", type=str, help="Logger for experiment tracking.")
    parser_trainer_group.add_argument("--overfit_batches", type=float, default=0.0, help="Percent of training set to overfit on.")
    parser_trainer_group.add_argument("--limit_test_batches", type=float, default=1.0, help="How much of the test set to use.")
    parser_trainer_group.add_argument("--patience", type=int, default=10, help="Patience for training")

    return trainer_args

# Entry Point
if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="Pytorch Trainer for Convolutional Mesh Autoencoders", argument_default=argparse.SUPPRESS)
    my_args = parser.add_argument_group("model")

    for k, v in CLI_args.items():
        my_args.add_argument(*k, **v)
        
    my_args.add_argument("--partition", type=str, default="left_ventricle")
    my_args.add_argument("--n_timeframes", type=int, default=50)
    my_args.add_argument("--static_representative", type=str, default="end_diastole", help="Currently, only \"end_diastole\" and \"temporal_mean\" are supported.")    
        
    trainer_args = add_trainer_args(parser)    
    args = parser.parse_args()

    if not os.path.exists(args.yaml_config_file):
        logger.error("Config not found: " + args.yaml_config_file)

    ref_config = load_yaml_config(args.yaml_config_file)

    try:
        config_to_replace = args.config
        config = overwrite_config_items(ref_config, config_to_replace)
    except AttributeError:
        config = ref_config

    arg_groups = {}    
    for group in parser._action_groups:
        group_dict = { a.dest: rgetattr(args, a.dest, None) for a in group._group_actions }
        arg_groups[group.title] = EasyDict(group_dict)
    
    trainer_args = arg_groups["trainer"]
            
    if args.disable_mlflow_logging:
        config.mlflow = None

    if config.mlflow:
        if config.mlflow.experiment_name is None:
            config.mlflow.experiment_name = f"{args.partition}"

        exp_info = {
            "experiment_name": config.mlflow.experiment_name,
            "artifact_location": config.mlflow.artifact_location
        }

        trainer_args.logger = MLFlowLogger(
            tracking_uri=config.mlflow.tracking_uri,
            **exp_info,
        )

        mlflow.set_tracking_uri(config.mlflow.tracking_uri)
    else:
        trainer_args.logger = None

    PARTITION = args.partition
    FACES_FILE = "CardioMesh/data/faces_and_downsampling_mtx_frac_0.1_LV.pkl"
    MEAN_ACROSS_CYCLE_FILE = f"CardioMesh/data/cached/mean_shape_time_avg__{PARTITION}.npy"
    PROCRUSTES_FILE = f"CardioMesh/data/cached/procrustes_transforms_{PARTITION}.pkl"    
    SUBSETTING_MATRIX_FILE = f"CardioMesh/data/cached/subsetting_matrix_{PARTITION}.pkl" 
    MESHES_PATH = "CardiacMotion/data/datasets/meshes"
    
    subsetting_matrix = pkl.load(open(SUBSETTING_MATRIX_FILE, "rb"))
    
    ID = "1000511"
    fhm_mesh = Cardiac3DMesh(
       filename=f"{os.environ['HOME']}/01_repos/CardiacMotion/data/datasets/meshes/{ID}/models/FHM_res_0.1_time001.npy",
       faces_filename="CardioMesh/data/faces_fhm_10pct_decimation.csv",
       subpart_id_filename="CardioMesh/data/subpartIDs_FHM_10pct.txt"
    )
    mean_shape = np.load(MEAN_ACROSS_CYCLE_FILE)
    faces = fhm_mesh[partitions[PARTITION]].f
    template = EasyDict({ "v": mean_shape, "f": faces })
    
    N_subj = 10000
    NT = args.n_timeframes
    PHASES = 1+(50/NT)*np.array(range(NT))

    cardiac_dataset = CardiacMeshPopulationDataset(
        root_path=MESHES_PATH, 
        procrustes_transforms=PROCRUSTES_FILE,
        faces=faces,
        subsetting_matrix=subsetting_matrix,
        template_mesh= EasyDict({ "v": mean_shape, "f": faces }),
        N_subj=N_subj,
        phases_filter=PHASES
    )
    
    mesh_dm = CardiacMeshPopulationDM(cardiac_dataset, batch_size=args.config.batch_size)        
    mesh_dm.setup()
    x = EasyDict(next(iter(mesh_dm.train_dataloader())))
    mesh_template = mesh_dm.dataset.template_mesh
            
    coma_args = get_coma_args(config)
    coma_matrices = get_coma_matrices(config, mesh_template, PARTITION)
    coma_args.update(coma_matrices)
  
    enc_config = EasyDict({k: v for k, v in coma_args.items() if k in ENCODER_ARGS})
    encoder = Encoder3DMesh(**enc_config, n_timeframes=NT)

    enc_config.latent_dim = config.network_architecture.latent_dim_c + config.network_architecture.latent_dim_s 
    h = encoder.forward_conv_stack(x.s_t, preserve_graph_structure=False)
    z_aggr = FCN_Aggregator(features_in = NT*h.shape[-1], features_out= enc_config.latent_dim)
    t_encoder = EncoderTemporalSequence(encoder3d = encoder, z_aggr_function=z_aggr, is_variational=coma_args.is_variational)   

    decoder_config_c = EasyDict({ k:v for k,v in coma_args.items() if k in DECODER_C_ARGS })
    decoder_config_s = EasyDict({ k:v for k,v in coma_args.items() if k in DECODER_S_ARGS })
    decoder_content = DecoderContent(decoder_config_c)
    decoder_style = DecoderStyle(decoder_config_s, phase_embedding_method="exp_v1", n_timeframes=NT)
    t_decoder = DecoderTemporalSequence(decoder_content, decoder_style, is_variational=coma_args.is_variational)
    t_ae = AutoencoderTemporalSequence(encoder=t_encoder, decoder=t_decoder, is_variational=coma_args.is_variational)
   
    lit_module = CoMA_Lightning(
        model=t_ae, 
        loss_params=config.loss, 
        optimizer_params=config.optimizer,
        additional_params=config,
        mesh_template=mesh_template
    )

    early_stopping = EarlyStopping(monitor="val_loss", mode="min", patience=trainer_args.patience)
    custom_checkpoint_callback = ModelCheckpointWithThreshold(monitor='val_recon_loss', threshold=3 , save_top_k=1)
    model_summary = ModelSummary(max_depth=-1)

    progress_bar = RichProgressBar(
        theme=RichProgressBarTheme(
        description="green_yellow",
        progress_bar="green1",
        progress_bar_finished="green1",
        progress_bar_pulse="#6206E0",
        batch_progress="green_yellow",
        time="grey82",
        processing_speed="grey82",
        metrics="grey82",
      )
    )

    trainer_kwargs = {
        "callbacks": [ early_stopping, custom_checkpoint_callback, model_summary, progress_bar, MemoryUsageCallback() ],
        "devices": trainer_args.devices,
        "accelerator": trainer_args.accelerator,        
        "min_epochs": trainer_args.min_epochs, "max_epochs": trainer_args.max_epochs,
        "logger": trainer_args.logger,
        "precision": trainer_args.precision,
        "overfit_batches": trainer_args.overfit_batches,
        "limit_test_batches": trainer_args.limit_test_batches
    }

    # trainer_kwargs["profiler"] = profiler
    
    trainer = pl.Trainer(**trainer_kwargs)

    if args.show_config or args.dry_run:
        pp = pprint.PrettyPrinter(indent=2, compact=True)
        pp.pprint(to_dict(config))
        if args.dry_run:
            exit()

    main(lit_module, mesh_dm, trainer, config.mlflow)
