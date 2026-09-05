import os, sys
import logging
import time
from contextlib import contextmanager

from packaging import version
import pytorch_lightning as pl
from pytorch_lightning.loggers import MLFlowLogger
import torch

import mlflow
from mlflow.tracking import MlflowClient

import argparse
from easydict import EasyDict
import pprint
import numpy as np

import cardio_mesh

from cardiac_motion import AutoencoderTemporalSequence

from lightning_modules.ComaLightningModule import CoMA_Lightning
from data.DataModules import CardiacMeshPopulationDataset, CardiacMeshPopulationDM


from utils.mlflow_write_helpers import (
    get_mlflow_parameters, 
    get_mlflow_dataset_params,
    mlflow_startup,
    mlflow_log_additional_params,
    prepare_mlflow_config,
)

from utils.helpers import (
    get_coma_args,
    get_coma_matrices,
    get_n_equispaced_timeframes
)

from utils.lightning_helpers import (
    early_stopping,
    model_checkpoint,
    rich_model_summary,
    progress_bar,
    MemoryUsageCallback,
    ModelCheckpointWithThreshold,
    EpochMetricsTableCallback,
)

from config.cli_args import (
    CLI_args, 
    overwrite_config_items
)

from config.load_config import (
    load_yaml_config, 
    rgetattr,
    to_dict
)

################################################################################################

from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.profilers import SimpleProfiler
profiler = SimpleProfiler(filename='simple_profiler_output.txt')

from cardiac_motion import logger

##########################################################################################

@contextmanager
def log_step(message: str):
    start = time.perf_counter()
    logger.info("%s...", message)
    try:
        yield
    except Exception:
        logger.exception("%s failed after %.2fs", message, time.perf_counter() - start)
        raise
    else:
        logger.info("%s done in %.2fs", message, time.perf_counter() - start)


def _print_config(cfg: dict, title: str = "Run configuration") -> None:
    from rich.console import Console
    from rich.table import Table
    from rich import box

    def _flatten(d: dict, prefix: str = "") -> list[tuple[str, str]]:
        rows = []
        for k, v in d.items():
            key = f"{prefix}.{k}" if prefix else k
            if isinstance(v, dict):
                rows.extend(_flatten(v, key))
            else:
                rows.append((key, str(v)))
        return rows

    t = Table(title=title, box=box.ROUNDED, show_header=True, header_style="bold cyan")
    t.add_column("Parameter", style="dim")
    t.add_column("Value")
    for k, v in _flatten(cfg):
        t.add_row(k, v)

    Console().print(t)

##########################################################################################

def add_trainer_args(parser):
    
    from packaging import version
    parser_trainer_group = parser.add_argument_group("trainer")
    
    trainer_args = ["max_epochs", "min_epochs", "precision", "logger", "overfit_batches", "limit_test_batches"]

    parser_trainer_group.add_argument("--max_epochs", "--max-epochs", type=int, default=100, help="Number of epochs to train the model.")
    parser_trainer_group.add_argument("--min_epochs", "--min-epochs", type=int, default=1, help="Minimum number of epochs to train the model.")
    
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
    parser_trainer_group.add_argument("--gradient_clip_val", "--gradient-clip-val", type=float, default=1.0, help="Max gradient norm for clipping (0 = disabled).")
    parser_trainer_group.add_argument("--log_level", "--log-level", type=str, default="INFO", help="Python logging level.")
    parser_trainer_group.add_argument(
        "--float32_matmul_precision",
        "--float32-matmul-precision",
        choices=["highest", "high", "medium"],
        default="medium",
        help="Float32 matmul precision for Tensor Core GPUs. Use 'highest' for strict FP32.",
    )
    parser_trainer_group.add_argument(
        "--enable_rich_progress",
        "--enable-rich-progress",
        action="store_true",
        default=False,
        help="Enable Rich progress callbacks. Disabled by default because RichProgressBar can fail in some TTY/MLflow runs.",
    )
    parser_trainer_group.add_argument(
        "--disable_metrics_table",
        "--disable-metrics-table",
        action="store_true",
        default=False,
        help="Disable the Rich epoch metrics table.",
    )
    parser_trainer_group.add_argument(
        "--metrics_table_rows",
        "--metrics-table-rows",
        type=int,
        default=20,
        help="Number of recent epochs to show in the metrics table.",
    )

    trainer_args.append("gradient_clip_val")
    trainer_args.append("log_level")
    trainer_args.append("float32_matmul_precision")
    trainer_args.append("enable_rich_progress")
    trainer_args.append("disable_metrics_table")
    trainer_args.append("metrics_table_rows")
    return trainer_args


def main(model, datamodule, trainer, mlflow_config=None):

    '''
      config (Namespace):       
      trainer_args (Namespace):
      mlflow_config (Namespace):
    '''

    if mlflow_config:
        logger.info("Starting MLflow run in experiment=%s", mlflow_config.experiment_name)
        mlflow_config.run_id = trainer.logger.run_id
        mlflow_startup(mlflow_config)             
        mlflow_log_additional_params(config)

    with log_step("Training model"):
        trainer.fit(model, datamodule=datamodule)

    with log_step("Testing best checkpoint"):
        trainer.test(datamodule=datamodule, ckpt_path='best') # Generates metrics for the full test dataset
    # trainer.predict(ckpt_path='best', datamodule=datamodule) # Generates figures for a few samples

    mlflow.end_run()
    logger.info("Run finished")


if __name__ == "__main__":

    # --------------------------------
    # 1. Parse Command-line Arguments
    # --------------------------------

    parser = argparse.ArgumentParser(
        description="Pytorch Trainer for Spatio-temporal Convolutional Mesh Autoencoders",
        argument_default=argparse.SUPPRESS
    )
    
    my_args = parser.add_argument_group("model")
    for k, v in CLI_args.items():
        my_args.add_argument(*k, **v)
    
    my_args.add_argument("--n_subjects", type=int, default=1000)
    my_args.add_argument("--partition", type=str, default="left_ventricle")
    my_args.add_argument("--n_timeframes", type=int, default=50)
    my_args.add_argument("--use-closed-chambers", default=True, action='store_true')
    my_args.add_argument("--static_representative", type=str, default="end_diastole",
                         help="Currently, only 'end_diastole' and 'temporal_mean' are supported.")
    my_args.add_argument("--center_around_mean", "--center-around-mean", default=False,
                         action='store_true',
                         help="Subtract the population mean shape from all meshes before training. "
                              "Reduces data scale to small residuals and stabilizes training.")
    my_args.add_argument("--mlflow_log_models", "--mlflow-log-models", default=False,
                         action="store_true",
                         help="Enable MLflow model autologging. Disabled by default because stale "
                              "artifact locations can point to non-writable paths.")
    trainer_args = add_trainer_args(parser)    
    args = parser.parse_args()

    # --------------------------
    # 2. Load Configuration File
    # --------------------------
    assert os.path.exists(args.yaml_config_file), f"Config file not found: {args.yaml_config_file}"
    ref_config = load_yaml_config(args.yaml_config_file)
    config = overwrite_config_items(ref_config, getattr(args, 'config', {}))

    assert os.path.exists(config.mlflow.tracking_uri), f"MLflow tracking URI, {config.mlflow.tracking_uri}, does not exist"
    assert config.mlflow.artifact_location is None or os.path.exists(config.mlflow.artifact_location), f"MLflow artifact location, {config.mlflow.artifact_location}, does not exist"

    # https://stackoverflow.com/questions/38884513/python-argparse-how-can-i-get-namespace-objects-for-argument-groups-separately
    arg_groups = {}    
    for group in parser._action_groups:
        # print(group.title)
        # print(group._group_actions)
        group_dict = { a.dest: rgetattr(args, a.dest, None) for a in group._group_actions }
        arg_groups[group.title] = EasyDict(group_dict)
    
    trainer_args = arg_groups["trainer"]
    logger.setLevel(getattr(logging, trainer_args.log_level.upper()))
    logger.info("Starting CardiacMotion training script")
    logger.info("Arguments: partition=%s, n_subjects=%s, n_timeframes=%s, accelerator=%s, devices=%s",
                args.partition, args.n_subjects, args.n_timeframes, trainer_args.accelerator, trainer_args.devices)
    torch.set_float32_matmul_precision(trainer_args.float32_matmul_precision)
    logger.info("Float32 matmul precision set to %s", trainer_args.float32_matmul_precision)

    # --------------------------
    # 3. MLflow Configuration
    # --------------------------
    if args.disable_mlflow_logging:
        logger.info("MLflow logging disabled")
        config.mlflow = None

    if config.mlflow:
        if config.mlflow.experiment_name is None:
            config.mlflow.experiment_name = args.partition

        config.mlflow.log_models = args.mlflow_log_models
        config.mlflow = prepare_mlflow_config(config.mlflow)
        logger.info("MLflow configuration: %s", config.mlflow)
        
        trainer_args.logger = MLFlowLogger(
            tracking_uri=config.mlflow.tracking_uri,
            experiment_name=config.mlflow.experiment_name,
            artifact_location=config.mlflow.artifact_location
        )
        mlflow.set_tracking_uri(config.mlflow.tracking_uri)
    else:
        trainer_args.logger = None
    
    # --------------------------
    # 4. Load Data and Preprocess
    # --------------------------
    
    ONE_RANDOM_ID = "1000511"; END_DIASTOLE = 1
    partition = args.partition
    logger.info("Using cardiac mesh root: %s", cardio_mesh.MESHES_DIR)
    with log_step(f"Loading template and cached preprocessing for partition={partition}"):
        subsetting_matrix = cardio_mesh.paths.get_subsetting_matrix(partition)
        mean_shape        = cardio_mesh.paths.get_mean_shape(partition)
        template_fhm_mesh = cardio_mesh.load_full_heart_mesh(ONE_RANDOM_ID, timeframe=END_DIASTOLE)
    logger.info("Subsetting matrix shape=%s; mean shape=%s", getattr(subsetting_matrix, "shape", None), mean_shape.shape)
    
    # This adds to the mesh the valve surfaces that close up the different chambers
    _partition_to_closed = {
        "left_ventricle":  "LV_closed",
        "right_ventricle": "RV_closed",
        "biventricle":     "BV_closed",
        "left_atrium":     "LA_closed",
        "right_atrium":    "RA_closed",
        "aorta":           "aorta",
    }
    if args.use_closed_chambers:
        closed_chamber = cardio_mesh.close_chamber(
            _partition_to_closed.get(args.partition, args.partition)
        )
    else:
        closed_chamber = args.partition
    logger.info("Using mesh partition labels: %s", closed_chamber)

    phases_filter = get_n_equispaced_timeframes(args.n_timeframes)
    logger.info("Using %d phases: %s", len(phases_filter), phases_filter)

    with log_step("Building cardiac dataset index"):
        cardiac_dataset = CardiacMeshPopulationDataset(
            root_path=cardio_mesh.MESHES_DIR,
            procrustes_transforms=cardio_mesh.paths.get_procrustes_file(partition),
            faces=(faces := template_fhm_mesh[closed_chamber].f),
            subsetting_matrix=subsetting_matrix,
            template_mesh=(mesh_template := EasyDict({"v": mean_shape, "f": faces})),
            N_subj=(N_subj := args.n_subjects),
            phases_filter=phases_filter,
            center_around_mean=args.center_around_mean,
        )
    logger.info("Dataset ready: subjects=%d, frames_per_subject=%d, vertices=%d, faces=%d",
                len(cardiac_dataset), len(phases_filter), mesh_template.v.shape[0], mesh_template.f.shape[0])

    with log_step("Setting up datamodule splits"):
        ( mesh_dm := CardiacMeshPopulationDM(cardiac_dataset, batch_size=config.batch_size) ).setup()

    # --------------------------
    # 5. Define Model
    # --------------------------
    
    with log_step("Building model and COMA matrices"):
        model      = AutoencoderTemporalSequence.build_from_config(config, mesh_template, args.partition, args.n_timeframes)
        lit_module = CoMA_Lightning(model=model, loss_params=config.loss, optimizer_params=config.optimizer, additional_params=config, mesh_template=mesh_template)
    logger.info("Model ready: trainable_parameters=%d", sum(p.numel() for p in lit_module.parameters() if p.requires_grad))

    # --------------------------
    # 6. Configure Trainer and Run
    # --------------------------
    callbacks = [
        early_stopping,
        model_checkpoint,
        # MemoryUsageCallback()
    ]
    metrics_table_enabled = not trainer_args.disable_metrics_table
    if metrics_table_enabled:
        callbacks.append(EpochMetricsTableCallback(max_rows=trainer_args.metrics_table_rows))
        if trainer_args.enable_rich_progress:
            logger.warning("Metrics table is enabled, so progress bars are disabled. Use --disable-metrics-table to enable Rich progress.")
    elif trainer_args.enable_rich_progress:
        callbacks.extend([rich_model_summary, progress_bar])

    (( trainer_kwargs := dict(callbacks=callbacks) )
        .update({ k: getattr(trainer_args, k) for k in ["devices", "accelerator", "min_epochs", "max_epochs", "logger", "precision", "gradient_clip_val"] } ))
    trainer_kwargs["enable_progress_bar"] = not metrics_table_enabled

    logger.info("Trainer kwargs: %s", {k: v for k, v in trainer_kwargs.items() if k != "callbacks"})
    trainer = pl.Trainer(**trainer_kwargs)

    if args.show_config or args.dry_run:
        _print_config(to_dict(config))
        if args.dry_run:
            exit()

    main(lit_module, mesh_dm, trainer, config.mlflow)
