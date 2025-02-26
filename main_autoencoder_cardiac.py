import os, sys

from packaging import version
import pytorch_lightning as pl
from pytorch_lightning.loggers import MLFlowLogger

import mlflow
from mlflow.tracking import MlflowClient

import argparse
from easydict import EasyDict
import pprint
import numpy as np

import cardio_mesh

from cardiac_motion import (
    Encoder3DMesh,
    FCN_Aggregator,
    EncoderTemporalSequence,
    DecoderContent,
    DecoderStyle,
    DecoderTemporalSequence,    
    AutoencoderTemporalSequence,
    ENCODER_ARGS,
    DECODER_C_ARGS,
    DECODER_S_ARGS,        
)

from config.cli_args import (
    CLI_args, overwrite_config_items
)

from config.load_config import (
    load_yaml_config, 
    rgetattr
)

from lightning_modules.ComaLightningModule import CoMA_Lightning
from data.DataModules import CardiacMeshPopulationDataset, CardiacMeshPopulationDM


from utils.mlflow_helpers import (
    get_mlflow_parameters, 
    get_mlflow_dataset_params    
)

from utils.helpers import (
    get_coma_args,
    get_coma_matrices,
    early_stopping,
    model_checkpoint,
    rich_model_summary,
    progress_bar
)

################################################################################################

from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.profilers import SimpleProfiler
profiler = SimpleProfiler(filename='simple_profiler_output.txt')

import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()])

logger = logging.getLogger()

################################################################################################

def mlflow_startup(mlflow_config):
    
    '''
      Starts MLflow run      
      mlflow_config: Namespace including run_id, experiment_name, run_name, artifact_location            
    
    '''
    
    mlflow.pytorch.autolog(log_models=True)
 
    if mlflow_config.tracking_uri is not None:
        mlflow.set_tracking_uri(mlflow_config.tracking_uri)
    
    try:
        exp_id = mlflow.create_experiment(mlflow_config.experiment_name, artifact_location=mlflow_config.artifact_location)
    except:
      # If the experiment already exists, we can just retrieve its ID
        experiment = mlflow.get_experiment_by_name(mlflow_config.experiment_name)
        print(experiment)
        exp_id = experiment.experiment_id
    run_info = {
        "run_id": trainer.logger.run_id,
        "experiment_id": exp_id,
        "run_name": mlflow_config.run_name,
        # "tags": config.additional_mlflow_tags
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
        

class MemoryUsageCallback(pl.Callback):
    def on_epoch_end(self, trainer, pl_module):
        print(f'Memory allocated: {torch.cuda.memory_allocated()} bytes')
        print(f'Memory cached: {torch.cuda.memory_reserved()} bytes')


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


##########################################################################################

def get_n_equispaced_timeframes(n_timeframes):
        assert n_timeframes in {2, 5, 10, 25, 50}, f"Number of timeframes (args.n_timeframes) is {args.n_timeframes} which does not divide 50."
        phases = 1 + (50 / n_timeframes) * np.array(range(n_timeframes))    


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

    return trainer_args


def main(model, datamodule, trainer, mlflow_config=None):

    '''
      config (Namespace):       
      trainer_args (Namespace):
      mlflow_config (Namespace):
    '''

    if mlflow_config:
        mlflow_config.run_id = trainer.logger.run_id
        mlflow_startup(mlflow_config)             
        mlflow_log_additional_params(config)
    
    trainer.fit(model, datamodule=datamodule)
    trainer.test(datamodule=datamodule, ckpt_path='best') # Generates metrics for the full test dataset
    # trainer.predict(ckpt_path='best', datamodule=datamodule) # Generates figures for a few samples

    mlflow.end_run()


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
    
    my_args.add_argument("--partition", type=str, default="left_ventricle")
    my_args.add_argument("--n_timeframes", type=int, default=50)
    my_args.add_argument("--use-closed-chambers", default=True, action='store_true')
    my_args.add_argument("--static_representative", type=str, default="end_diastole", 
                         help="Currently, only 'end_diastole' and 'temporal_mean' are supported.")    
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

    try:
        config_to_replace = args.config
        config = overwrite_config_items(ref_config, config_to_replace)
    except AttributeError:
        # If there are no elements to replace
        config = ref_config
        pass

    # https://stackoverflow.com/questions/38884513/python-argparse-how-can-i-get-namespace-objects-for-argument-groups-separately
    arg_groups = {}    
    for group in parser._action_groups:
        # print(group.title)
        # print(group._group_actions)
        group_dict = { a.dest: rgetattr(args, a.dest, None) for a in group._group_actions }
        arg_groups[group.title] = EasyDict(group_dict)
    
    trainer_args = arg_groups["trainer"]


    # --------------------------
    # 3. MLflow Configuration
    # --------------------------
    if args.disable_mlflow_logging:
        config.mlflow = None

    if config.mlflow:
        print("MLflow Configuration:")
        pprint.pprint(config.mlflow)

        if config.mlflow.experiment_name is None:
            config.mlflow.experiment_name = args.partition
        
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
    subsetting_matrix = cardio_mesh.paths.get_subsetting_matrix(partition := args.partition)
    mean_shape        = cardio_mesh.paths.get_mean_shape(partition)
    template_fhm_mesh = cardio_mesh.load_full_heart_mesh(ONE_RANDOM_ID, timeframe=END_DIASTOLE)
    
    # This adds to the mesh the valve surfaces that close up the different chambers
    if args.use_closed_chambers:
        closed_chamber = cardio_mesh.close_chamber(args.partition)

    cardiac_dataset = CardiacMeshPopulationDataset(
        root_path=cardio_mesh.MESHES_DIR, 
        procrustes_transforms=cardio_mesh.paths.get_procrustes_file(partition),
        faces=(faces := template_fhm_mesh[closed_chamber].f),
        subsetting_matrix=subsetting_matrix,
        template_mesh=(mesh_template := EasyDict({"v": mean_shape, "f": faces})),
        N_subj=(N_subj := 10000),
        phases_filter=get_n_equispaced_timeframes(args.n_timeframes)
    )
    
    (mesh_dm := CardiacMeshPopulationDM(cardiac_dataset, batch_size=config.batch_size)).setup()

    # --------------------------
    # 5. Define Model
    # --------------------------
    
    coma_matrices = get_coma_matrices(config, mesh_dm.dataset.template_mesh, partition)
    (coma_args := get_coma_args(config)).update(coma_matrices)
  
    enc_config = EasyDict({k: v for k, v in coma_args.items() if k in ENCODER_ARGS})
    encoder = Encoder3DMesh(**enc_config, n_timeframes=args.n_timeframes)
    
    enc_config.latent_dim = config.network_architecture.latent_dim_c + config.network_architecture.latent_dim_s 
    h = encoder.forward_conv_stack(next(iter(mesh_dm.train_dataloader())).s_t, preserve_graph_structure=False)
    
    z_aggr    = FCN_Aggregator(features_in=args.n_timeframes * h.shape[-1], features_out=enc_config.latent_dim)
    t_encoder = EncoderTemporalSequence(encoder3d=encoder, z_aggr_function=z_aggr, is_variational=coma_args.is_variational)   
    
    decoder_content = DecoderContent({k: v for k, v in coma_args.items() if k in DECODER_C_ARGS})
    decoder_style   = DecoderStyle({k: v for k, v in coma_args.items() if k in DECODER_S_ARGS}, phase_embedding_method="exp_v1", n_timeframes=args.n_timeframes)
    t_decoder       = DecoderTemporalSequence(decoder_content, decoder_style, is_variational=coma_args.is_variational)
    
    t_ae = AutoencoderTemporalSequence(encoder=t_encoder, decoder=t_decoder, is_variational=coma_args.is_variational)
    
    lit_module = CoMA_Lightning(
        model=t_ae, 
        loss_params=config.loss, 
        optimizer_params=config.optimizer,
        additional_params=config,
        mesh_template=mesh_template
    )

    # --------------------------
    # 6. Configure Trainer and Run
    # --------------------------
    callbacks = [ 
        early_stopping, 
        model_checkpoint, 
        rich_model_summary,
        progress_bar, 
        MemoryUsageCallback() 
    ]
    # callbacks = [ EarlyStopping(monitor="val_loss", mode="min", patience=trainer_args.patience) ]
    (trainer_kwargs := dict(callbacks=callbacks)).update({ k: getattr(trainer_args, k) for k in ["devices", "accelerator", "min_epochs", "max_epochs", "logger", "precision"] })
    
    trainer = pl.Trainer(**trainer_kwargs)
    main(lit_module, mesh_dm, trainer, config.mlflow)