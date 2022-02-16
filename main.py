import pickle as pkl
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
import pytorch_lightning as pl

from IPython import embed
from utils import mesh_operations
from utils.helpers import *
from models.model import Coma4D
from models.coma_ml_module import CoMA

import mlflow.pytorch
from mlflow.tracking import MlflowClient

from psbody.mesh import Mesh
from config.load_config import load_config

from data.DataModules import CardiacMeshPopulationDM
from data.SyntheticDataModules import SyntheticMeshesDM
from pytorch_lightning.loggers import MLFlowLogger

import argparse

def get_matrices(config, dm, cache=True, from_cached=True):

    mesh_popu = dm.train_dataset.dataset.mesh_popu
    matrices_hash = hash((mesh_popu._object_hash, tuple(config.network_architecture.pooling.parameters.downsampling_factors))) % 1000000
    cached_file = f"data/cached/matrices/{matrices_hash}.pkl"
    
    if from_cached and os.path.exists(cached_file):
        A_t, D_t, U_t, n_nodes = pkl.load(open(cached_file, "rb"))
    else:
          template_mesh = Mesh(mesh_popu.template.vertices, mesh_popu.template.faces)
          M, A, D, U = mesh_operations.generate_transform_matrices(
            template_mesh, config.network_architecture.pooling.parameters.downsampling_factors,
          )
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
    }


def get_coma_args(config, dm):

    net = config.network_architecture

    convs = net.convolution
    coma_args = {
        "num_features": net.n_features,
        "n_layers": len(convs.channels),  # REDUNDANT
        "num_conv_filters": convs.channels,
        "polygon_order": convs.parameters.polynomial_degree,
        "latent_dim": net.latent_dim,
        "is_variational": config.loss.regularization.weight != 0,
        "mode": "testing",
    }

    matrices = get_matrices(config, dm, from_cached=False)
    coma_args.update(matrices)
    return coma_args


def get_datamodule(config):

    #with open(config.dataset.cached_file, "rb") as ff:
    #    data = pkl.load(ff)

    # TODO: MERGE THESE TWO INTO ONE DATAMODULE CLASS
    if config.dataset.data_type.startswith("cardiac"):
        return CardiacMeshPopulationDM(cardiac_population=data, batch_size=config.batch_size)
    elif config.dataset.data_type.startswith("synthetic"):
        return SyntheticMeshesDM(
            batch_size=config.batch_size, 
            data_params=config.dataset.parameters.__dict__, 
            preprocessing_params=config.dataset.preprocessing
        )


def print_auto_logged_info(r):

    tags = {k: v for k, v in r.data.tags.items() if not k.startswith("mlflow.")}
    artifacts = [f.path for f in MlflowClient().list_artifacts(r.info.run_id, "model")]
    print("run_id: {}".format(r.info.run_id))
    print("artifacts: {}".format(artifacts))
    print("params: {}".format(r.data.params))
    print("metrics: {}".format(r.data.metrics))
    print("tags: {}".format(tags))


def get_dm_model_trainer(config, trainer_args):

    # LOAD DATA
    dm = get_datamodule(config)
    dm.setup()

    # Initialize PyTorch model
    coma_args = get_coma_args(config, dm)
    coma4D = Coma4D(**coma_args)

    # Initialize PyTorch Lightning module
    model = CoMA(coma4D, config)

    # trainer
    trainer = pl.Trainer(
        callbacks=[EarlyStopping(monitor="val_loss", mode="min")],
        gpus=trainer_args.gpus,
        min_epochs=trainer_args.min_epochs, max_epochs=trainer_args.max_epochs,
        auto_scale_batch_size=trainer_args.auto_scale_batch_size,
        logger=trainer_args.logger
    )

    return dm, model, trainer

def get_mlflow_parameters(config):

    mlflow_parameters = {
      "platform": check_output(["hostname"]).strip().decode(),
      "w_kl": config.loss.regularization.weight,
      "latent_dim": config.network_architecture.latent_dim,
      "convolution_type": config.network_architecture.convolution.type,
      "n_channels": config.network_architecture.convolution.channels,
      "reduction_factors": config.network_architecture.pooling.parameters.downsampling_factors,
      "center_around_mean": config.dataset.preprocessing.center_around_mean,
      "phase_input": config.network_architecture.phase_input
    }

    return mlflow_parameters


def main(config, trainer_args):

    #
    if config.log_to_mlflow:
        if config.mlflow.experiment_name is None:
            config.mlflow.experiment_name = "default"
        trainer_args.logger = MLFlowLogger(experiment_name=config.mlflow.experiment_name, tracking_uri="file:./mlruns")
    else:
        trainer_args.logger = None

    dm, model, trainer = get_dm_model_trainer(config, trainer_args)
        
    if config.log_to_mlflow:
        mlflow.pytorch.autolog()
        try:
            exp_id = mlflow.create_experiment(config.mlflow.experiment_name)
        except:
          # If the experiment already exists, we can just retrieve its ID
            exp_id = mlflow.get_experiment_by_name(config.mlflow.experiment_name).experiment_id
        
        with mlflow.start_run(run_id=trainer.logger.run_id, experiment_id=exp_id, run_name=config.mlflow.run_name) as run:
            mlflow.log_params(get_mlflow_parameters(config))
            trainer.fit(model, datamodule=dm)
            result = trainer.test(datamodule=dm)
            print_auto_logged_info(mlflow.get_run(run_id=run.info.run_id))
    else:
        trainer.fit(model, datamodule=dm)
        result = trainer.test(datamodule=dm)


if __name__ == "__main__":

    def overwrite_config_items(config, args):
        for attr, value in args.__dict__.items():
            if attr in config.keys() and value is not None:
                config[attr] = value

    parser = argparse.ArgumentParser(
        description="Pytorch Trainer for Convolutional Mesh Autoencoders"
    )

    CLI_args = {
      ("-c", "--conf",):  { 
          "help": "path of config file", 
          "default": "config/config_test.yaml" }, 
      ("--w_kl",): { 
          "help":"Dimension of the latent space. If provided will overwrite the batch size from the configuration file.",
          "type": int, "default": None }, 
      ("--latent_dim",): { 
          "help": "Weight of the Kullback-Leibler regularization term. If provided will overwrite the batch size from the configuration file.",
          "type": float, "default": None }, 
      ("--batch_size",): { 
          "help":"Training batch size. If provided will overwrite the batch size from the configuration file.",
          "type": int, "default": None }, 
      ("--disable_mlflow_logging",): { 
          "help": "Set this flag if you don't want to log the run's data to MLflow.",
          "default": False, "action": "store_true", }
    #    ("--dry-run",): {
    #    default=False,
    #    action="store_true",
    #    help="Dry run: just prints out the parameters of the execution but performs no training.",
    #  }
    }
    
    #to avoid a little bit of boilerplate
    for k, v in CLI_args.items():
        print(k,v)
        parser.add_argument(*k, **v)

    # add arguments specific to the 
    parser = pl.Trainer.add_argparse_args(parser)   

    args = parser.parse_args()

    ### Load configuration
    if not os.path.exists(args.conf):
        logger.error("Config not found" + args.conf)

    #TOFIX: args contains other arguments that do not correspond to the trainer
    trainer_args = args

    config = load_config(args.conf, args)
    config.log_to_mlflow = not args.disable_mlflow_logging
    
    main(config, trainer_args)
