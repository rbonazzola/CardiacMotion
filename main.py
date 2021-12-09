import sys, os
import yaml
import logging
import pickle as pkl
import utils.VTKHelpers

from CardiacMesh import Cardiac3DMesh, Cardiac4DMesh, CardiacMeshPopulation
from models import layers

from pprint import pprint
from argparse import Namespace

import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader, random_split
import pytorch_lightning as pl

import ipywidgets as widgets
from IPython.display import display, HTML

from utils import mesh_operations
from utils.helpers import *
from models.model import Coma4D
from models.coma_ml_module import CoMA

import mlflow.pytorch
from mlflow.tracking import MlflowClient

from config.load_config import load_config

from data.DataModules import CardiacMeshPopulationDataset, CardiacMeshPopulationDM


def get_matrices(config, cached_file):

    M, A, D, U = mesh_operations.generate_transform_matrices(
        Cardiac3DMesh(config.network_architecture.template_mesh), 
        config.network_architecture.pooling.parameters.downsampling_factors
    )

    A_t, D_t, U_t = ([scipy_to_torch_sparse(x) for x in X] for X in (A, D, U))
    n_nodes = [len(M[i].v) for i in range(len(M))]


    if config.network_architeture.cached_matrices is not None:
      A_t, D_t, U_t, n_nodes = pkl.load(open(config.network_architeture.cached_matrices, "rb"))

    return {
        "downsample_matrices": D_t,
        "upsample_matrices": U_t,
        "adjacency_matrices": A_t,
        "n_nodes": n_nodes
    }

def get_datamodule(config):
    
    with open(config.dataset.cached_file) as ff:
        data = pkl.load(ff)
   
    #TODO: MERGE THESE TWO INTO ONE DATAMODULE CLASS 
    if config.dataset.data_type.startswith("cardiac"):
        DataModule = CardiacMeshPopulationDM
    elif config.dataset.data_type.startswith("synthetic"):
        DataModule = SyntheticMeshesDM
      
    dm = DataModule(cardiac_population=data, batch_size=config.optimizer.batch_size)
    return dm

def get_coma_args(config, matrices):
    coma_args = {
        "num_features": config.network_architecture.n_features,
        "n_layers": len(
            config.network_architecture.convolution.parameters.channels
        ),  # REDUNDANT
        "num_conv_filters": config.network_architecture.convolution.parameters.channels,
        "polygon_order": config.network_architecture.convolution.parameters.polynomial_degree,
        "latent_dim": config.network_architecture.latent_dim,
        "is_variational": config.loss.regularization_loss.weight != 0,
    }

    coma_args.update(matrices)

    #{
    #    "downsample_matrices": D_t,
    #    "upsample_matrices": U_t,
    #    "adjacency_matrices": A_t,
    #    "n_nodes": n_nodes,
    #    "mode": "testing",
    #}
    return coma_args

# optimizers_menu = {
#   "adam": torch.optim.Adam(coma.parameters(), lr=lr, betas=(0.5,0.99), weight_decay=weight_decay),
#   "sgd": torch.optim.SGD(coma.parameters(), lr=lr, weight_decay=weight_decay, momentum=0.9)
# }
#
# losses_menu = {
#   "l1": {"name": "L1", "function": F.l1_loss},
#   "mse": {"name": "MSE", "function": F.mse_loss}
# }


def print_auto_logged_info(r):

    tags = {k: v for k, v in r.data.tags.items() if not k.startswith("mlflow.")}
    artifacts = [f.path for f in MlflowClient().list_artifacts(r.info.run_id, "model")]
    print("run_id: {}".format(r.info.run_id))
    print("artifacts: {}".format(artifacts))
    print("params: {}".format(r.data.params))
    print("metrics: {}".format(r.data.metrics))
    print("tags: {}".format(tags))


def main(config):

    # LOAD DATA
    # INIT MODEL
    matrices = get_matrices("data/cached/matrices.pkl")
    coma_args = get_coma_args(config, matrices)
    coma4D = Coma4D(**coma_args)
    model = CoMA(coma4D, config)

    # train
    trainer = pl.Trainer()

    if config.log_to_mlflow:
        mlflow.pytorch.autolog()
        with mlflow.start_run() as run:
            trainer.fit(model, datamodule=dm)
    else:
        trainer.fit(model, datamodule=dm)


if __name__ == "__main__":

    import argparse

    def overwrite_config_items(config, args):
        for attr, value in args.__dict__.items():
            if attr in config.keys() and value is not None:
                config[attr] = value

    parser = argparse.ArgumentParser(
        description="Pytorch Trainer for Convolutional Mesh Autoencoders"
    )

    parser.add_argument(
        "-c", "--conf", help="path of config file", default="config/config.yaml"
    )
    
    parser.add_argument(
        "--log_to_mlflow",
        default=False,
        action="store_true",
        help="Set this flag if you want to log the run's data to MLflow.",
    )
    parser.add_argument(
        "--dry-run",
        dest="dry_run",
        default=False,
        action="store_true",
        help="Dry run: just prints out the parameters of the execution but performs no training.",
    )

    args = parser.parse_args()

    ### Load configuration
    if not os.path.exists(args.conf):
        logger.error("Config not found" + args.conf)

    config = load_config(args.conf)
    config.log_to_mlflow = args.log_to_mlflow
    main(config)
