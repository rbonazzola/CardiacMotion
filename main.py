import sys
import utils.VTKHelpers
# sys.path.append("utils/VTKHelpers/")

from CardiacMesh import Cardiac3DMesh, Cardiac4DMesh, CardiacMeshPopulation
from models import layers

import yaml
from pprint import pprint
from argparse import Namespace
import logging

import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader, random_split
import pytorch_lightning as pl

import ipywidgets as widgets
from IPython.display import display, HTML

import os
import pickle as pkl
from utils import mesh_operations
from utils.helpers import *
from models.model import Coma4D

import mlflow.pytorch
from mlflow.tracking import MlflowClient

from config.load_config import load_config


optimizers_menu = {
  "adam": torch.optim.Adam(coma.parameters(), lr=lr, betas=(0.5,0.99), weight_decay=weight_decay),
  "sgd": torch.optim.SGD(coma.parameters(), lr=lr, weight_decay=weight_decay, momentum=0.9)
}

losses_menu = {
  "l1": {"name": "L1", "function": F.l1_loss},
  "mse": {"name": "MSE", "function": F.mse_loss}
}

def print_auto_logged_info(r):

    tags = {k: v for k, v in r.data.tags.items() if not k.startswith("mlflow.")}
    artifacts = [f.path for f in MlflowClient().list_artifacts(r.info.run_id, "model")]
    print("run_id: {}".format(r.info.run_id))
    print("artifacts: {}".format(artifacts))
    print("params: {}".format(r.data.params))
    print("metrics: {}".format(r.data.metrics))
    print("tags: {}".format(tags))


def main(config):

    dm = CardiacMeshPopulationDM(config.root_folder, batch_size=2)

    A_t, D_t, U_t, n_nodes = pkl.load(open("data/cached/matrices.pkl", "rb"))

    # init model

    coma_args = {
      "num_features": config.network_architecture.n_features,
      "n_layers": len(config.network_architecture.convolution.parameters.channels), # REDUNDANT
      "num_conv_filters": config.network_architecture.convolution.parameters.channels,
      "polygon_order": config.network_architecture.convolution.parameters.polynomial_degree,
      "latent_dim": config.network_architecture.latent_dim,
      "is_variational": config.loss.regularization_loss.weight != 0,
      "downsample_matrices": D_t,
      "upsample_matrices": U_t, 
      "adjacency_matrices": A_t,
      "n_nodes": n_nodes, 
      "mode": "testing"
    }
        
    if config.log_to_mflow:
        mlflow.pytorch.autolog()

    coma4D = Coma4D(**coma_args)
    model = CoMA(coma4D, config)

    # train
    trainer = pl.Trainer()

    if config.log_to_mflow:
        with mlflow.start_run() as run:
            trainer.fit(model, datamodule=dm)
    else:
        trainer.fit(model, datamodule=dm)
    
    
if __name__ == '__main__':

    import argparse
    
    def overwrite_config_items(config, args):
      for attr, value in args.__dict__.items():
        if attr in config.keys() and value is not None:
          config[attr] = value
       
    parser = argparse.ArgumentParser(description='Pytorch Trainer for Convolutional Mesh Autoencoders')

    parser.add_argument('-c', '--conf', help='path of config file', default="config_files/default.cfg")
    # parser.add_argument('-od', '--output_dir', default=None, help='path where to store output')
    # parser.add_argument('-id', '--data_dir', default=None, help='path where to fetch input data from')
    # parser.add_argument('--preprocessed_data', default=None, type=str, help='Location of cached input data.')
    # parser.add_argument('--partition', default=None, type=str, help='Cardiac chamber.')
    # parser.add_argument('--procrustes_scaling', default=False, action="store_true", help="Whether to perform scaling transformation after Procrustes alignment (to make mean distance to origin equal to 1).")
    # parser.add_argument('--phase', default=None, help="cardiac phase (1-50|ED|ES)")
    # parser.add_argument('--z', default=None, type=int, help='Number of latent variables.')
    # parser.add_argument('--optimizer', default=None, type=str, help='optimizer (adam or sgd).')
    # parser.add_argument('--epoch', default=None, type=int, help='Maximum number of epochs.')
    # parser.add_argument('--nTraining', default=None, type=int, help='Number of training samples.')
    # parser.add_argument('--nVal', default=None, type=int, help='Number of validation samples.')
    # parser.add_argument('--kld_weight', type=float, default=None, help='Weight of Kullback-Leibler divergence.')
    # parser.add_argument('--learning_rate', type=float, default=None, help='Learning rate.')
    # parser.add_argument('--seed', default=None, help="Seed for PyTorch's Random Number Generator.")
    # parser.add_argument('--stop_if_not_learning', default=None, action="store_true", help='Stop training if losses do not change.')
    # parser.add_argument('--save_all_models', default=False, action="store_true",
    #                     help='Save all models instead of just the best one until the current epoch.')
    # parser.add_argument('--test', default=False, action="store_true", help='Set this flag if you just want to test whether the code executes properly. ')
    parser.add_argument('--log_to_mflow', default=False, action="store_true", help="Set this flag if you want to log the run's data to MLflow.")
    parser.add_argument('--dry-run', dest="dry_run", default=False, action="store_true",
                        help='Dry run: just prints out the parameters of the execution but performs no training.')


    args = parser.parse_args()

    # if args.conf is None:
    #     args.conf = os.path.join(os.path.dirname(__file__), 'default.cfg')
    #     logger.error('configuration file not specified, trying to load '
    #           'it from current directory', args.conf)

    ################################################################################
    ### Load configuration
    if not os.path.exists(args.conf):
        logger.error('Config not found' + args.conf)    

    config = load_config(args.conf)

    config.log_to_mlflow = args.log_to_mlflow

    main(config)


    # if args.data_dir:
    #     config['data_dir'] = args.data_dir

    # if args.output_dir:
    #     config['output_dir'] = args.output_dir

    # if args.test:
    #     # some small values so that the execution ends quickly
    #     config['comments'] = "this is a test"
    #     config['nTraining'] = 500
    #     config['nVal'] = 80
    #     config['epoch'] = 20
    #     config['output_dir'] = "output/test_{TIMESTAMP}"
    # config['test'] = args.test
       
    # overwrite_config_items(config, args)

    # if args.dry_run:
    #   pprint(config)
    #   exit()



