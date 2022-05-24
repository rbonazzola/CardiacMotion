import os, sys
sys.path.append("..")
from config.load_config import load_yaml_config, to_dict, recursive_namespace
import torch
import pytorch_lightning as pl

from data.synthetic.SyntheticMeshPopulation import SyntheticMeshPopulation
from data.SyntheticDataModules import SyntheticMeshesDataset
from torch.utils.data import DataLoader
from utils.helpers import get_datamodule

config = load_yaml_config("../config_files/config_folded_c_and_s.yaml")
dm = get_datamodule(config)


from models.Model3D import Decoder3DMesh
import models.Model3D

from importlib import reload

import models
models = reload(module=models)
Model3D = reload(module=models.Model3D)
Decoder3DMesh = Model3D.Decoder3DMesh

coma_args["num_conv_filters_dec"] = coma_args["num_conv_filters_dec_c"]

decoder_args = [
    "num_features",
    "n_layers",
    "n_nodes",
    "num_conv_filters_dec",
    "cheb_polynomial_order",
    "latent_dim_content",
    "is_variational",
    "upsample_matrices",
    "adjacency_matrices",
    "activation_layers"
]

dec_config = {k: v for k,v in coma_args.items() if k in decoder_args}

Decoder3DMesh(**dec_config)