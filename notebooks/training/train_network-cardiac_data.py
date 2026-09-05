# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.17.3
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %%
import os
os.environ["CARDIAC_MOTION_REPO"] = f"{os.environ['HOME']}/Rodrigo_repos/CardiacMotion"
repo_dir = os.environ.get("CARDIAC_MOTION_REPO", "kk")
os.chdir(repo_dir)

# %%
from utils import mesh_operations
from utils.helpers import *
from config.load_config import load_yaml_config
from models.Model4D import AutoencoderTemporalSequence
from models.lightning.ComaLightningModule import CoMA_Lightning

from main import get_coma_matrices, get_datamodule
from data.DataModules import CardiacMeshPopulationDM
from data.SyntheticDataModules import SyntheticMeshesDM
from pytorch_lightning.loggers import MLFlowLogger

import pytorch_lightning as pl
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

from easydict import EasyDict
from models.Model4D import ENCODER_ARGS, DECODER_C_ARGS, DECODER_S_ARGS

import pytorch_lightning as pl
from pytorch_lightning.loggers import MLFlowLogger

import mlflow
from mlflow.tracking import MlflowClient

from config.cli_args import CLI_args, overwrite_config_items
from config.load_config import load_yaml_config, to_dict

from utils.helpers import *
from utils.mlflow_helpers import get_mlflow_parameters, get_mlflow_dataset_params

# %%
get_datamodule()

# %%
config = load_yaml_config("config_folded_c_and_s.yaml")
dm = get_datamodule(config.dataset, batch_size=config.batch_size)

# %%

coma_args = get_coma_args(config, dm)

enc_config = {k: v for k, v in coma_args.items() if k in ENCODER_ARGS}
dec_c_config = {k: v for k,v in coma_args.items() if k in DECODER_C_ARGS}
dec_s_config = {k: v for k,v in coma_args.items() if k in DECODER_S_ARGS}

ae = AutoencoderTemporalSequence(
    enc_config, 
    dec_c_config, 
    dec_s_config,
    z_aggr_function=config.network_architecture.z_aggr_function,
    n_timeframes=config.dataset.parameters.T
)

# %%
callbacks = [
    EarlyStopping(monitor="val_loss", mode="min", patience=3),
    #RichModelSummary(max_depth=-1)
]

trainer = pl.Trainer(callbacks=callbacks)

# %%
# Initialize PyTorch Lightning module
ptl_module = CoMA_Lightning(ae, config)

# %%
trainer.fit(ptl_module, dm)

# %%
dm.test_dataset[0]['s_t'].shape

# %%
ptl_module.model

# %%

# %%

# %%
