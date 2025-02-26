import torch
import os
import sys;

import pytorch_lightning as pl

from data.DataModules import CardiacMeshPopulationDM
from data.SyntheticDataModules import SyntheticMeshesDataset, SyntheticMeshesDM
from utils import mesh_operations
from utils.mesh_operations import Mesh

import pickle as pkl

from easydict import EasyDict
from typing import Union, Mapping, Sequence


def scipy_to_torch_sparse(scp_matrix):

    import numpy as np
    indices = np.vstack((scp_matrix.row, scp_matrix.col))
    i = torch.LongTensor(indices)
    values = scp_matrix.data
    v = torch.FloatTensor(values)
    shape = scp_matrix.shape

    sparse_tensor = torch.sparse.FloatTensor(i, v, torch.Size(shape))
    return sparse_tensor


def get_datamodule(dataset_params: Mapping, batch_size: Union[int, Sequence[int]], perform_setup: bool=True):

    '''
      Arguments:
        - config:
        - perform_setup: whether to run DataModule.setup()
    '''

    # TODO: MERGE THESE TWO INTO ONE DATAMODULE CLASS
    if dataset_params.data_type.startswith("cardiac"):
        dm = CardiacMeshPopulationDM(cardiac_population=data, batch_size=batch_size)
        
    elif dataset_params.data_type.startswith("synthetic"):
        dataset = SyntheticMeshesDataset(
            data_params=dataset_params.parameters.__dict__,
            preprocessing_params=dataset_params.preprocessing
        )
        data_module = SyntheticMeshesDM(dataset, batch_size=batch_size)

    if perform_setup:
        data_module.setup()

    return data_module


def get_coma_matrices(config, template, partition, from_cached=True, cache=True):
    
    
    '''
    :param downsample_factors: list of downsampling factors, e.g. [2, 2, 2, 2]
    :param template: a Namespace with attributes corresponding to vertices and faces
    :param cache: if True, will cache the matrices in a pkl file, unless this file already exists.
    :param from_cached: if True, will try to fetch the matrices from a previously cached pkl file.
    :return: a dictionary with keys "downsample_matrices", "upsample_matrices", "adjacency_matrices" and "n_nodes",
    where the first three elements are lists of matrices and the last is a list of integers.
    '''

    # mesh_popu = dm.train_dataset.dataset.mesh_popu
    downsample_factors = config.network_architecture.pooling.parameters.downsampling_factors

    matrices_hash = hash((
        hash("1000215"), 
        hash(tuple(downsample_factors)), 
        hash(partition)
    )) % 1000000

    cached_file = f"data/cached/matrices/{matrices_hash}.pkl"

    if from_cached and os.path.exists(cached_file):
        A_t, D_t, U_t, n_nodes = pkl.load(open(cached_file, "rb"))
    else:
        template_mesh = Mesh(template.v, template.f)
        M, A, D, U = mesh_operations.generate_transform_matrices(
            template_mesh, downsample_factors,
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
        "template": template
    }


def get_coma_args(config: Mapping): #, mesh_dataset: torch.utils.data.Dataset):
    
    '''
      Arguments:
        - config: Namespace with all the necessary configuraton.
        - mesh_dataset: torch Dataset with attributes mesh_popu.template with a template Mesh.
    '''

    net = config.network_architecture

    convs = net.convolution
    coma_args = {
        "num_features": net.n_features,
        "n_layers": len(convs.channels_enc),  # REDUNDANT
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

    downsample_factors = config.network_architecture.pooling.parameters.downsampling_factors
    
    # matrices = get_coma_matrices(
    #     downsample_factors,                 
    #     mesh_dataset.mesh_popu.template,         
    #     from_cached=False
    # )
    # coma_args.update(matrices)

    return EasyDict(coma_args)


def get_lightning_module(config: Mapping, dm: pl.LightningDataModule):
    
    '''
      Arguments:
        - config:
        - dm:    
    '''

    # Initialize PyTorch model
    coma_args = get_coma_args(config, dm)

    if config.only_decoder:

        from models.Model4D import DecoderTemporalSequence, DECODER_C_ARGS, DECODER_S_ARGS
        from models.lightning.DecoderLightningModule import TemporalDecoderLightning

        dec_c_config = {k: v for k,v in coma_args.items() if k in DECODER_C_ARGS}
        dec_s_config = {k: v for k,v in coma_args.items() if k in DECODER_S_ARGS}

        decoder = DecoderTemporalSequence(
            dec_c_config, dec_s_config,
            phase_embedding_method="exp",
            n_timeframes=config.dataset.parameters.T
        )

        model = TemporalDecoderLightning(decoder, config)

    elif config.only_encoder:

        from models.Model4D import EncoderTemporalSequence, ENCODER_ARGS
        from models.lightning.EncoderLightningModule import TemporalEncoderLightning

        enc_config = {k: v for k, v in coma_args.items() if k in ENCODER_ARGS}

        encoder = EncoderTemporalSequence(
            enc_config, z_aggr_function=config.network_architecture.z_aggr_function,
            n_timeframes=config.dataset.parameters.T
        )

        model = TemporalEncoderLightning(encoder, config)

    else:
        from models.Model4D import AutoencoderTemporalSequence
        from models.lightning.ComaLightningModule import CoMA_Lightning
        from models.Model4D import EncoderTemporalSequence, ENCODER_ARGS
        from models.Model4D import DecoderTemporalSequence, DECODER_C_ARGS, DECODER_S_ARGS
        
        enc_config = {k: v for k, v in coma_args.items() if k in ENCODER_ARGS}
        dec_c_config = {k: v for k,v in coma_args.items() if k in DECODER_C_ARGS}
        dec_s_config = {k: v for k,v in coma_args.items() if k in DECODER_S_ARGS}
        
        autoencoder = AutoencoderTemporalSequence(
            enc_config, 
            dec_c_config, 
            dec_s_config,
            z_aggr_function=coma_args.z_aggr_function,
            n_timeframes=coma_args.n_timeframes        
        )
        
        # Initialize PyTorch Lightning module
        model = CoMA_Lightning(autoencoder, config)

    return model


##########################################################################################

# PyTorch Lightning Callbacks

from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import RichProgressBar
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.progress.rich_progress import RichProgressBarTheme
from pytorch_lightning.callbacks import RichModelSummary

early_stopping = EarlyStopping(monitor="val_loss", mode="min", patience=10)

model_checkpoint = ModelCheckpoint(monitor='val_loss', save_top_k=1)

rich_model_summary = RichModelSummary(max_depth=-1)

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

def get_lightning_trainer(trainer_args: Mapping):

    '''
      trainer_args:
    '''
    
    # trainer
    trainer_kwargs = {
        "callbacks": [ early_stopping, model_checkpoint, rich_model_summary, progress_bar ],
        # "gpus": trainer_args.gpus,
        "devices": trainer_args.devices,
        "accelerator": trainer_args.accelerator,
        # "auto_select_gpus": trainer_args.auto_select_gpus,
        "min_epochs": trainer_args.min_epochs, "max_epochs": trainer_args.max_epochs,
        # "auto_scale_batch_size": trainer_args.auto_scale_batch_size,
        "logger": trainer_args.logger,
        "precision": trainer_args.precision,
        "overfit_batches": trainer_args.overfit_batches,
        "limit_test_batches": trainer_args.limit_test_batches
    }

    try:
        trainer = pl.Trainer(**trainer_kwargs)
    except:
        trainer_kwargs["gpus"] = None
        trainer = pl.Trainer(**trainer_kwargs)
    return trainer


# def get_dm_model_trainer(config: Mapping, trainer_args: Mapping):
#     
#     '''
#       Arguments:
#         - config:
#         - dm:    
#       
#       Return:
#         a tuple of (PytorchLightning datamodule, PytorchLightning model, PytorchLightning trainer)
#     '''
# 
#     # LOAD DATA
#     dm = get_datamodule(config)
#     model = get_lightning_module(config, dm.dataset)
#     trainer = get_lightning_trainer(trainer_args)
# 
#     return dm, model, trainer