import torch
import os
import sys;
import numpy as np

import pytorch_lightning as pl

from data.DataModules import CardiacMeshPopulationDM
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

    sparse_tensor = torch.sparse_coo_tensor(i, v, torch.Size(shape))
    
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


def get_n_equispaced_timeframes(n_timeframes):
    
    assert n_timeframes in {2, 5, 10, 25, 50}, f"Number of timeframes (args.n_timeframes) is {args.n_timeframes} which does not divide 50."
    phases = 1 + (50 / n_timeframes) * np.array(range(n_timeframes))    