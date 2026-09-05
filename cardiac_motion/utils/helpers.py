import torch
import hashlib
import logging
import os
import sys;
import time
import numpy as np

import pytorch_lightning as pl

from data.DataModules import CardiacMeshPopulationDM
from utils import mesh_operations
from utils.mesh_operations import Mesh

import pickle as pkl

from easydict import EasyDict
from typing import Union, Mapping, Sequence

logger = logging.getLogger(__name__)


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

    cache_key = "|".join(
        [
            str(partition),
            "downsample=" + ",".join(str(x) for x in downsample_factors),
            f"vertices={len(template.v)}",
            f"faces={len(template.f)}",
        ]
    )
    matrices_hash = hashlib.sha1(cache_key.encode("utf-8")).hexdigest()[:12]

    cached_file = f"data/cached/matrices/{matrices_hash}.pkl"
    logger.info(
        "COMA matrices cache key=%s file=%s",
        cache_key,
        cached_file,
    )

    if from_cached and os.path.exists(cached_file):
        start = time.perf_counter()
        logger.info("Loading cached COMA matrices from %s", cached_file)
        A_t, D_t, U_t, n_nodes = pkl.load(open(cached_file, "rb"))
        logger.info("Loaded cached COMA matrices in %.2fs; n_nodes=%s", time.perf_counter() - start, n_nodes)
    else:
        start = time.perf_counter()
        logger.info(
            "Generating COMA matrices from template: vertices=%d, faces=%d, downsample_factors=%s",
            len(template.v),
            len(template.f),
            downsample_factors,
        )
        template_mesh = Mesh(template.v, template.f)
        M, A, D, U = mesh_operations.generate_transform_matrices(
            template_mesh, downsample_factors,
        )
        n_nodes = [len(M[i].v) for i in range(len(M))]
        logger.info("Converting scipy sparse matrices to torch sparse tensors")
        A_t, D_t, U_t = ([scipy_to_torch_sparse(x).float() for x in X] for X in (A, D, U))
        if cache:
            os.makedirs(os.path.dirname(cached_file), exist_ok=True)
            logger.info("Saving COMA matrices cache to %s", cached_file)
            with open(cached_file, "wb") as ff:
                pkl.dump((A_t, D_t, U_t, n_nodes), ff)
        logger.info("Generated COMA matrices in %.2fs; n_nodes=%s", time.perf_counter() - start, n_nodes)

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
    
    valid_timeframes = {2, 5, 10, 25, 50}
    assert n_timeframes in valid_timeframes, (
        f"Number of timeframes is {n_timeframes}, but expected one of "
        f"{sorted(valid_timeframes)}."
    )
    phases = 1 + (50 / n_timeframes) * np.array(range(n_timeframes))
    return phases.astype(int).tolist()
