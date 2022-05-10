import torch
import sys; sys.path.append("..")

from models.model_c_and_s import Coma4D_C_and_S
from models.ComaLightningModule import CoMA

from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import RichModelSummary

from data.DataModules import CardiacMeshPopulationDM
from data.SyntheticDataModules import SyntheticMeshesDM
from utils import mesh_operations
from VTKHelpers.CardiacMesh import Cardiac3DMesh as Mesh
from subprocess import check_output
import shlex

def scipy_to_torch_sparse(scp_matrix):
    import numpy as np
    indices = np.vstack((scp_matrix.row, scp_matrix.col))
    i = torch.LongTensor(indices)
    values = scp_matrix.data
    v = torch.FloatTensor(values)
    shape = scp_matrix.shape

    sparse_tensor = torch.sparse.FloatTensor(i, v, torch.Size(shape))
    return sparse_tensor


def get_current_commit_hash():
    return check_output(shlex.split("git rev-parse HEAD")).decode().strip()


def get_datamodule(config):
    '''

    '''

    # TODO: MERGE THESE TWO INTO ONE DATAMODULE CLASS
    if config.dataset.data_type.startswith("cardiac"):
        return CardiacMeshPopulationDM(cardiac_population=data, batch_size=config.batch_size)
    elif config.dataset.data_type.startswith("synthetic"):
        return SyntheticMeshesDM(
            batch_size=config.batch_size,
            data_params=config.dataset.parameters.__dict__,
            preprocessing_params=config.dataset.preprocessing
        )


def get_dm_model_trainer(config, trainer_args):
    '''
    Returns a tuple of (PytorchLightning datamodule, PytorchLightning model, PytorchLightning trainer)
    '''

    # LOAD DATA
    dm = get_datamodule(config)
    dm.setup()

    # Initialize PyTorch model
    coma_args = get_coma_args(config, dm)

    coma4D = Coma4D_C_and_S(**coma_args)

    # Initialize PyTorch Lightning module
    model = CoMA(coma4D, config)

    # trainer
    trainer = pl.Trainer(
        callbacks=[
            EarlyStopping(monitor="val_loss", mode="min", patience=10),
            RichModelSummary(max_depth=-1)
        ],
        gpus=[trainer_args.gpus],
        auto_select_gpus=trainer_args.auto_select_gpus,
        min_epochs=trainer_args.min_epochs, max_epochs=trainer_args.max_epochs,
        auto_scale_batch_size=trainer_args.auto_scale_batch_size,
        logger=trainer_args.logger,
        precision=trainer_args.precision,
        overfit_batches=args.overfit_batches,
    )

    return dm, model, trainer


def get_coma_matrices(config, dm, cache=True, from_cached=True):
    '''
    :param config: configuration Namespace, with a list called "network_architecture.pooling.parameters.downsampling_factors" as attribute.
    :param dm: a PyTorch Lightning datamodule, with attributes train_dataset.dataset.mesh_popu and train_dataset.dataset.mesh_popu.template
    :param cache: if True, will cache the matrices in a pkl file, unless this file already exists.
    :param from_cached: if True, will try to fetch the matrices from a previously cached pkl file.
    :return: a dictionary with keys "downsample_matrices", "upsample_matrices", "adjacency_matrices" and "n_nodes",
    where the first three elements are lists of matrices and the last is a list of integers.
    '''

    mesh_popu = dm.train_dataset.dataset.mesh_popu
    matrices_hash = hash(
        (mesh_popu._object_hash, tuple(config.network_architecture.pooling.parameters.downsampling_factors))) % 1000000
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
        "template": template_mesh
    }


def get_coma_args(config, dm):
    net = config.network_architecture

    convs = net.convolution
    coma_args = {
        "num_features": net.n_features,
        "n_layers": len(convs.channels_enc),  # REDUNDANT
        "num_conv_filters_enc": convs.channels_enc,
        "num_conv_filters_dec_c": convs.channels_dec_c,
        "num_conv_filters_dec_s": convs.channels_dec_s,
        "polygon_order": convs.parameters.polynomial_degree,
        "latent_dim_content": net.latent_dim_c,
        "latent_dim_style": net.latent_dim_s,
        "is_variational": config.loss.regularization.weight != 0,
        "mode": "testing",
        "n_timeframes": config.dataset.parameters.T,
        "phase_input": net.phase_input,
        "z_aggr_function": net.z_aggr_function
    }

    matrices = get_coma_matrices(config, dm, from_cached=False)
    coma_args.update(matrices)
    return coma_args


def get_mlflow_parameters(config):
    loss = config.loss
    net = config.network_architecture
    loss_params = {
        "w_kl": loss.regularization.weight,
        "w_s": loss.reconstruction_s.weight
    }
    net_params = {
        "latent_dim_s": net.latent_dim_s,
        "latent_dim_c": net.latent_dim_c,
        "z_aggr_function": net.z_aggr_function,
        "convolution_type": net.convolution.type,
        "n_channels_enc": net.convolution.channels_enc,
        "n_channels_dec_c": net.convolution.channels_dec_c,
        "n_channels_dec_s": net.convolution.channels_dec_s,
        "reduction_factors": net.pooling.parameters.downsampling_factors,
        "phase_input": net.phase_input
    }

    mlflow_parameters = {
        "platform": check_output(["hostname"]).strip().decode(),
        **loss_params,
        **net_params,
    }

    return mlflow_parameters


###
def get_mlflow_dataset_params(config):
    '''
    Returns a dictionary containing the dataset parameters, to be logged to MLflow.
    '''
    d = config.dataset

    mlflow_dataset_params = {
        "dataset_type": "synthetic",
        "dataset_max_static_amplitude": d.parameters.amplitude_static_max,
        "dataset_max_dynamic_amplitude": d.parameters.amplitude_dynamic_max,
        "dataset_n_timeframes": d.parameters.T,
        "dataset_freq_max": d.parameters.freq_max,
        "dataset_l_max": d.parameters.l_max,
        "dataset_resolution": d.parameters.mesh_resolution,
        "dataset_complexity_c": (d.parameters.l_max + 1) ** 2,
        "dataset_complexity_s": ((d.parameters.l_max + 1) ** 2) * d.parameters.freq_max,
        "dataset_complexity": ((d.parameters.l_max + 1) ** 2) * (d.parameters.freq_max + 1),
        "dataset_random_seed": d.parameters.random_seed,
        "dataset_template": "icosphere",  # TODO: add this as parameter in the configuration
        "dataset_center_around_mean": d.preprocessing.center_around_mean
    }

    return mlflow_dataset_params


def print_auto_logged_info(r):

    tags = {k: v for k, v in r.data.tags.items() if not k.startswith("mlflow.")}
    artifacts = [f.path for f in MlflowClient().list_artifacts(r.info.run_id, "model")]
    print("run_id: {}".format(r.info.run_id))
    print("artifacts: {}".format(artifacts))
    print("params: {}".format(r.data.params))
    print("metrics: {}".format(r.data.metrics))
    print("tags: {}".format(tags))