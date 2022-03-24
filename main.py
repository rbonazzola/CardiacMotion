import pickle as pkl

import pytorch_lightning as pl
from pytorch_lightning.callbacks.early_stopping import EarlyStopping 
from pytorch_lightning.callbacks import RichModelSummary


from IPython import embed
from utils import mesh_operations
from utils.helpers import *
from models.model import Coma4D
from models.model_c_and_s import Coma4D_C_and_S
from models.coma_ml_module import CoMA
from torchviz import make_dot

import mlflow.pytorch
from mlflow.tracking import MlflowClient

from psbody.mesh import Mesh
from config.load_config import load_yaml_config, to_dict

from data.DataModules import CardiacMeshPopulationDM
from data.SyntheticDataModules import SyntheticMeshesDM
from pytorch_lightning.loggers import MLFlowLogger

import argparse
import pprint

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
        "n_layers": len(convs.channels_enc),  # REDUNDANT
        "num_conv_filters_enc": convs.channels_enc,
        "num_conv_filters_dec_c": convs.channels_dec_c,
        "num_conv_filters_dec_s": convs.channels_dec_s,
        "polygon_order": convs.parameters.polynomial_degree,
        "latent_dim_content": net.latent_dim_c,
        "latent_dim_style": net.latent_dim_s,
        "is_variational": config.loss.regularization.weight != 0,
        "mode": "testing",
    }

    matrices = get_coma_matrices(config, dm, from_cached=False)
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

    coma4D = Coma4D_C_and_S(**coma_args)

    # Initialize PyTorch Lightning module
    model = CoMA(coma4D, config)

    # trainer
    trainer = pl.Trainer(
        callbacks=[
            EarlyStopping(monitor="val_loss", mode="min", patience=10),
            RichModelSummary(max_depth=-1)            
        ],
        gpus=trainer_args.gpus,
        auto_select_gpus=trainer_args.auto_select_gpus,
        min_epochs=trainer_args.min_epochs, max_epochs=trainer_args.max_epochs,
        auto_scale_batch_size=trainer_args.auto_scale_batch_size,
        logger=trainer_args.logger,
        precision=trainer_args.precision,
        overfit_batches=args.overfit_batches,
    )

    return dm, model, trainer

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
        "convolution_type": net.convolution.type,
        "n_channels_enc": net.convolution.channels_enc,
        "n_channels_dec_c": net.convolution.channels_dec_c,
        "n_channels_dec_s": net.convolution.channels_dec_s,
        "reduction_factors": net.pooling.parameters.downsampling_factors,
        "phase_input": net.phase_input
    }

    mlflow_parameters = {
        "platform": check_output(["hostname"]).strip().decode(),
        ** loss_params,
        ** net_params,
    }

    return mlflow_parameters


def get_mlflow_dataset_params(config):

    d = config.dataset

    mlflow_dataset_params = {
         "dataset_type" : d.data_type,
         "dataset_max_static_amplitude" : d.parameters.amplitde_static_max,
         "dataset_max_dynamic_amplitude" : d.parameters.amplitde_dynamic_max,
         "dataset_n_timeframes" : d.parameters.T,
         "dataset_freq_max" : d.parameters.freq_max,
         "dataset_l_max" : d.parameters.l_max,
         "dataset_complexity_c": (d.parameters.l_max + 1) ** 2,
         "dataset_complexity_s": ((d.parameters.l_max + 1) ** 2) * d.parameters.freq_max,
         "dataset_complexity": ((d.parameters.l_max + 1) ** 2) * (d.parameters.freq_max + 1),
         "dataset_random_seed" : d.parameters.random_seed,
         "dataset_template": "icosphere", #TODO: add this as parameter in the configuration
         "dataset_center_around_mean" : d.preprocessing.center_around_mean
    }

    return mlflow_dataset_params


def main(config, trainer_args):

    #

    if config.mlflow:
        if config.mlflow.experiment_name is None:
            config.mlflow.experiment_name = "default"
        trainer_args.logger = MLFlowLogger(
            experiment_name=config.mlflow.experiment_name,
            tracking_uri=config.mlflow.tracking_uri,
            artifact_location=config.mlflow.artifact_location
        )
        mlflow.set_tracking_uri(config.mlflow.tracking_uri)
    else:
        trainer_args.logger = None

    dm, model, trainer = get_dm_model_trainer(config, trainer_args)

    # gg = hl.build_graph(model, next(iter(dm.train_dataloader()))[0] )

    if config.mlflow:

        mlflow.pytorch.autolog()
        try:
            exp_id = mlflow.create_experiment(config.mlflow.experiment_name, artifact_location=config.mlflow.artifact_location)
        except:
          # If the experiment already exists, we can just retrieve its ID
            exp_id = mlflow.get_experiment_by_name(config.mlflow.experiment_name).experiment_id

        run_info = {
            "run_id": trainer.logger.run_id,
            "experiment_id": exp_id,
            "run_name": config.mlflow.run_name,
            "tags": config.additional_mlflow_tags
        }

        with mlflow.start_run(**run_info) as run:
            
            if config.log_computational_graph:
                yhat = model(next(iter(dm.train_dataloader()))[0])
                make_dot(yhat, params=dict(list(model.named_parameters()))).render("comp_graph_network", format="png")
                mlflow.log_figure("comp_graph_network.png")

            mlflow.log_params(get_mlflow_parameters(config))
            mlflow.log_params(get_mlflow_dataset_params(config))
            mlflow.log_params(config.additional_mlflow_params)

            trainer.fit(model, datamodule=dm)
            result = trainer.test(datamodule=dm)
            # print_auto_logged_info(mlflow.get_run(run_id=run.info.run_id))
    else:
        trainer.fit(model, datamodule=dm)
        result = trainer.test(datamodule=dm)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Pytorch Trainer for Convolutional Mesh Autoencoders",
        argument_default=argparse.SUPPRESS
    )

    from config.cli_args import CLI_args, ArgumentAction, overwrite_config_items

    #to avoid a little bit of boilerplate
    for k, v in CLI_args.items():
        parser.add_argument(*k, **v)

    # add arguments specific to the PyTorch Lightning trainer.
    parser = pl.Trainer.add_argparse_args(parser)

    args = parser.parse_args()

    ### Load configuration
    if not os.path.exists(args.yaml_config_file):
        logger.error("Config not found" + args.yaml_config_file)

    ref_config = load_yaml_config(args.yaml_config_file)
    config_to_replace = args.config
    config = overwrite_config_items(ref_config, config_to_replace)

    #TOFIX: args contains other arguments that do not correspond to the trainer
    trainer_args = args

    config.log_computational_graph = args.log_computational_graph
    if args.disable_mlflow_logging:
        config.mlflow = None
      
    if args.show_config or args.dry_run:
        pp = pprint.PrettyPrinter(indent=2, compact=True)
        pp.pprint(to_dict(config))
        if args.dry_run:
            exit()

    main(config, trainer_args)
