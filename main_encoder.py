import pytorch_lightning as pl
from pytorch_lightning.loggers import MLFlowLogger
import mlflow.pytorch
from mlflow.tracking import MlflowClient

from config.cli_args import CLI_args, overwrite_config_items

from utils.helpers import *
from torchviz import make_dot

from config.load_config import load_yaml_config, to_dict

import argparse
import pprint

from IPython import embed


###
def main(config, trainer_args):
    #
    dm = get_datamodule(config)
    from models.Model import Encoder3DMesh
    from utils.helpers import get_coma_args
    coma_args = get_coma_args(config, dm)
    coma_args["phase_input"] = False

    encoder_args = [
        "num_features",
        "n_layers",
        "n_nodes",
        "num_conv_filters_enc",
        "cheb_polynomial_order",
        "latent_dim_content",
        "is_variational",
        "phase_input",
        "downsample_matrices",
        "adjacency_matrices",
        "activation_layers"
    ]

    enc_config = {k: v for k, v in coma_args.items() if k in encoder_args}

    from models.Model4D import EncoderTemporalSequence
    cine_encoder = EncoderTemporalSequence(enc_config, z_aggr_function="DFT", n_timeframes=20)

    from models.EncoderPLModule import CineComaEncoder
    PLEncoder = CineComaEncoder(cine_encoder, config)
    trainer = pl.Trainer()
    trainer.fit(PLEncoder, dm)

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
            exp_id = mlflow.create_experiment(config.mlflow.experiment_name,
                                              artifact_location=config.mlflow.artifact_location)
        except:
            # If the experiment already exists, we can just retrieve its ID
            exp_id = mlflow.get_experiment_by_name(config.mlflow.experiment_name).experiment_id

        run_info = {
            "run_id": trainer.logger.run_id,
            "experiment_id": exp_id,
            "run_name": config.mlflow.run_name,
            # "tags": config.additional_mlflow_tags
        }

        with mlflow.start_run(**run_info) as run:

            if config.log_computational_graph:
                yhat = model(next(iter(dm.train_dataloader()))[0])
                make_dot(yhat, params=dict(list(model.named_parameters()))).render("comp_graph_network", format="png")
                mlflow.log_figure("comp_graph_network.png")

            mlflow.log_params(get_mlflow_parameters(config))
            mlflow.log_params(get_mlflow_dataset_params(config))
            # mlflow.log_params(config.additional_mlflow_params)

            trainer.fit(model, datamodule=dm)
            trainer.test(datamodule=dm)  # Generates metrics for the full test dataset
            trainer.predict(ckpt_path='best', datamodule=dm)  # Generates figures for a few samples
            # print_auto_logged_info(mlflow.get_run(run_id=run.info.run_id))
    else:
        trainer.fit(model, datamodule=dm)
        result = trainer.test(datamodule=dm)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Pytorch Trainer for Convolutional Mesh Autoencoders",
        argument_default=argparse.SUPPRESS
    )

    # to avoid a little bit of boilerplate
    for k, v in CLI_args.items():
        parser.add_argument(*k, **v)

    # add arguments specific to the PyTorch Lightning trainer.
    parser = pl.Trainer.add_argparse_args(parser)

    args = parser.parse_args()

    ### Load configuration
    if not os.path.exists(args.yaml_config_file):
        logger.error("Config not found" + args.yaml_config_file)

    ref_config = load_yaml_config(args.yaml_config_file)

    try:
        config_to_replace = args.config
        config = overwrite_config_items(ref_config, config_to_replace)
    except AttributeError:
        # If there are no elements to replace
        pass

    # TOFIX: args contains other arguments that do not correspond to the trainer
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