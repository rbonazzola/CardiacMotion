import pytorch_lightning as pl
from pytorch_lightning.loggers import MLFlowLogger
import mlflow.pytorch
from mlflow.tracking import MlflowClient

from config.cli_args import CLI_args, overwrite_config_items

from utils.helpers import *
from utils.mlflow_helpers import get_mlflow_parameters, get_mlflow_dataset_params
from torchviz import make_dot

from config.load_config import load_yaml_config, to_dict

import argparse
import pprint

from IPython import embed

from utils.helpers import get_coma_args


###
def main(config, trainer_args):

    #
    dm = get_datamodule(config)
    coma_args = get_coma_args(config, dm)



    else:

    if config.mlflow:
        if config.mlflow.experiment_name is None:
            config.mlflow.experiment_name = "Decoder only"
        exp_info = {
            "experiment_name": config.mlflow.experiment_name,
            "artifact_location": config.mlflow.artifact_location
        }
        trainer_args.logger = MLFlowLogger(
            tracking_uri=config.mlflow.tracking_uri,
            **exp_info,
        )
        mlflow.set_tracking_uri(config.mlflow.tracking_uri)
    else:
        trainer_args.logger = None

    trainer = get_lightning_trainer(trainer_args)

    if config.mlflow:

        mlflow.pytorch.autolog()
        try:
            exp_id = mlflow.create_experiment(**exp_info)
        except:
            # If the experiment already exists, we can just retrieve its ID
            exp_id = mlflow.get_experiment_by_name(exp_info["experiment_name"]).experiment_id

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

            mlflow_params = get_mlflow_parameters(config)
            mlflow_params.update(get_mlflow_dataset_params(config))
            mlflow.log_params(mlflow_params)
            # mlflow.log_params(config.additional_mlflow_params)

            trainer.fit(model, datamodule=dm)
            trainer.test(datamodule=dm)  # Generates metrics for the full test dataset
            # trainer.predict(ckpt_path='best', datamodule=dm)  # Generates figures for a few samples
            # print_auto_logged_info(mlflow.get_run(run_id=run.info.run_id))
    else:
        trainer.fit(model, datamodule=dm)
        trainer.test(datamodule=dm)


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

    print(config.mlflow)
    main(config, trainer_args)
