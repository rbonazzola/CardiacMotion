import pytorch_lightning as pl
from pytorch_lightning.loggers import MLFlowLogger

import mlflow
from mlflow.tracking import MlflowClient

from config.cli_args import CLI_args, overwrite_config_items
from config.load_config import load_yaml_config, to_dict

from utils.helpers import *
from utils.mlflow_helpers import get_mlflow_parameters, get_mlflow_dataset_params

import os
import argparse
import pprint

# from typing import Namespace

from IPython import embed


def mlflow_startup(mlflow_config):
    
    '''
      Starts MLflow run, using
      
      mlflow_config: Namespace including run_id, experiment_name, run_name, artifact_location            
    
    '''
    
    mlflow.pytorch.autolog(log_models=True)
 
    if mlflow_config.tracking_uri is not None:
        mlflow.set_tracking_uri(mlflow_config.tracking_uri)
    
    try:
        exp_id = mlflow.create_experiment(mlflow_config.experiment_name, artifact_location=mlflow_config.artifact_location)
    except:
      # If the experiment already exists, we can just retrieve its ID
        experiment = mlflow.get_experiment_by_name(mlflow_config.experiment_name)
        exp_id = experiment.experiment
    run_info = {
        "run_id": trainer.logger.run_id,
        "experiment_id": exp_id,
        "run_name": mlflow_config.run_name,
        #"tags": config.additional_mlflow_tags
    }
    
    mlflow.start_run(**run_info)
    
        
def mlflow_log_additional_params(config):
    
    '''
    Log additional parameters, extracted from config
    '''
        
    mlflow_params = get_mlflow_parameters(config)
    mlflow_dataset_params = get_mlflow_dataset_params(config)
    mlflow_params.update(mlflow_dataset_params)
    mlflow.log_params(mlflow_params)    
        
        
def log_computational_graph(model, x):
    
    from torchviz import make_dot
    yhat = model(x)
    make_dot(yhat, params=dict(list(model.named_parameters()))).render("comp_graph_network", format="png")
    mlflow.log_figure("comp_graph_network.png")
        
        
###
def main(config, trainer_args, mlflow_config=None):

    '''
      config (Namespace):       
      trainer_args (Namespace):
      mlflow_config (Namespace):
      
      Example:
      
      
    '''

    dm, model, trainer = get_dm_model_trainer(config, trainer_args)
        
    if mlflow_config:
        mlflow_config.run_id = trainer.logger.run_id
        mlflow_startup(mlflow_config)             
        mlflow_log_additional_params(config)
                                       
    trainer.fit(model, datamodule=dm)
    trainer.test(datamodule=dm, ckpt_path='best') # Generates metrics for the full test dataset
    trainer.predict(ckpt_path='best', datamodule=dm) # Generates figures for a few samples

    mlflow.end_run()    


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Pytorch Trainer for Convolutional Mesh Autoencoders",
        argument_default=argparse.SUPPRESS
    )

    #to avoid a little bit of boilerplate
    for k, v in CLI_args.items():
        parser.add_argument(*k, **v)

    # adding arguments specific to the PyTorch Lightning trainer.
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
        config = ref_config
        pass

    #TOFIX: args contains other arguments that do not correspond to the trainer
    trainer_args = args


    config.log_computational_graph = args.log_computational_graph
    if args.disable_mlflow_logging:
        config.mlflow = None


    if config.mlflow:

        if config.mlflow.experiment_name is None:
            config.mlflow.experiment_name = "rbonazzola - Default"

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

    
    if hasattr(args, "only_decoder"):
        config.only_decoder = args.only_decoder
        if config.mlflow:
            config.mlflow.experiment_name = "rbonazzola - decoder only"
    elif hasattr(args, "only_encoder"):
        config.only_encoder = args.only_encoder
        if config.mlflow:
            config.mlflow.experiment_name = "rbonazzola - encoder only"

    #if config.only_decoder or config.only_encoder:
    d = config.dataset
    config.network_architecture.latent_dim_c = (d.parameters.l_max + 1) ** 2
    config.network_architecture.latent_dim_s = ((d.parameters.l_max + 1) ** 2) * d.parameters.freq_max
    
    if args.show_config or args.dry_run:
        pp = pprint.PrettyPrinter(indent=2, compact=True)
        pp.pprint(to_dict(config))
        if args.dry_run:
            exit()


    main(config, trainer_args, config.mlflow)
