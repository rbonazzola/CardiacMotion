import argparse
from config.load_config import load_yaml_config, to_dict, flatten_dict, rsetattr, rgetattr
from copy import deepcopy

class ArgumentAction(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        rsetattr(namespace, self.dest, values)

class kwargs_append_action(argparse.Action):
    """
    argparse action to split an argument into KEY=VALUE form
    on append to a dictionary.
    """

    def __call__(self, parser, args, values, option_string=None):
        try:
            d = dict(map(lambda x: x.split('='),values))
        except ValueError as ex:
            raise argparse.ArgumentError(self, f"Could not parse argument \"{values}\" as k1=v1 k2=v2 ... format")
        setattr(args, self.dest, d)

network_architecture_args = {
    ("--n_channels",): {
        "help": "Number of channels (feature maps). If the rest of the --n_channels_* arguments are not provided, it will assign these numbers to the encoder, content decoder and style decoder.",
        "nargs": "+", "type": int,
        "action": ArgumentAction,
        "dest": "config.network_architecture.convolution.channels"
    },
    ("--reduction_factors",): {
        "help": "Decimation factors for the mesh",
        "nargs": "+", "type": int,
        "action": ArgumentAction,
        "dest": "config.network_architecture.pooling.parameters.downsampling_factors"},
    ("--n_channels_enc",): {
        "help": "Number of channels (feature maps) in the encoder, from input to the most hidden layer.",
        "nargs": "+", "type": int,
        "action": ArgumentAction,
        "dest": "config.network_architecture.convolution.channels_enc"},
    ("--n_channels_dec_c",): {
        "help": "Number of channels (feature maps) in the content decoder, from the most hidden layer to the output.",
        "nargs": "+", "type": int,
        "action": ArgumentAction,
        "dest": "config.network_architecture.convolution.channels_dec_c"},
    ("--n_channels_dec_s",): {
        "help": "Number of channels (feature maps) in the style decoder, from the most hidden layer to the output.",
        "nargs": "+", "type": int,
        "action": ArgumentAction,
        "dest": "config.network_architecture.convolution.channels_dec_s"},
    ("--latent_dim_c",): {
        "help": "Dimension of the content part of the latent space",
        "type": int,
        "action": ArgumentAction,
        "dest": "config.network_architecture.latent_dim_c"},
    ("--latent_dim_s",): {
        "help": "Dimension of the style part of the latent space",
        "type": int,
        "action": ArgumentAction,
        "dest": "config.network_architecture.latent_dim_s"},
    ("--activation_function",): {
        "help": "Activation functions to be used",
        "nargs": "+", "type": str,
        "action": ArgumentAction,
        "dest": "config.network_architecture.activation_function"},
    ("--polynomial_degree",): {
        "help": "Chebyshev polynomial degree",
        "nargs": "+", "type": int,
        "action": ArgumentAction,
        "dest": "config.network_architecture.convolution.parameters.polynomial_degree"},
    ("--z_aggr_function",): {
        "help": "Temporal aggregation method",
        "type": str,
        "action": ArgumentAction,
        "dest": "config.network_architecture.z_aggr_function"},
    ("--z_aggr_function",): {
        "help": "Temporal aggregation method",
        "type": str,
        "action": ArgumentAction,
        "dest": "config.network_architecture.z_aggr_function"},
    ("--transformer_d_model",): {
        "help": "Hidden dimension of the transformer z_aggr_function (ignored unless --z_aggr_function transformer)",
        "type": int,
        "action": ArgumentAction,
        "dest": "config.network_architecture.transformer.d_model"},
    ("--transformer_n_heads",): {
        "help": "Number of attention heads of the transformer z_aggr_function",
        "type": int,
        "action": ArgumentAction,
        "dest": "config.network_architecture.transformer.n_heads"},
    ("--transformer_n_layers",): {
        "help": "Number of transformer encoder layers of the transformer z_aggr_function",
        "type": int,
        "action": ArgumentAction,
        "dest": "config.network_architecture.transformer.n_layers"},
    ("--transformer_d_ff",): {
        "help": "Feed-forward hidden dimension of the transformer z_aggr_function",
        "type": int,
        "action": ArgumentAction,
        "dest": "config.network_architecture.transformer.d_ff"},
    ("--transformer_dropout",): {
        "help": "Dropout of the transformer z_aggr_function",
        "type": float,
        "action": ArgumentAction,
        "dest": "config.network_architecture.transformer.dropout"},
    ("--only_decoder",): {
        "help": "Flag to run only the decoder",
        "action": "store_true"},
    ("--only_encoder",): {
        "help": "Flag to run only the encoder",
        "action": "store_true"},

    #("--phase_input" ): {
    #    "help": "If this flag is set, the phase embedding is not applied to the input mesh coordinates.",
    #    "default": True,
    #    "dest": "config.network_architecture.phase_input",
    #    "action": argparse.BooleanOptionalAction}
}

loss_args = {
    ("--reconstruction_loss_type",): {
        "help": "Type of reconstruction loss",
        "dest": "config.loss.reconstruction_c.type",
        "type": str,
        "action": ArgumentAction},
    ("--w_kl",): {
        "help": "weight of KL term",
        "dest": "config.loss.regularization.weight",
        "type": float,
        "action": ArgumentAction},
    ("--w_s",): {
        "help": "weight of the \"style\" reconstruction term in the lost function "
                "(the target/final value when --w_s_ramp_epochs > 0).",
        "dest": "config.loss.reconstruction_s.weight",
        "type": float,
        "action": ArgumentAction},
    ("--w_s_start",): {
        "help": "starting weight for the \"style\" term while content (recon_loss_c) hasn't "
                "converged yet (default 0.1). Only takes effect when --w_s_ramp_epochs > 0.",
        "dest": "config.loss.reconstruction_s.start_weight",
        "type": float,
        "action": ArgumentAction},
    ("--w_s_ramp_epochs",): {
        "help": "epochs over which w_s ramps linearly from --w_s_start to --w_s, "
                "*starting once content has converged* (see --w_s_content_patience/"
                "--w_s_content_min_delta), not from epoch 0. "
                "0 (default) disables the ramp -- w_s is constant from epoch 0, matching prior behavior.",
        "dest": "config.loss.reconstruction_s.ramp_epochs",
        "type": int,
        "action": ArgumentAction},
    ("--w_s_content_patience",): {
        "help": "epochs of val_recon_loss_c improving by less than --w_s_content_min_delta "
                "(relative) before content is considered converged and the w_s ramp begins. "
                "Only matters when --w_s_ramp_epochs > 0.",
        "dest": "config.loss.reconstruction_s.content_patience",
        "type": int,
        "action": ArgumentAction},
    ("--w_s_content_min_delta",): {
        "help": "relative improvement in val_recon_loss_c (vs. its best-so-far value) below "
                "which an epoch counts towards --w_s_content_patience. Only matters when "
                "--w_s_ramp_epochs > 0.",
        "dest": "config.loss.reconstruction_s.content_min_delta",
        "type": float,
        "action": ArgumentAction},
    ("--w_translation",): {
        "help": "weight of the rigid-translation component of recon_loss_s (per-frame centroid "
                "MSE). Default 1.0, same as --w_shape -- together they reproduce plain recon_loss_s "
                "exactly (the split is an exact decomposition, not an approximation). Raise this "
                "relative to --w_shape to prioritize getting each frame's global position right.",
        "dest": "config.loss.reconstruction_s.translation_weight",
        "type": float,
        "action": ArgumentAction},
    ("--w_shape",): {
        "help": "weight of the translation-invariant shape component of recon_loss_s (MSE after "
                "centering both meshes on their own centroid). Default 1.0, see --w_translation.",
        "dest": "config.loss.reconstruction_s.shape_weight",
        "type": float,
        "action": ArgumentAction},
    ("--w_smooth",): {
        "help": "target weight of the Laplacian smoothness regularizer on the reconstructed mesh "
                "(penalizes each predicted vertex for deviating from its neighbors' average; "
                "0 disables it, matching prior behavior).",
        "dest": "config.loss.smoothness.weight",
        "type": float,
        "action": ArgumentAction},
    ("--w_smooth_start",): {
        "help": "starting weight for the smoothness term while style (recon_loss_s) hasn't "
                "converged yet (default 0 -- no smoothing pressure until reconstruction itself "
                "has stabilized). Only takes effect when --w_smooth_ramp_epochs > 0.",
        "dest": "config.loss.smoothness.start_weight",
        "type": float,
        "action": ArgumentAction},
    ("--w_smooth_ramp_epochs",): {
        "help": "epochs over which w_smooth ramps linearly from --w_smooth_start to --w_smooth, "
                "starting once style (recon_loss_s) has converged (see --w_smooth_style_patience/"
                "--w_smooth_style_min_delta) -- i.e. smoothing only kicks in once the network has "
                "learned to reconstruct the motion, so it polishes instead of fighting early "
                "training. 0 (default) disables the ramp -- w_smooth is constant from epoch 0.",
        "dest": "config.loss.smoothness.ramp_epochs",
        "type": int,
        "action": ArgumentAction},
    ("--w_smooth_style_patience",): {
        "help": "epochs of val_recon_loss_s improving by less than --w_smooth_style_min_delta "
                "(relative) before style is considered converged and the w_smooth ramp begins. "
                "Only matters when --w_smooth_ramp_epochs > 0.",
        "dest": "config.loss.smoothness.style_patience",
        "type": int,
        "action": ArgumentAction},
    ("--w_smooth_style_min_delta",): {
        "help": "relative improvement in val_recon_loss_s (vs. its best-so-far value) below "
                "which an epoch counts towards --w_smooth_style_patience. Only matters when "
                "--w_smooth_ramp_epochs > 0.",
        "dest": "config.loss.smoothness.style_min_delta",
        "type": float,
        "action": ArgumentAction},
    ("--smooth_mask_percentile",): {
        "help": "excludes the roughest (100 - this)%% of vertices in the population TEMPLATE's "
                "own Laplacian magnitude from the smoothness penalty -- e.g. 90 skips the "
                "roughest 10%% (partition cut boundaries, valve annuli, etc. that are genuinely "
                "not smooth in real anatomy) so the regularizer doesn't fight them. "
                "100 (default) masks nothing -- every vertex is penalized.",
        "dest": "config.loss.smoothness.mask_percentile",
        "type": float,
        "action": ArgumentAction},
}

dataset_args = {
    ("--dataset.amplitude_static_max",): {
        "help": "" ,
        "dest": "config.dataset.parameters.amplitude_static_max" , 
        "type": float,
        "action": ArgumentAction},
    ("--dataset.amplitude_dynamic_max",): {
        "help": "",
        "dest": "config.dataset.parameters.amplitude_dynamic_max" , 
        "type": float,
        "action": ArgumentAction},
    ("--dataset.N_subjects",): {
        "help": "",
        "dest": "config.dataset.parameters.N" , 
        "type": int,
        "action": ArgumentAction},
    ("--dataset.N_timeframes",): {
        "help": "",
        "dest": "config.dataset.parameters.T" , 
        "type": int,
        "action": ArgumentAction},
    ("--dataset.freq_max",): {
        "help": "",
        "dest": "config.dataset.parameters.freq_max" , 
        "type": int,
        "action": ArgumentAction},
    ("--dataset.l_max",): {
        "help": "",
        "dest": "config.dataset.parameters.l_max" , 
        "type": int,
        "action": ArgumentAction},
    ("--dataset.mesh_resolution",): {
        "help": "",
        "dest": "config.dataset.parameters.mesh_resolution" , 
        "type": int,
        "action": ArgumentAction},
    ("--dataset.center_around_mean",): {
        "help": "Not working! Always sets this value to True.",
        "dest": "config.dataset.preprocessing.center_around_mean" , 
        "type": bool,
        "action": ArgumentAction}
}

training_args = {
    ("--learning_rate", "-lr",): {
        "help": "Learning rate",
        "dest": "config.optimizer.parameters.lr",
        "type": float,
        "action": ArgumentAction},
    ("--batch_size",): {
        "help": "Training batch size. If provided will overwrite the batch size from the configuration file.",
        "dest": "config.batch_size",
        "type": int,
        "action": ArgumentAction},
    ("--partition_lengths", "--partition-lengths"): {
        "nargs":"+", "help": "List of two [or three] integers (floats) representing the number of samples (fraction of samples) to be used for training, validation [and testing].",
        "dest": "config.sample_sizes",
        "action": ArgumentAction},
}

mlflow_args = {
    ("--disable_mlflow_logging", "--no_mlflow"): {
        "dest": "disable_mlflow_logging",
        "help": "Set this flag if you don't want to log the run's data to MLflow.",
        "default": False,
        "action": "store_true"},
    ("--mlflow_experiment",): {
        "help": "MLflow experiment's name",
        "dest": "config.mlflow.experiment_name",
        "action": ArgumentAction
    },
    ("--additional_mlflow_params",): {
        "nargs": '+',
        "required": False,
        "dest": "config.additional_mlflow_params",
        "action": kwargs_append_action,
        "metavar": "KEY=VALUE",
        "help": "Add additional key/value params to MLflow."
    },
    ("--additional_mlflow_tags",): {
        "nargs": '+',
        "required": False,
        "dest": "config.additional_mlflow_tags",
        "action": kwargs_append_action,
        "metavar": "KEY=VALUE",
        "help": "Add additional key/value tags to MLflow."
    }
}

#   ("--mlflow_config",): {
#       "action": LoadYamlConfig,
#       "help": "YAML configuration file containing information to log model information to MLflow.",
#       "dest": "config.mlflow"},

#########################################################################
#### Put all the arguments together
#########################################################################

CLI_args = {
    ("-c", "--conf",): {
        "help": "Path of a YAML configuration file to be used as a reference configuration.",
        "default": "config_files/config_folded_c_and_s.yaml",
        "dest": "yaml_config_file"
    },
    ** network_architecture_args,
    ** loss_args,
    ** training_args,
    ** dataset_args,
    ** mlflow_args,
    ("--show_config", "--show-config"): {
        "dest": "show_config",
        "default": False,
        "action": "store_true",
        "help": "Display run's configuration"
    },
    ("--dry-run", "--dry_run", "--dryrun"): {
        "dest": "dry_run",
        "default": False,
        "action": "store_true",
        "help": "Dry run: just prints out the parameters of the execution but performs no training.",
    },
    ("--log_computational_graph",): {
        "default": False,
        "action": "store_true",
        "help": "If True, will log the computational graph as an artifact (not fully functional due to limitations of the torchviz library)"
    }
}


def overwrite_config_items(ref_config, config_to_replace):
    '''
    params:
    :: ref_config ::
    :: config_to_replace ::
    '''

    config = deepcopy(ref_config)
    for k, v in flatten_dict(to_dict(config_to_replace)).items():
        rsetattr(config, k, v)

    return config
