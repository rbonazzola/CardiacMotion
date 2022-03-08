import os
import yaml
from argparse import Namespace
from IPython import embed


def is_yaml_file(x):
    if isinstance(x, str):
        return x.endswith("yaml") or x.endswith("yml")
    return False


def get_repo_rootdir():
    import shlex
    from subprocess import check_output
    repo_rootdir = check_output(shlex.split("git rev-parse --show-toplevel")).strip().decode('ascii')
    return repo_rootdir


def unfold_config(token, no_unfolding_for=[]):
    '''
    Parameters: 
      token: a recursive structure composed of a path to a yaml file or a dictionary composed of such structures.
      no_unfolding_for: a list of dict keys for which the yaml shouldn't be unfolded, and instead kept as a path
    Returns: A dictionary with all the yaml files replaces by their content.
    '''

    #
    if is_yaml_file(token):
        #TODO: COMMENT AND DOCUMENT THIS!!!
        yaml_file_base = token
        try:            
            yaml_dir = get_repo_rootdir()
            yaml_file = os.path.join(yaml_dir, yaml_file_base)
            token = yaml.safe_load(open(yaml_file))
        except FileNotFoundError:
            yaml_dir = os.path.join(get_repo_rootdir(), "config")
            yaml_file = os.path.join(yaml_dir, yaml_file_base)
            token = yaml.safe_load(open(yaml_file))

    if isinstance(token, dict):
        for k, v in token.items():
            if k not in no_unfolding_for:
                token[k] = unfold_config(v, no_unfolding_for)

    return token


def recursive_namespace(dd):
    '''
    Converts a (possibly nested) dictionary into a namespace.
    This allows for auto-completion
    '''
    for d in dd:
        has_any_dicts = False
        if isinstance(dd[d], dict):
            dd[d] = recursive_namespace(dd[d])
            has_any_dicts = True
    return Namespace(**dd)


def sanity_check(config):
    
    '''
    Perform sanity check on the configuration provided.
    '''

    pol_deg_dim = len(config.network_architecture.convolution.parameters.polynomial_degree)
    downsampling_factors_dim = len(config.network_architecture.pooling.parameters.downsampling_factors)
    
    try:
      n_channels_dim = len(config.network_architecture.convolution.channels)
    except:
      n_channels_dim = len(config.network_architecture.convolution.channels_enc)

    if not ((pol_deg_dim == downsampling_factors_dim) and (pol_deg_dim == n_channels_dim)):       
       raise ValueError(
          f"Dimensions of polynomial degrees, downsampling factors and number of channels should match \
          (but are {pol_deg_dim}, {downsampling_factors_dim} and {n_channels_dim}.)"
       )


def load_config(yaml_config_file, args=None):
    
    
    # config = yaml.safe_load(config)    
    config = unfold_config(yaml_config_file)    
        # I am using a namespace instead of a dictionary mainly because it enables auto-completion
    config = recursive_namespace(config)
    
    
    # The following parameters are meant to be lists of numbers, so they are parsed here from their string representation in the YAML file.
    config.network_architecture.convolution.parameters.polynomial_degree = \
    [int(x) for x in config.network_architecture.convolution.parameters.polynomial_degree.split()]
    
    config.network_architecture.pooling.parameters.downsampling_factors = \
    [int(x) for x in config.network_architecture.pooling.parameters.downsampling_factors.split()]
    


    if hasattr(config.network_architecture.convolution, "channels"):
      config.network_architecture.convolution.channels = \
      [int(x) for x in config.network_architecture.convolution.channels.split()]

    if hasattr(config.network_architecture.convolution, "channels_enc"):
      config.network_architecture.convolution.channels_enc = \
      [int(x) for x in config.network_architecture.convolution.channels_enc.split()]
  
    if hasattr(config.network_architecture.convolution, "channels_dec_c"):
      config.network_architecture.convolution.channels_dec_c = \
      [int(x) for x in config.network_architecture.convolution.channels_dec_c.split()]

    if hasattr(config.network_architecture.convolution, "channels_dec_s"):
      config.network_architecture.convolution.channels_dec_s = \
      [int(x) for x in config.network_architecture.convolution.channels_dec_s.split()]

    sanity_check(config)

    if args is not None:

        if args.w_kl is not None:
            config.loss.regularization.weight = args.w_kl
    
        if args.latent_dim_c is not None:
            config.network_architecture.latent_dim_c = args.latent_dim_c
        
        if args.latent_dim_s is not None:
            config.network_architecture.latent_dim_s = args.latent_dim_s
    
        if args.batch_size is not None:
            config.optimizer.batch_size = args.batch_size
        
        if args.no_phase_input is not None:
            config.network_architecture.phase_input = not args.no_phase_input
        
        if args.learning_rate is not None:
            config.optimizer.parameters.lr = args.learning_rate

    return config
