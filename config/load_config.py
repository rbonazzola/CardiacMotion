import yaml
from argparse import Namespace

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
    n_channels_dim = len(config.network_architecture.convolution.channels)

    if not ((pol_deg_dim == downsampling_factors_dim) and (pol_deg_dim == n_channels_dim)):       
       raise ValueError(
          f"Dimensions of polynomial degrees, downsampling factors and number of channels should match \
          (but are {pol_deg_dim}, {downsampling_factors_dim} and {n_channels_dim}.)"
       )


def load_config(yaml_config_file, args):
    
    
    with open(yaml_config_file) as config:
        config = yaml.safe_load(config)    
        # I am using a namespace instead of a dictionary mainly because it enables auto-completion
        config = recursive_namespace(config)
    
    
    # The following parameters are meant to be lists of numbers, so they are parsed here from their string representation in the YAML file.
    config.network_architecture.convolution.parameters.polynomial_degree = \
    [int(x) for x in config.network_architecture.convolution.parameters.polynomial_degree.split()]
    
    config.network_architecture.pooling.parameters.downsampling_factors = \
    [int(x) for x in config.network_architecture.pooling.parameters.downsampling_factors.split()]
    
    config.network_architecture.convolution.channels = \
    [int(x) for x in config.network_architecture.convolution.channels.split()]
  
    sanity_check(config)

    if args.w_kl is not None:
        config.regularization_loss.weight = args.w_kl

    if args.latent_dim is not None:
        config.network_architecture.latent_dim = args.latent_dim

    if args.batch_size is not None:
        config.optimizer.batch_size = args.batch_size

    return config
